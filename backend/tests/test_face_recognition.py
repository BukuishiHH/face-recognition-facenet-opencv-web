"""
Step 2-2 人脸识别特征提取与比对模块单元测试
被测模块: backend/face_recognition.py
说明: 需要 FaceNet(casia-webface) 权重, 已缓存于 ~/.cache/torch/checkpoints;
      特征向量相关用例在 CPU / CUDA 下均可运行(自动降级)。
"""

import cv2
import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from face_detection import FaceDetection
from face_recognition import FaceRecognition

FEATURE_DIM = 512
DEFAULT_THRESHOLD = 0.6


@pytest.fixture(scope='module')
def face_roi(imread, one_face_path):
    """从单人脸样例中裁剪出的 BGR 人脸 ROI"""
    img = imread(one_face_path)
    detector = FaceDetection()
    faces = detector.detect_faces(img)
    assert faces, "样例图未检测到人脸, 无法测试特征模块"
    return detector.crop_face_rois(img, faces)[0][0]


@pytest.fixture(scope='module')
def cached_feature(face_roi):
    """模块级缓存: 同一样例的特征只提取一次, 控制推理耗时"""
    return FaceRecognition.extract_feature(face_roi)


# --------------------------------------- 模型加载 ---------------------------------------
class TestModelLoading:

    def test_model_loaded(self, cached_feature):
        assert FaceRecognition._model is not None
        assert FaceRecognition._device in ('cpu', 'cuda')

    def test_model_reused_singleton(self):
        first = FaceRecognition._get_model()
        second = FaceRecognition._get_model()
        assert first is second                             # 全局复用, 不重复加载


# --------------------------------------- 特征提取 ---------------------------------------
class TestExtractFeature:

    def test_shape_and_dtype(self, cached_feature):
        assert cached_feature.shape == (1, FEATURE_DIM)
        assert cached_feature.dtype == np.float32

    def test_l2_normalized(self, cached_feature):
        """L2 归一化后模长恒为 1, 使点积 = 余弦相似度"""
        assert np.isclose(np.linalg.norm(cached_feature), 1.0, atol=1e-5)

    def test_deterministic_for_same_input(self, face_roi, cached_feature):
        again = FaceRecognition.extract_feature(face_roi)
        assert np.allclose(again, cached_feature, atol=1e-5)

    def test_same_person_variant_high_similarity(self, face_roi, cached_feature):
        """同人不同照片(亮度/缩放扰动)相似度应高于识别阈值"""
        variant = cv2.convertScaleAbs(face_roi, alpha=1.08, beta=12)
        variant = cv2.resize(variant, (face_roi.shape[1], face_roi.shape[0]))
        variant_feature = FaceRecognition.extract_feature(variant)
        score = float(cosine_similarity(cached_feature, variant_feature)[0][0])
        assert score >= DEFAULT_THRESHOLD

    def test_accepts_image_path_string(self, one_face_path):
        feature = FaceRecognition.extract_feature(one_face_path)
        assert feature is not None
        assert feature.shape == (1, FEATURE_DIM)

    def test_missing_path_returns_none(self):
        assert FaceRecognition.extract_feature('no_such_file.jpg') is None


# --------------------------------------- 身份比对 ---------------------------------------
class TestRecognizeFace:

    def test_same_person_recognized(self, isolated_db, face_roi, cached_feature):
        isolated_db.save_feature(cached_feature, '测试员')
        result = FaceRecognition.recognize_face(face_roi)
        assert result is not None
        assert result['name'] == '测试员'
        assert result['similarity'] >= DEFAULT_THRESHOLD

    def test_empty_library_returns_none(self, isolated_db, face_roi):
        assert FaceRecognition.recognize_face(face_roi) is None

    def test_threshold_gate_blocks_match(self, isolated_db, face_roi, cached_feature):
        """阈值机制: 阈值设为 1.01(不可达)时任何相似度都无法通过"""
        isolated_db.save_feature(cached_feature, '测试员')
        assert FaceRecognition.recognize_face(face_roi, threshold=1.01) is None

    def test_zero_threshold_always_matches(self, isolated_db, face_roi, cached_feature):
        isolated_db.save_feature(cached_feature, '测试员')
        result = FaceRecognition.recognize_face(face_roi, threshold=0.0)
        assert result is not None and result['name'] == '测试员'

    def test_most_similar_record_selected(self, isolated_db, face_roi, cached_feature):
        """多人库中应选出最相似的一条"""
        rng = np.random.default_rng(0)
        other = rng.random((1, FEATURE_DIM)).astype(np.float32)
        other = other / np.linalg.norm(other)              # 归一化, 模拟另一身份
        isolated_db.save_feature(other, '其他人')
        isolated_db.save_feature(cached_feature, '测试员')

        result = FaceRecognition.recognize_face(face_roi)
        assert result is not None
        assert result['name'] == '测试员'
