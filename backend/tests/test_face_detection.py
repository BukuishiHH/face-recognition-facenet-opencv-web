"""
Step 2-1 人脸检测模块单元测试
被测模块: backend/face_detection.py
"""

import cv2
import numpy as np
import pytest

from face_detection import FaceDetection


@pytest.fixture(scope='module')
def detector():
    """模块级复用: Haar 模型只加载一次"""
    return FaceDetection()


# --------------------------------------- 模型加载 ---------------------------------------
class TestModelLoading:

    def test_model_loaded(self, detector):
        assert detector.face_engine is not None
        assert not detector.face_engine.empty()

    def test_invalid_model_raises_value_error(self):
        with pytest.raises(ValueError):
            FaceDetection(model_name='no_such_model.xml')


# --------------------------------------- 图像预处理 ---------------------------------------
class TestPreprocess:

    def test_output_is_single_channel_gray(self, imread, one_face_path):
        img = imread(one_face_path)
        gray = FaceDetection.preprocess(img)
        assert gray.ndim == 2                              # 单通道
        assert gray.shape == img.shape[:2]                 # 尺寸不变
        assert gray.dtype == np.uint8


# --------------------------------------- 人脸检测 ---------------------------------------
class TestDetectFaces:

    def test_single_face_image_returns_one_box(self, detector, imread, one_face_path):
        img = imread(one_face_path)
        faces = detector.detect_faces(img)
        assert len(faces) == 1

    def test_multi_face_image_returns_multiple_boxes(self, detector, imread, multi_face_path):
        img = imread(multi_face_path)
        faces = detector.detect_faces(img)
        assert len(faces) >= 2

    def test_no_face_image_returns_empty_list(self, detector, imread, no_face_path):
        img = imread(no_face_path)
        faces = detector.detect_faces(img)
        assert isinstance(faces, list)
        assert faces == []                                 # 无脸不崩溃, 返回空列表

    def test_return_type_is_list_of_int_tuples(self, detector, imread, one_face_path):
        img = imread(one_face_path)
        for face in detector.detect_faces(img):
            assert isinstance(face, tuple) and len(face) == 4
            assert all(isinstance(v, (int, np.integer)) for v in face)

    def test_non_ndarray_raises_type_error(self, detector):
        with pytest.raises(TypeError):
            detector.detect_faces('not an ndarray')

    def test_boxes_within_bounds_and_min_size(self, detector, imread, one_face_path):
        """边界校验: 返回框必须不越界且不小于 10x10"""
        img = imread(one_face_path)
        height, width = img.shape[:2]
        for (x, y, w, h) in detector.detect_faces(img):
            assert x >= 0 and y >= 0
            assert x + w <= width and y + h <= height
            assert w >= 10 and h >= 10


# --------------------------------------- ROI 裁剪 ---------------------------------------
class TestCropFaceRois:

    def test_roi_shape_matches_box(self, detector, imread, one_face_path):
        img = imread(one_face_path)
        faces = detector.detect_faces(img)
        rois = detector.crop_face_rois(img, faces)
        assert len(rois) == len(faces)
        for roi, (x, y, w, h) in rois:
            assert roi.shape[0] == h
            assert roi.shape[1] == w

    def test_empty_faces_returns_empty_rois(self, detector, imread, one_face_path):
        img = imread(one_face_path)
        assert detector.crop_face_rois(img, []) == []
