"""
Step 3-3 人脸识别推理接口测试

覆盖:
  1. 图片识别: 命中、多脸、空库、无人脸、坏图、超限、阈值、标注图、大图坐标;
  2. 帧识别: dataURL / 裸 base64、空库、无人脸、非法/超大帧、阈值优先级;
  3. 批量比对只取库一次、与 CLI recognize_face 数值一致;
  4. 并发保护: 帧接口忙时 1008/429, 图片接口排队超时 1008;
  5. OpenAPI 契约(必填字段、默认阈值)。

说明: 每个用例通过 isolated_db 使用临时特征库, 不触碰 backend/face_features.db。
"""

import base64
import os
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_server
from face_detection import FaceDetection
from face_recognition import FaceRecognition

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_THRESHOLD = 0.6
DINGZHEN_PATH = os.path.join(BACKEND_DIR, 'sample_faces', 'dingzhen', 'dingzhen_1.jpg')
SUN_PATH = os.path.join(BACKEND_DIR, 'sample_faces', 'sunxiaochuan', 'sunxiaochuan_1.jpg')


# --------------------------------------- 夹具与辅助函数 ---------------------------------------
@pytest.fixture()
def api_client(isolated_db):
    """启动完整生命周期(检测器 + FaceNet + 临时特征库)的测试客户端"""
    with TestClient(api_server.app) as client:
        yield client


def _feature_from_image(image):
    """对样例图检测第一张脸并提取特征(用于预置特征库)"""
    faces = api_server.app.state.detector.detect_faces(image)
    assert faces, '预置特征失败: 样例图未检测到人脸'
    roi = FaceDetection.crop_face_rois(image, faces)[0][0]
    feature = FaceRecognition.extract_feature(roi)
    assert feature is not None, '预置特征失败: extract_feature 返回 None'
    return feature


@pytest.fixture()
def seeded_name(api_client, isolated_db, one_face_path, imread):
    """把 one_face.jpg 的人脸以固定姓名写入隔离库"""
    name = '样本甲'
    assert isolated_db.save_feature(_feature_from_image(imread(one_face_path)), name) is not None
    return name


@pytest.fixture()
def seeded_dingzhen(api_client, isolated_db, imread):
    """把 dingzhen_1.jpg 的人脸写入隔离库, 用于多脸命中/未知脸对照"""
    if not os.path.exists(DINGZHEN_PATH):
        pytest.skip(f'缺少样例: {DINGZHEN_PATH}')
    name = '丁真'
    assert isolated_db.save_feature(_feature_from_image(imread(DINGZHEN_PATH)), name) is not None
    return name


def _post_image(client, path, threshold=DEFAULT_THRESHOLD, save_result=False):
    with open(path, 'rb') as fp:
        raw = fp.read()
    return _post_image_raw(client, raw, threshold=threshold, save_result=save_result)


def _post_image_raw(client, raw, threshold=DEFAULT_THRESHOLD, save_result=False,
                    filename='image.jpg'):
    return client.post(
        '/api/recognize/image',
        files={'file': (filename, raw, 'image/jpeg')},
        data={'threshold': str(threshold), 'save_result': str(save_result).lower()},
    )


def _post_image_array(client, image, threshold=DEFAULT_THRESHOLD, save_result=False):
    ok_flag, buffer = cv2.imencode('.jpg', image)
    assert ok_flag
    return _post_image_raw(client, buffer.tobytes(), threshold=threshold,
                           save_result=save_result)


def _data_url(path):
    with open(path, 'rb') as fp:
        return 'data:image/jpeg;base64,' + base64.b64encode(fp.read()).decode()


def _raw_base64(path):
    with open(path, 'rb') as fp:
        return base64.b64encode(fp.read()).decode()


def _post_frame(client, image_base64, threshold=DEFAULT_THRESHOLD):
    return client.post('/api/recognize/frame',
                       json={'image_base64': image_base64, 'threshold': threshold})


def _multi_person_image(imread):
    """拼接丁真与孙笑川两张不同人脸, 用作多脸场景(检测顺序不影响断言)"""
    if not (os.path.exists(DINGZHEN_PATH) and os.path.exists(SUN_PATH)):
        pytest.skip('缺少多脸对照样例(dingzhen_1 / sunxiaochuan_1)')
    first = cv2.resize(imread(DINGZHEN_PATH), (512, 512), interpolation=cv2.INTER_AREA)
    second = cv2.resize(imread(SUN_PATH), (512, 512), interpolation=cv2.INTER_AREA)
    return np.hstack([first, second])


# --------------------------------------- 图片识别 ---------------------------------------
class TestRecognizeImage:

    def test_matched(self, api_client, seeded_name, one_face_path):
        payload = _post_image(api_client, one_face_path).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is False
        assert data['total_faces'] >= 1
        assert data['matched_count'] >= 1
        hit = [face for face in data['faces'] if face['matched']]
        assert seeded_name in [face['name'] for face in hit]
        assert hit[0]['similarity'] >= DEFAULT_THRESHOLD
        assert data['elapsed_ms'] >= 0
        assert data['result_image_url'] is None               # 未开启 save_result

    def test_similarity_rounded_in_json(self, api_client, seeded_name, one_face_path, imread):
        code, _, data = api_server._recognize_bgr(imread(one_face_path), DEFAULT_THRESHOLD)
        assert code == 0
        raw_similarity = data.faces[0].similarity

        payload = _post_image(api_client, one_face_path).json()
        api_similarity = payload['data']['faces'][0]['similarity']
        assert api_similarity == round(raw_similarity, 4)

    def test_multi_face_unknown(self, api_client, seeded_dingzhen, imread):
        payload = _post_image_array(api_client, _multi_person_image(imread)).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['total_faces'] >= 2
        assert 1 <= data['matched_count'] < data['total_faces']
        matched = [face for face in data['faces'] if face['matched']]
        unknown = [face for face in data['faces'] if not face['matched']]
        assert seeded_dingzhen in [face['name'] for face in matched]
        assert all(face['name'] is None for face in unknown)

    def test_empty_library_keeps_boxes_and_skips_features(
            self, api_client, isolated_db, one_face_path, monkeypatch):
        def _unexpected(*_args, **_kwargs):
            pytest.fail('空库不应调用 extract_feature')

        monkeypatch.setattr(FaceRecognition, 'extract_feature', staticmethod(_unexpected))
        payload = _post_image(api_client, one_face_path).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is True
        assert data['total_faces'] >= 1
        assert data['matched_count'] == 0
        assert all(face['name'] is None for face in data['faces'])
        assert all(face['similarity'] == 0.0 for face in data['faces'])
        assert all(face['box']['w'] > 0 and face['box']['h'] > 0 for face in data['faces'])

    def test_feature_failure_marks_unmatched(self, api_client, seeded_name,
                                             one_face_path, monkeypatch):
        monkeypatch.setattr(FaceRecognition, 'extract_feature',
                            staticmethod(lambda _roi: None))
        payload = _post_image(api_client, one_face_path).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is False
        assert data['matched_count'] == 0
        assert all(face['matched'] is False and face['name'] is None
                   for face in data['faces'])
        assert all(face['similarity'] == 0.0 for face in data['faces'])

    def test_no_face(self, api_client, isolated_db, no_face_path):
        payload = _post_image(api_client, no_face_path).json()
        assert payload['code'] == 1002
        assert payload['data'] is None

    @pytest.mark.parametrize('raw', [b'', b'not an image'])
    def test_bad_file(self, api_client, isolated_db, raw):
        payload = _post_image_raw(api_client, raw).json()
        assert payload['code'] == 1001

    def test_oversize_rejected_before_inference(
            self, api_client, isolated_db, monkeypatch):
        def _unexpected(*_args, **_kwargs):
            pytest.fail('超大文件不应进入推理')

        monkeypatch.setattr(api_server, '_recognize_bgr', _unexpected)
        raw = b'0' * (api_server.MAX_UPLOAD_BYTES + 1)
        payload = _post_image_raw(api_client, raw).json()
        assert payload['code'] == 1001

    @pytest.mark.parametrize('threshold', [0.1, 0.99])
    def test_threshold_out_of_range(self, api_client, isolated_db, one_face_path, threshold):
        payload = _post_image(api_client, one_face_path, threshold=threshold).json()
        assert payload['code'] == 1001
        assert '阈值' in payload['message']

    @pytest.mark.parametrize('threshold', [0.3, 0.95])
    def test_threshold_boundaries_accepted(self, api_client, isolated_db,
                                           one_face_path, threshold):
        payload = _post_image(api_client, one_face_path, threshold=threshold).json()
        assert payload['code'] == 0

    def test_threshold_non_numeric(self, api_client, isolated_db, one_face_path):
        resp = _post_image(api_client, one_face_path, threshold='abc')
        assert resp.status_code == 422
        assert resp.json()['code'] == 422

    def test_save_result_image(self, api_client, seeded_name, one_face_path):
        payload = _post_image(api_client, one_face_path, save_result=True).json()
        url = payload['data']['result_image_url']
        assert url.startswith('/static/') and url.endswith('.jpg')

        saved = os.path.join(api_server.RESULT_DIR, os.path.basename(url))
        try:
            assert os.path.exists(saved)
            resp = api_client.get(url)
            assert resp.status_code == 200
            assert resp.headers['content-type'].startswith('image/jpeg')
            assert len(resp.content) > 0
        finally:
            if os.path.exists(saved):
                os.remove(saved)

    def test_large_image_box_scaled(self, api_client, isolated_db, one_face_path, imread):
        """长边 > 1280 时检测在缩略图进行, box 必须乘回 1/scale 还原到原图"""
        big = cv2.resize(imread(one_face_path), None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)
        ok_flag, buffer = cv2.imencode('.jpg', big)
        assert ok_flag
        uploaded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)     # 路由实际解码后的图像
        payload = _post_image_raw(api_client, buffer.tobytes()).json()
        assert payload['code'] == 0

        scaled, scale = api_server.limit_max_side(uploaded)
        raw_faces = api_server.app.state.detector.detect_faces(scaled)
        assert raw_faces
        inv = 1.0 / scale
        expected = [(int(round(x * inv)), int(round(y * inv)),
                     int(round(w * inv)), int(round(h * inv)))
                    for x, y, w, h in raw_faces]
        actual = [(face['box']['x'], face['box']['y'],
                   face['box']['w'], face['box']['h'])
                  for face in payload['data']['faces']]
        assert actual == expected
        assert all(x + w <= uploaded.shape[1] and y + h <= uploaded.shape[0]
                   for x, y, w, h in actual)

    def test_batch_compare_reads_library_once(self, api_client, isolated_db,
                                              seeded_dingzhen, imread, monkeypatch):
        calls = []
        original = isolated_db.get_all_features

        def _counted():
            calls.append(1)
            return original()

        monkeypatch.setattr(isolated_db, 'get_all_features', _counted)
        payload = _post_image_array(api_client, _multi_person_image(imread)).json()

        assert payload['code'] == 0
        assert payload['data']['total_faces'] >= 2
        assert len(calls) == 1

    def test_similarity_consistent_with_cli(self, api_client, seeded_name,
                                            one_face_path, imread):
        """接口批量点积与 CLI cosine_similarity 对同一张脸的原始相似度误差 < 1e-6"""
        image = imread(one_face_path)
        code, _, data = api_server._recognize_bgr(image, DEFAULT_THRESHOLD)
        assert code == 0 and data.matched_count >= 1

        faces = api_server.app.state.detector.detect_faces(image)
        roi = FaceDetection.crop_face_rois(image, faces)[0][0]
        cli_result = FaceRecognition.recognize_face(roi, threshold=DEFAULT_THRESHOLD)
        assert cli_result is not None
        assert cli_result['name'] == seeded_name
        assert abs(data.faces[0].similarity - cli_result['similarity']) < 1e-6




# --------------------------------------- 摄像头帧识别 ---------------------------------------
class TestRecognizeFrame:

    def test_data_url_matched(self, api_client, seeded_name, one_face_path):
        payload = _post_frame(api_client, _data_url(one_face_path)).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is False
        assert data['total_faces'] >= 1
        assert data['matched_count'] >= 1
        assert data['elapsed_ms'] >= 0
        assert data['result_image_url'] is None               # 帧接口固定不落盘

    def test_raw_base64_matched(self, api_client, seeded_name, one_face_path):
        payload = _post_frame(api_client, _raw_base64(one_face_path)).json()
        assert payload['code'] == 0
        assert payload['data']['matched_count'] >= 1

    def test_empty_library(self, api_client, isolated_db, one_face_path):
        payload = _post_frame(api_client, _data_url(one_face_path)).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is True
        assert data['matched_count'] == 0
        assert all(face['name'] is None for face in data['faces'])

    def test_no_face(self, api_client, isolated_db, no_face_path):
        payload = _post_frame(api_client, _data_url(no_face_path)).json()
        assert payload['code'] == 1002

    @pytest.mark.parametrize('raw', ['@@@', 'not-base64', ''])
    def test_bad_base64(self, api_client, isolated_db, raw):
        payload = _post_frame(api_client, raw).json()
        assert payload['code'] == 1001

    def test_oversize_rejected_before_decode(self, api_client, isolated_db, monkeypatch):
        monkeypatch.setattr(api_server, 'MAX_UPLOAD_BYTES', 16)

        def _unexpected(*_args, **_kwargs):
            pytest.fail('超限帧不应进入 b64decode')

        monkeypatch.setattr(api_server.base64, 'b64decode', _unexpected)
        payload = _post_frame(api_client, 'A' * 25).json()   # 16B 对应编码上限 24 字符
        assert payload['code'] == 1001

    @pytest.mark.parametrize('threshold', [0.1, 0.99])
    def test_threshold_out_of_range(self, api_client, isolated_db,
                                    one_face_path, threshold):
        payload = _post_frame(api_client, _data_url(one_face_path),
                              threshold=threshold).json()
        assert payload['code'] == 1001
        assert '阈值' in payload['message']

    def test_threshold_checked_before_decode(self, api_client, isolated_db):
        """阈值越界应优先于 base64 解码报错, 与图片接口错误优先级一致"""
        payload = _post_frame(api_client, 'AAAA', threshold=0.99).json()
        assert payload['code'] == 1001
        assert '阈值' in payload['message']

    def test_threshold_non_numeric(self, api_client, isolated_db, one_face_path):
        resp = _post_frame(api_client, _data_url(one_face_path), threshold='abc')
        assert resp.status_code == 422
        assert resp.json()['code'] == 422

    def test_missing_image_field(self, api_client, isolated_db):
        resp = api_client.post('/api/recognize/frame', json={'threshold': DEFAULT_THRESHOLD})
        assert resp.status_code == 422
        assert resp.json()['code'] == 422

    def test_unknown_face(self, api_client, seeded_dingzhen):
        if not os.path.exists(SUN_PATH):
            pytest.skip(f'缺少样例: {SUN_PATH}')
        payload = _post_frame(api_client, _data_url(SUN_PATH)).json()

        assert payload['code'] == 0
        data = payload['data']
        assert data['matched_count'] == 0
        assert all(face['name'] is None for face in data['faces'])

    def test_no_cli_unmatched_print(self, api_client, seeded_dingzhen, capsys):
        """接口层不走 CLI recognize_face(), 未命中时不应打印 Error! 无匹配人脸"""
        if not os.path.exists(SUN_PATH):
            pytest.skip(f'缺少样例: {SUN_PATH}')
        assert _post_frame(api_client, _data_url(SUN_PATH)).json()['code'] == 0
        captured = capsys.readouterr()
        assert '无匹配人脸' not in captured.out




# --------------------------------------- 并发保护 ---------------------------------------
class TestConcurrency:

    def test_frame_busy_returns_1008(self, api_client, isolated_db, one_face_path,
                                     monkeypatch):
        busy_lock = threading.Lock()
        busy_lock.acquire()                       # 模拟已有推理任务正在跑
        monkeypatch.setattr(api_server, '_infer_lock', busy_lock)
        try:
            resp = _post_frame(api_client, _data_url(one_face_path))
        finally:
            busy_lock.release()

        assert resp.status_code == 429
        assert resp.json()['code'] == 1008
        assert '丢弃' in resp.json()['message']

    def test_image_queue_timeout_returns_1008(self, api_client, isolated_db,
                                              one_face_path, monkeypatch):
        busy_lock = threading.Lock()
        busy_lock.acquire()
        monkeypatch.setattr(api_server, '_infer_lock', busy_lock)
        monkeypatch.setattr(api_server, '_INFER_TIMEOUT', 0.01)
        try:
            resp = _post_image(api_client, one_face_path)
        finally:
            busy_lock.release()

        assert resp.status_code == 200
        assert resp.json()['code'] == 1008

    def test_infer_lock_released_after_request(self, api_client, isolated_db,
                                               one_face_path):
        assert _post_image(api_client, one_face_path).json()['code'] == 0
        assert api_server._infer_lock.acquire(blocking=False) is True
        api_server._infer_lock.release()


# --------------------------------------- OpenAPI 契约 ---------------------------------------
class TestRecognizeOpenAPI:

    def test_routes_and_parameters(self, api_client):
        schema = api_client.get('/openapi.json').json()
        paths = schema['paths']
        assert '/api/recognize/image' in paths
        assert '/api/recognize/frame' in paths

        form_ref = paths['/api/recognize/image']['post'][
            'requestBody']['content']['multipart/form-data']['schema']['$ref']
        form = schema['components']['schemas'][form_ref.rsplit('/', 1)[-1]]
        assert form['required'] == ['file']
        assert form['properties']['threshold']['default'] == DEFAULT_THRESHOLD
        assert form['properties']['save_result']['default'] is False

        frame_ref = paths['/api/recognize/frame']['post'][
            'requestBody']['content']['application/json']['schema']['$ref']
        frame = schema['components']['schemas'][frame_ref.rsplit('/', 1)[-1]]
        assert frame['required'] == ['image_base64']
        assert frame['properties']['threshold']['default'] == DEFAULT_THRESHOLD

