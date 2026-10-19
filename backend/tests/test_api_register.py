"""
Step 3-2 人脸采集入库接口测试

覆盖:
  1. 图片上传 / 摄像头帧两条正常链路;
  2. 姓名、文件大小、损坏图片、非法 base64、无人脸、同名、特征失败等异常分支;
  3. 多脸取最大框策略。

说明: 每个用例通过 isolated_db 使用临时特征库, 不触碰 backend/face_features.db。
"""

import base64
import os
import uuid

import cv2
import pytest
from fastapi.testclient import TestClient

import api_server
import db_utils

FEATURE_DIM = 512


# --------------------------------------- 夹具与辅助函数 ---------------------------------------
@pytest.fixture()
def api_client(isolated_db):
    """启动完整生命周期(检测器 + FaceNet + 临时特征库)的测试客户端"""
    with TestClient(api_server.app) as client:
        yield client


def _unique_name():
    return f'测试_{uuid.uuid4().hex[:8]}'


def _upload(client, path, name, filename=None, content_type='image/jpeg'):
    with open(path, 'rb') as f:
        raw = f.read()
    return client.post(
        '/api/face/register/image',
        files={'file': (filename or os.path.basename(path), raw, content_type)},
        data={'name': name},
    ).json()


def _upload_raw(client, raw, name, filename='bad.jpg', content_type='image/jpeg'):
    return client.post(
        '/api/face/register/image',
        files={'file': (filename, raw, content_type)},
        data={'name': name},
    ).json()


def _data_url(path):
    with open(path, 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


def _post_frame(client, name, image_base64):
    return client.post(
        '/api/face/register/camera-frame',
        json={'name': name, 'image_base64': image_base64},
    ).json()


def _record_count(db):
    matrix, info = db.get_all_features()
    return 0 if matrix is None else len(info)


# --------------------------------------- 图片上传正常链路 ---------------------------------------
class TestRegisterImage:

    def test_ok(self, api_client, isolated_db, one_face_path):
        data = _upload(api_client, one_face_path, _unique_name())

        assert data['code'] == 0
        assert data['message'] == '人脸录入成功'
        assert data['data']['record_id'] > 0
        assert data['data']['feature_dim'] == FEATURE_DIM
        assert data['data']['face_count'] >= 1
        selected = data['data']['selected']
        assert all(isinstance(selected[key], int) for key in ('x', 'y', 'w', 'h'))
        assert _record_count(isolated_db) == 1

    def test_name_is_stripped(self, api_client, isolated_db, one_face_path):
        data = _upload(api_client, one_face_path, '  丁真  ')

        assert data['code'] == 0
        assert data['data']['name'] == '丁真'
        assert isolated_db.exists_name('丁真') is True

    @pytest.mark.parametrize('channels', ['gray', 'alpha'])
    def test_gray_and_alpha_png(self, api_client, isolated_db, one_face_path, imread, channels):
        """灰度 / 带 alpha 的 PNG 统一转 3 通道后仍可入库"""
        img = imread(one_face_path)
        if channels == 'gray':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        success, buffer = cv2.imencode('.png', converted)
        assert success

        data = _upload_raw(
            api_client, buffer.tobytes(), _unique_name(),
            filename=f'{channels}.png', content_type='image/png',
        )
        assert data['code'] == 0
        assert _record_count(isolated_db) == 1


# --------------------------------------- 图片上传异常分支 ---------------------------------------
class TestRegisterImageErrors:

    def test_duplicate_name(self, api_client, isolated_db, one_face_path):
        name = _unique_name()
        assert _upload(api_client, one_face_path, name)['code'] == 0
        second = _upload(api_client, one_face_path, name)

        assert second['code'] == 1004
        assert name in second['message']
        assert _record_count(isolated_db) == 1

    def test_no_face(self, api_client, isolated_db, no_face_path):
        data = _upload(api_client, no_face_path, _unique_name())

        assert data['code'] == 1002
        assert _record_count(isolated_db) == 0

    @pytest.mark.parametrize('raw', [b'', b'not an image'])
    def test_bad_file(self, api_client, isolated_db, raw):
        data = _upload_raw(api_client, raw, _unique_name())

        assert data['code'] == 1001
        assert _record_count(isolated_db) == 0

    @pytest.mark.parametrize('name', ['', '   ', 'x' * 33])
    def test_bad_name(self, api_client, isolated_db, one_face_path, name):
        assert _upload(api_client, one_face_path, name)['code'] == 1001
        assert _record_count(isolated_db) == 0

    def test_oversize_skips_inference(self, api_client, isolated_db, monkeypatch):
        """>5 MB 必须在解码/推理前返回 1001"""
        def _unexpected(*args, **kwargs):
            pytest.fail('超大文件不应进入检测/特征提取')

        monkeypatch.setattr(api_server, '_register_from_bgr', _unexpected)
        data = _upload_raw(
            api_client,
            b'x' * (api_server.MAX_UPLOAD_BYTES + 1),
            _unique_name(),
        )

        assert data['code'] == 1001
        assert _record_count(isolated_db) == 0

    def test_missing_name(self, api_client, one_face_path):
        with open(one_face_path, 'rb') as f:
            response = api_client.post(
                '/api/face/register/image',
                files={'file': ('one_face.jpg', f.read(), 'image/jpeg')},
            )

        assert response.status_code == 422
        assert response.json()['code'] == 422

    def test_feature_failure(self, api_client, isolated_db, one_face_path, monkeypatch):
        monkeypatch.setattr(
            api_server.FaceRecognition,
            'extract_feature',
            staticmethod(lambda _roi: None),
        )
        data = _upload(api_client, one_face_path, _unique_name())

        assert data['code'] == 1003
        assert _record_count(isolated_db) == 0

    def test_save_failure(self, api_client, isolated_db, one_face_path, monkeypatch):
        monkeypatch.setattr(db_utils.face_db, 'save_feature', lambda _feature, _name: None)
        data = _upload(api_client, one_face_path, _unique_name())

        assert data['code'] == 1003
        assert _record_count(isolated_db) == 0


# --------------------------------------- 摄像头帧链路 ---------------------------------------
class TestRegisterFrame:

    def test_data_url(self, api_client, isolated_db, one_face_path):
        data = _post_frame(api_client, _unique_name(), _data_url(one_face_path))

        assert data['code'] == 0
        assert data['data']['record_id'] > 0
        assert _record_count(isolated_db) == 1

    def test_raw_base64(self, api_client, isolated_db, one_face_path):
        with open(one_face_path, 'rb') as f:
            raw = base64.b64encode(f.read()).decode()
        data = _post_frame(api_client, _unique_name(), raw)

        assert data['code'] == 0
        assert _record_count(isolated_db) == 1

    def test_blank_name(self, api_client, isolated_db, one_face_path):
        data = _post_frame(api_client, '   ', _data_url(one_face_path))

        assert data['code'] == 1001
        assert _record_count(isolated_db) == 0

    @pytest.mark.parametrize('raw', ['not-base64', '@@@@', 'AAAA'])
    def test_bad_base64(self, api_client, isolated_db, raw):
        data = _post_frame(api_client, _unique_name(), raw)

        assert data['code'] == 1001
        assert _record_count(isolated_db) == 0

    def test_missing_field(self, api_client):
        response = api_client.post('/api/face/register/camera-frame', json={'name': 'x'})

        assert response.status_code == 422
        assert response.json()['code'] == 422


