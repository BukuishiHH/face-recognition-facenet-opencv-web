"""
Step 3-5 后端接口整体集成测试

覆盖:
  1. 7 个接口的正常链路(health / register×2 / recognize×2 / library×2);
  2. 无人脸、坏图、空文件、姓名非法、同名、非法 base64、阈值越界、空库、
     缺字段、记录不存在、推理繁忙等异常矩阵;
  3. 「录入 → 列表 → 识别命中 → 删除 → 再识别不命中」跨接口闭环;
  4. CORS 预检、OpenAPI 路由契约、大图坐标还原、标注图落盘。

说明:
  - api_client 为会话级夹具, 整份文件只加载/预热一次 FaceNet;
  - isolated_db 为函数级夹具, 每个用例使用独立临时 SQLite;
  - 标注图写入会话级临时目录, 不污染 backend/result_output/。
"""

import base64
import os
import threading
import uuid
from datetime import datetime

import cv2
import pytest

import api_server

FEATURE_DIM = 512
DEFAULT_THRESHOLD = 0.6

pytestmark = pytest.mark.api          # 支持 pytest -m "not api" 在无权重环境跳过


# --------------------------------------- 辅助函数 ---------------------------------------
def _unique_name():
    """唯一姓名: 保证用例可重复执行, 不会因残留数据触发 1004"""
    return f'测试_{uuid.uuid4().hex[:8]}'


def _data_url(path):
    """图片 → dataURL(带前缀), 与前端 canvas.toDataURL() 输出一致"""
    with open(path, 'rb') as fp:
        return 'data:image/jpeg;base64,' + base64.b64encode(fp.read()).decode()


def _upload(client, path, name=None, **form):
    """以 multipart 方式上传样例图, 返回 JSON"""
    with open(path, 'rb') as fp:
        files = {'file': (os.path.basename(path), fp.read(), 'image/jpeg')}
    data = {} if name is None else {'name': name}
    data.update({key: str(value) for key, value in form.items()})
    return client.post('/api/face/register/image', files=files, data=data).json()


def _upload_raw(client, raw, name=None, filename='bad.jpg'):
    """上传原始字节, 用于空文件 / 损坏图片场景"""
    data = {} if name is None else {'name': name}
    return client.post(
        '/api/face/register/image',
        files={'file': (filename, raw, 'image/jpeg')},
        data=data,
    ).json()


def _post_image_raw(client, raw, threshold=DEFAULT_THRESHOLD, save_result=False,
                    filename='image.jpg'):
    return client.post(
        '/api/recognize/image',
        files={'file': (filename, raw, 'image/jpeg')},
        data={'threshold': str(threshold), 'save_result': str(save_result).lower()},
    ).json()


def _post_image(client, path, threshold=DEFAULT_THRESHOLD, save_result=False):
    with open(path, 'rb') as fp:
        return _post_image_raw(client, fp.read(), threshold, save_result,
                              filename=os.path.basename(path))


def _post_frame(client, image_base64, threshold=DEFAULT_THRESHOLD):
    return client.post('/api/recognize/frame', json={
        'image_base64': image_base64,
        'threshold': threshold,
    }).json()


def _library(client):
    return client.get('/api/face/library').json()


def _record_count(db):
    matrix, info_list = db.get_all_features()
    return 0 if matrix is None else len(info_list)


def _post_frame_register(client, name, image_base64):
    return client.post('/api/face/register/camera-frame', json={
        'name': name,
        'image_base64': image_base64,
    }).json()


# --------------------------------------- 系统接口 ---------------------------------------
class TestHealth:

    def test_health(self, api_client, isolated_db):
        response = api_client.get('/api/health')
        payload = response.json()

        assert response.status_code == 200
        assert payload['code'] == 0
        data = payload['data']
        assert data['status'] == 'ok'
        assert data['version'] == api_server.VERSION
        assert data['device'] in {'cpu', 'cuda'}
        assert data['detector'] == api_server.DETECTOR_MODEL
        assert data['feature_dim'] == FEATURE_DIM
        assert data['library_size'] == 0

    def test_openapi_contains_all_routes(self, api_client):
        response = api_client.get('/openapi.json')
        schema = response.json()
        paths = schema['paths']

        assert response.status_code == 200
        for path in ('/api/health', '/api/face/register/image',
                     '/api/face/register/camera-frame', '/api/recognize/image',
                     '/api/recognize/frame', '/api/face/library'):
            assert path in paths
        assert set(paths['/api/face/library']) >= {'get', 'delete'}
        assert paths['/api/face/register/image']['post']['tags'] == ['face']
        assert paths['/api/recognize/frame']['post']['tags'] == ['recognize']

    def test_cors_preflight(self, api_client):
        response = api_client.options('/api/health', headers={
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'GET',
        })

        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'
        assert 'GET' in response.headers['access-control-allow-methods']


# --------------------------------------- 图片入库 ---------------------------------------
class TestRegisterImage:

    def test_ok(self, api_client, isolated_db, one_face_path):
        name = _unique_name()
        payload = _upload(api_client, one_face_path, name)

        assert payload['code'] == 0
        data = payload['data']
        assert data['record_id'] > 0
        assert data['name'] == name
        assert data['feature_dim'] == FEATURE_DIM
        assert data['face_count'] >= 1
        assert all(isinstance(data['selected'][key], int)
                   for key in ('x', 'y', 'w', 'h'))
        assert _record_count(isolated_db) == 1

    def test_duplicate_name(self, api_client, isolated_db, seeded_name, one_face_path):
        payload = _upload(api_client, one_face_path, seeded_name)

        assert payload['code'] == 1004
        assert seeded_name in payload['message']
        assert _record_count(isolated_db) == 1

    def test_no_face(self, api_client, isolated_db, no_face_path):
        payload = _upload(api_client, no_face_path, _unique_name())

        assert payload['code'] == 1002
        assert payload['data'] is None
        assert _record_count(isolated_db) == 0

    @pytest.mark.parametrize('raw', [b'', b'not an image'])
    def test_bad_file(self, api_client, isolated_db, raw):
        payload = _upload_raw(api_client, raw, _unique_name())

        assert payload['code'] == 1001
        assert payload['data'] is None
        assert _record_count(isolated_db) == 0

    @pytest.mark.parametrize('name', ['   ', 'x' * 33])
    def test_bad_name(self, api_client, isolated_db, one_face_path, name):
        payload = _upload(api_client, one_face_path, name)

        assert payload['code'] == 1001
        assert '姓名' in payload['message']
        assert _record_count(isolated_db) == 0

    def test_missing_name(self, api_client, isolated_db, one_face_path):
        with open(one_face_path, 'rb') as fp:
            response = api_client.post(
                '/api/face/register/image',
                files={'file': ('one_face.jpg', fp.read(), 'image/jpeg')},
            )

        assert response.status_code == 422
        assert response.json()['code'] == 422
        assert 'name' in response.json()['message']
        assert _record_count(isolated_db) == 0

    def test_multi_face_takes_one(self, api_client, isolated_db, multi_face_path):
        payload = _upload(api_client, multi_face_path, _unique_name())

        assert payload['code'] == 0
        assert payload['data']['face_count'] >= 2
        assert '最大人脸' in payload['message']
        assert _record_count(isolated_db) == 1


# --------------------------------------- 摄像头帧入库 ---------------------------------------
class TestRegisterFrame:

    def test_data_url(self, api_client, isolated_db, one_face_path):
        name = _unique_name()
        payload = _post_frame_register(api_client, name, _data_url(one_face_path))

        assert payload['code'] == 0
        assert payload['data']['record_id'] > 0
        assert payload['data']['name'] == name
        assert _record_count(isolated_db) == 1

    def test_raw_base64(self, api_client, isolated_db, one_face_path):
        with open(one_face_path, 'rb') as fp:
            raw = base64.b64encode(fp.read()).decode()
        payload = _post_frame_register(api_client, _unique_name(), raw)

        assert payload['code'] == 0
        assert payload['data']['record_id'] > 0
        assert _record_count(isolated_db) == 1

    def test_blank_name(self, api_client, isolated_db, one_face_path):
        payload = _post_frame_register(api_client, '   ', _data_url(one_face_path))

        assert payload['code'] == 1001
        assert '姓名' in payload['message']
        assert _record_count(isolated_db) == 0

    def test_bad_base64(self, api_client, isolated_db):
        payload = _post_frame_register(api_client, _unique_name(), 'not-base64')

        assert payload['code'] == 1001
        assert payload['data'] is None
        assert _record_count(isolated_db) == 0

    def test_missing_field(self, api_client, isolated_db):
        response = api_client.post(
            '/api/face/register/camera-frame',
            json={'name': _unique_name()},
        )

        assert response.status_code == 422
        assert response.json()['code'] == 422
        assert 'image_base64' in response.json()['message']
        assert _record_count(isolated_db) == 0


# --------------------------------------- 图片识别 ---------------------------------------
class TestRecognizeImage:

    def test_matched(self, api_client, seeded_name, one_face_path):
        payload = _post_image(api_client, one_face_path)

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is False
        assert data['total_faces'] >= 1
        assert data['matched_count'] >= 1
        hit = [face for face in data['faces'] if face['matched']]
        assert seeded_name in [face['name'] for face in hit]
        assert hit[0]['similarity'] >= DEFAULT_THRESHOLD
        assert data['elapsed_ms'] >= 0
        assert data['result_image_url'] is None

    def test_empty_library(self, api_client, isolated_db, one_face_path):
        payload = _post_image(api_client, one_face_path)

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is True
        assert data['total_faces'] >= 1
        assert data['matched_count'] == 0
        assert all(face['name'] is None and face['matched'] is False
                   for face in data['faces'])
        assert all(face['similarity'] == 0.0 for face in data['faces'])
        assert all(face['box']['w'] > 0 and face['box']['h'] > 0
                   for face in data['faces'])

    def test_no_face(self, api_client, isolated_db, no_face_path):
        payload = _post_image(api_client, no_face_path)

        assert payload['code'] == 1002
        assert payload['data'] is None

    def test_bad_file(self, api_client, isolated_db):
        payload = _post_image_raw(api_client, b'broken')

        assert payload['code'] == 1001
        assert payload['data'] is None

    @pytest.mark.parametrize('threshold', ['0.1', '0.99'])
    def test_threshold_out_of_range(self, api_client, isolated_db,
                                    one_face_path, threshold):
        with open(one_face_path, 'rb') as fp:
            payload = api_client.post(
                '/api/recognize/image',
                files={'file': ('one_face.jpg', fp.read(), 'image/jpeg')},
                data={'threshold': threshold},
            ).json()

        assert payload['code'] == 1001
        assert '阈值' in payload['message']

    def test_save_result_image(self, api_client, seeded_name, one_face_path):
        payload = _post_image(api_client, one_face_path, save_result=True)

        assert payload['code'] == 0
        assert payload['data']['matched_count'] >= 1
        url = payload['data']['result_image_url']
        assert url.startswith(f'{api_server.STATIC_PREFIX}/')
        assert url.endswith('.jpg')
        result_path = os.path.join(api_server.RESULT_DIR, os.path.basename(url))
        assert os.path.isfile(result_path)
        assert os.path.getsize(result_path) > 0

    def test_large_image_box_scaled(self, api_client, seeded_name, one_face_path,
                                    imread):
        small = imread(one_face_path)
        big = cv2.resize(small, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        assert max(big.shape[:2]) > api_server.MAX_FRAME_SIDE

        def _post(image):
            ok_flag, buffer = cv2.imencode('.jpg', image)
            assert ok_flag
            return _post_image_raw(api_client, buffer.tobytes())

        small_payload = _post(small)
        big_payload = _post(big)
        assert small_payload['code'] == 0
        assert big_payload['code'] == 0
        assert small_payload['data']['faces']
        assert big_payload['data']['faces']
        # Haar 在放大 + JPEG 压缩后可能产生小面积误检, 按面积取真实人脸框
        small_face = max(small_payload['data']['faces'],
                         key=lambda face: face['box']['w'] * face['box']['h'])
        big_face = max(big_payload['data']['faces'],
                       key=lambda face: face['box']['w'] * face['box']['h'])
        small_box = small_face['box']
        big_box = big_face['box']

        assert all(face['box']['x'] + face['box']['w'] <= big.shape[1]
                   and face['box']['y'] + face['box']['h'] <= big.shape[0]
                   for face in big_payload['data']['faces'])
        assert big_box['w'] / big.shape[1] == pytest.approx(
            small_box['w'] / small.shape[1], abs=0.05)


# --------------------------------------- 摄像头帧识别 ---------------------------------------
class TestRecognizeFrame:

    def test_ok(self, api_client, seeded_name, one_face_path):
        payload = _post_frame(api_client, _data_url(one_face_path))

        assert payload['code'] == 0
        data = payload['data']
        assert data['library_empty'] is False
        assert data['total_faces'] >= 1
        assert data['matched_count'] >= 1
        assert seeded_name in [face['name'] for face in data['faces']
                               if face['matched']]
        assert data['elapsed_ms'] >= 0

    def test_threshold_out_of_range(self, api_client, isolated_db, one_face_path):
        payload = _post_frame(api_client, _data_url(one_face_path), threshold=0.99)

        assert payload['code'] == 1001
        assert '阈值' in payload['message']
        assert payload['data'] is None

    def test_bad_base64(self, api_client, isolated_db):
        payload = _post_frame(api_client, '@@@')

        assert payload['code'] == 1001
        assert payload['data'] is None

    def test_threshold_checked_before_decode(self, api_client, isolated_db):
        payload = _post_frame(api_client, 'AAAA', threshold=0.99)

        assert payload['code'] == 1001
        assert '阈值' in payload['message']
        assert payload['data'] is None

    def test_missing_field(self, api_client, isolated_db):
        response = api_client.post('/api/recognize/frame',
                                   json={'threshold': DEFAULT_THRESHOLD})

        assert response.status_code == 422
        assert response.json()['code'] == 422
        assert 'image_base64' in response.json()['message']


# --------------------------------------- 特征库管理 ---------------------------------------
class TestLibrary:

    def test_list(self, api_client, seeded_name):
        payload = _library(api_client)

        assert payload['code'] == 0
        data = payload['data']
        assert data['total'] == len(data['items']) == 1
        item = next(item for item in data['items'] if item['name'] == seeded_name)
        assert set(item) == {'id', 'name', 'timestamp', 'feature_dim'}
        assert item['id'] > 0
        assert item['feature_dim'] == FEATURE_DIM
        assert len(item['timestamp']) == 19
        datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S')
        assert 'feature' not in item

    def test_delete(self, api_client, seeded_name):
        payload = api_client.delete(
            '/api/face/library', params={'name': seeded_name}).json()

        assert payload['code'] == 0
        assert payload['data'] == {'name': seeded_name, 'deleted': 1}
        after = _library(api_client)['data']
        assert after['total'] == 0
        assert seeded_name not in [item['name'] for item in after['items']]

    def test_delete_not_found(self, api_client, seeded_name):
        first = api_client.delete(
            '/api/face/library', params={'name': seeded_name}).json()
        second = api_client.delete(
            '/api/face/library', params={'name': seeded_name}).json()

        assert first['code'] == 0
        assert second['code'] == 1005
        assert second['data'] is None

    def test_blank_name(self, api_client, isolated_db):
        payload = api_client.delete(
            '/api/face/library', params={'name': '   '}).json()

        assert payload['code'] == 1001
        assert '姓名' in payload['message']
        assert payload['data'] is None

    def test_missing_name(self, api_client, isolated_db):
        response = api_client.delete('/api/face/library')

        assert response.status_code == 422
        assert response.json()['code'] == 422
        assert 'name' in response.json()['message']


# --------------------------------------- 跨接口闭环 ---------------------------------------
class TestEndToEnd:

    def test_register_list_recognize_delete(self, api_client, isolated_db,
                                            one_face_path):
        name = _unique_name()

        # 1. 录入
        registered = _upload(api_client, one_face_path, name)
        assert registered['code'] == 0
        assert registered['data']['name'] == name

        # 2. 列表可见
        items = _library(api_client)['data']['items']
        assert [item['name'] for item in items] == [name]

        # 3. 识别命中
        hit = _post_image(api_client, one_face_path)
        assert hit['code'] == 0
        assert hit['data']['library_empty'] is False
        assert hit['data']['matched_count'] >= 1
        assert name in [face['name'] for face in hit['data']['faces']
                        if face['matched']]

        # 4. 删除
        deleted = api_client.delete(
            '/api/face/library', params={'name': name}).json()
        assert deleted['code'] == 0
        assert deleted['data']['deleted'] == 1
        assert _record_count(isolated_db) == 0

        # 5. 再识别: 空库不报错, 链路闭环
        after = _post_image(api_client, one_face_path)
        assert after['code'] == 0
        assert after['data']['library_empty'] is True
        assert after['data']['matched_count'] == 0
        assert after['data']['faces']
        assert all(face['name'] is None for face in after['data']['faces'])


# --------------------------------------- 并发保护 ---------------------------------------
class TestConcurrency:

    def test_frame_busy_returns_1008(self, api_client, isolated_db, one_face_path,
                                     monkeypatch):
        busy_lock = threading.Lock()
        busy_lock.acquire()                       # 模拟已有推理任务正在跑
        monkeypatch.setattr(api_server, '_infer_lock', busy_lock)
        try:
            response = api_client.post('/api/recognize/frame', json={
                'image_base64': _data_url(one_face_path),
                'threshold': DEFAULT_THRESHOLD,
            })
        finally:
            busy_lock.release()

        payload = response.json()
        assert response.status_code == 429
        assert payload['code'] == 1008
        assert '丢弃' in payload['message']
        assert payload['data'] is None

