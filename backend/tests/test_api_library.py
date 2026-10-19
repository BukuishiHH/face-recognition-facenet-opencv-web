"""
Step 3-4 特征库管理接口测试

覆盖:
  1. 数据层 list_records: 空库、排序、元信息字段、NULL 姓名、只读;
  2. GET /api/face/library: 空库、字段、时间格式、不读取 feature BLOB、1000 条体积;
  3. DELETE /api/face/library: 中文姓名、同名多条、1005、1001、422、1007、重复删除;
  4. 删除后识别链路立即失效;
  5. 删除成功/失败日志与 OpenAPI 契约。

说明: 每个用例通过 isolated_db 使用临时特征库, 不触碰 backend/face_features.db。
"""

import logging
import os
import re
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_server
from face_detection import FaceDetection
from face_recognition import FaceRecognition

FEATURE_DIM = 512
TIMESTAMP_RE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')


# --------------------------------------- 夹具与辅助函数 ---------------------------------------
@pytest.fixture()
def api_client(isolated_db):
    """启动完整生命周期(检测器 + FaceNet + 临时特征库)的测试客户端"""
    with TestClient(api_server.app) as client:
        yield client


@pytest.fixture()
def api_logs(caplog):
    """临时允许 api_server logger 向 root 传播, 便于 caplog 捕获"""
    previous = api_server.logger.propagate
    api_server.logger.propagate = True
    caplog.set_level(logging.INFO, logger='api_server')
    try:
        yield caplog
    finally:
        api_server.logger.propagate = previous


def _make_feature(seed=0):
    return np.random.default_rng(seed).random((1, FEATURE_DIM)).astype(np.float32)


def _insert_record(db, name, seed=0, timestamp='2026-10-19 10:00:00'):
    """直接写库, 构造同名多条 / NULL 姓名 / 大批量等 save_feature 不便构造的数据"""
    cursor = db.conn.cursor()
    cursor.execute(
        'INSERT INTO face_features (feature, timestamp, face_id, feature_dim) '
        'VALUES (?, ?, ?, ?)',
        (_make_feature(seed).tobytes(), timestamp, name, FEATURE_DIM),
    )
    db.conn.commit()
    record_id = cursor.lastrowid
    cursor.close()
    return record_id


def _feature_from_image(image):
    """对样例图检测第一张脸并提取特征(用于预置识别联动数据)"""
    faces = api_server.app.state.detector.detect_faces(image)
    assert faces, '预置特征失败: 样例图未检测到人脸'
    roi = FaceDetection.crop_face_rois(image, faces)[0][0]
    feature = FaceRecognition.extract_feature(roi)
    assert feature is not None, '预置特征失败: extract_feature 返回 None'
    return feature


def _post_image(client, path):
    with open(path, 'rb') as fp:
        raw = fp.read()
    return client.post(
        '/api/recognize/image',
        files={'file': (os.path.basename(path), raw, 'image/jpeg')},
        data={'threshold': '0.6'},
    )


def _get_library(client):
    return client.get('/api/face/library')


def _delete(client, name):
    return client.delete('/api/face/library', params={'name': name})


# --------------------------------------- 列表接口 ---------------------------------------
class TestLibraryList:

    def test_empty_returns_empty_list(self, api_client):
        payload = _get_library(api_client).json()
        assert payload['code'] == 0
        assert payload['message'] == '共 0 条特征记录'
        assert payload['data'] == {'total': 0, 'items': []}

    def test_fields_and_order(self, api_client, isolated_db):
        _insert_record(isolated_db, '丁真', seed=1, timestamp='2026-10-19 10:00:00')
        _insert_record(isolated_db, '测试用户A', seed=2, timestamp='2026-10-19 10:01:00')

        payload = _get_library(api_client).json()
        data = payload['data']
        assert payload['code'] == 0
        assert payload['message'] == '共 2 条特征记录'
        assert data['total'] == 2
        assert [item['name'] for item in data['items']] == ['丁真', '测试用户A']
        assert [item['id'] for item in data['items']] == sorted(
            item['id'] for item in data['items'])
        for item in data['items']:
            assert set(item) == {'id', 'name', 'timestamp', 'feature_dim'}
            assert item['feature_dim'] == FEATURE_DIM
            assert TIMESTAMP_RE.match(item['timestamp'])

    def test_null_name_rendered_as_empty_string(self, api_client, isolated_db):
        _insert_record(isolated_db, None, seed=3)
        item = _get_library(api_client).json()['data']['items'][0]
        assert item['name'] == ''

    def test_response_has_no_feature_field(self, api_client, isolated_db):
        _insert_record(isolated_db, '无特征字段', seed=4)
        item = _get_library(api_client).json()['data']['items'][0]
        assert 'feature' not in item
        assert 'feature_blob' not in item

    def test_sql_does_not_select_feature_blob(self, api_client, isolated_db):
        _insert_record(isolated_db, 'SQL 检查', seed=5)
        queries = []
        isolated_db.conn.set_trace_callback(queries.append)
        try:
            assert _get_library(api_client).json()['code'] == 0
        finally:
            isolated_db.conn.set_trace_callback(None)

        selects = [re.sub(r'\s+', ' ', query).strip().lower()
                   for query in queries if 'from face_features' in query.lower()]
        assert selects, '未捕获到 list_records 的 SELECT 语句'
        assert all('select id, face_id, timestamp, feature_dim from face_features' in query
                   for query in selects), selects

    def test_1000_records_response_under_200kb(self, api_client, isolated_db):
        blob = _make_feature(seed=6).tobytes()
        rows = [(blob, '2026-10-19 10:00:00', f'user{i:04d}', FEATURE_DIM)
                for i in range(1000)]
        cursor = isolated_db.conn.cursor()
        cursor.executemany(
            'INSERT INTO face_features (feature, timestamp, face_id, feature_dim) '
            'VALUES (?, ?, ?, ?)', rows)
        isolated_db.conn.commit()
        cursor.close()

        response = _get_library(api_client)
        data = response.json()['data']
        assert response.status_code == 200
        assert data['total'] == 1000
        assert len(data['items']) == 1000
        assert len(response.content) < 200 * 1024
        assert 'feature' not in data['items'][0]

    def test_db_error_returns_1007_without_traceback(self, api_client, isolated_db,
                                                    monkeypatch):
        def _raise():
            raise sqlite3.OperationalError('database is locked')

        monkeypatch.setattr(isolated_db, 'list_records', _raise)
        response = _get_library(api_client)
        payload = response.json()
        assert response.status_code == 200
        assert payload['code'] == 1007
        assert '读取特征库失败' in payload['message']
        assert payload['data'] is None
        assert 'Traceback' not in response.text


# --------------------------------------- 删除接口 ---------------------------------------
class TestLibraryDelete:

    def test_delete_chinese_name(self, api_client, isolated_db):
        _insert_record(isolated_db, '丁真', seed=10)
        _insert_record(isolated_db, '测试用户A', seed=11)

        payload = _delete(api_client, '丁真').json()
        assert payload['code'] == 0
        assert payload['data'] == {'name': '丁真', 'deleted': 1}
        assert payload['message'] == '已删除 1 条「丁真」特征'

        names = [item['name'] for item in _get_library(api_client).json()['data']['items']]
        assert names == ['测试用户A']

    def test_delete_strips_whitespace(self, api_client, isolated_db):
        _insert_record(isolated_db, '丁真', seed=12)
        payload = _delete(api_client, '  丁真  ').json()
        assert payload['code'] == 0
        assert payload['data']['name'] == '丁真'
        assert payload['data']['deleted'] == 1

    def test_delete_all_same_name_records(self, api_client, isolated_db):
        for seed in range(3):
            _insert_record(isolated_db, '同名多条', seed=20 + seed)

        payload = _delete(api_client, '同名多条').json()
        assert payload['code'] == 0
        assert payload['data']['deleted'] == 3
        assert _get_library(api_client).json()['data']['total'] == 0

    def test_delete_missing_returns_1005_and_keeps_data(self, api_client, isolated_db):
        _insert_record(isolated_db, '保留', seed=30)
        response = _delete(api_client, '不存在的人')
        payload = response.json()
        assert response.status_code == 200
        assert payload['code'] == 1005
        assert '不存在' in payload['message']
        assert payload['data'] is None
        assert _get_library(api_client).json()['data']['total'] == 1

    @pytest.mark.parametrize('name', ['', '   ', '\t\n'])
    def test_blank_name_returns_1001(self, api_client, isolated_db, name):
        payload = _delete(api_client, name).json()
        assert payload['code'] == 1001
        assert payload['message'] == '姓名不能为空或纯空格'
        assert payload['data'] is None

    def test_missing_query_returns_422(self, api_client):
        response = api_client.delete('/api/face/library')
        payload = response.json()
        assert response.status_code == 422
        assert payload['code'] == 422
        assert 'name' in payload['message']
        assert 'Traceback' not in response.text

    def test_repeated_delete_never_returns_500(self, api_client, isolated_db):
        _insert_record(isolated_db, '重复删除', seed=31)
        for attempt in range(10):
            response = _delete(api_client, '重复删除')
            assert response.status_code == 200
            assert response.json()['code'] == (0 if attempt == 0 else 1005)

    def test_db_error_returns_1007_without_traceback(self, api_client, isolated_db,
                                                    api_logs, monkeypatch):
        def _raise(_name):
            raise sqlite3.OperationalError('database is locked')

        monkeypatch.setattr(isolated_db, 'delete_by_name', _raise)
        response = _delete(api_client, '丁真')
        payload = response.json()
        assert response.status_code == 200
        assert payload['code'] == 1007
        assert payload['message'] == '删除失败，请查看服务端日志'
        assert payload['data'] is None
        assert 'Traceback' not in response.text
        assert '删除特征失败' in api_logs.text
        assert 'database is locked' in api_logs.text


# --------------------------------------- 日志与可观测 ---------------------------------------
class TestLibraryLogging:

    def test_delete_success_logged(self, api_client, isolated_db, api_logs):
        _insert_record(isolated_db, '日志测试', seed=40)
        assert _delete(api_client, '日志测试').json()['code'] == 0
        assert '删除特征成功: name=日志测试, deleted=1' in api_logs.text
        assert '耗时=' in api_logs.text

    def test_delete_missing_logged(self, api_client, isolated_db, api_logs):
        assert _delete(api_client, '日志缺人').json()['code'] == 1005
        assert '删除特征未命中: name=日志缺人, deleted=0' in api_logs.text


# --------------------------------------- 与识别链路联动 ---------------------------------------
class TestDeleteRecognitionIntegration:

    def test_deleted_name_no_longer_matched(self, api_client, isolated_db,
                                           one_face_path, imread):
        name = '联动测试'
        feature = _feature_from_image(imread(one_face_path))
        assert isolated_db.save_feature(feature, name) is not None

        before = _post_image(api_client, one_face_path).json()
        assert before['code'] == 0
        assert before['data']['matched_count'] == 1
        assert before['data']['faces'][0]['name'] == name
        assert before['data']['faces'][0]['matched'] is True

        assert _delete(api_client, name).json()['data']['deleted'] == 1
        assert isolated_db.exists_name(name) is False

        after = _post_image(api_client, one_face_path).json()
        assert after['code'] == 0
        assert after['data']['library_empty'] is True
        assert after['data']['matched_count'] == 0
        assert after['data']['faces']           # 空库仍返回人脸框
        assert all(item['matched'] is False and item['name'] is None
                   for item in after['data']['faces'])


# --------------------------------------- OpenAPI 契约 ---------------------------------------
class TestLibraryOpenAPI:

    def test_get_and_delete_contract(self, api_client):
        schema = api_client.get('/openapi.json').json()
        operations = schema['paths']['/api/face/library']
        get_op = operations['get']
        delete_op = operations['delete']

        assert get_op['tags'] == ['library']
        assert not get_op.get('parameters')
        assert '200' in get_op['responses']

        assert delete_op['tags'] == ['library']
        params = delete_op['parameters']
        assert len(params) == 1
        param = params[0]
        assert param['name'] == 'name'
        assert param['in'] == 'query'
        assert param['required'] is True
        assert param['schema']['type'] == 'string'

        components = schema['components']['schemas']
        assert 'LibraryItem' in components
        assert 'LibraryData' in components
        assert 'DeleteData' in components
