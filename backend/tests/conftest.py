"""
pytest 共享配置与夹具

职责:
  1. 将 backend 目录注入 sys.path, 使被测模块(face_detection / face_recognition /
     db_utils / main)可直接 import(与 CLI 运行时"同目录平级导入"保持一致);
  2. 提供测试样例路径夹具(样例缺失时自动跳过对应用例, 不误报失败);
  3. 提供隔离的特征库夹具, 避免测试数据写入真实 backend/face_features.db;
  4. 提供会话级 API 客户端与预置姓名夹具, 供阶段 3-5 接口集成测试复用。
"""

import os
import sys
import uuid

import cv2
import numpy as np
import pytest

# --------------------------------------- 路径注入 ---------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

SAMPLE_DIR = os.path.join(BACKEND_DIR, 'sample_faces', '_test_samples')


def imread_unicode(path):
    """
    读取图片(兼容中文/非 ASCII 路径)

    与 backend/main.py 的 read_image 采用同一方案: Windows 下 cv2.imread 不支持
    非 ASCII 路径(会静默返回 None), 必须 np.fromfile + cv2.imdecode。
    本项目路径含中文, 因此测试统一通过该函数读图。
    """
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.fixture(scope='session')
def imread():
    """提供兼容中文路径的读图函数"""
    return imread_unicode


# --------------------------------------- 测试样例夹具 ---------------------------------------
@pytest.fixture(scope='session')
def sample_dir():
    """测试样例目录: backend/sample_faces/_test_samples"""
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip(f"缺少样例目录: {SAMPLE_DIR} (见 temp/TEST_SAMPLES_GUIDE.md)")
    return SAMPLE_DIR


@pytest.fixture(scope='session')
def one_face_path(sample_dir):
    """单人脸样例(见 temp/TEST_SAMPLES_GUIDE.md 第 1 节)"""
    path = os.path.join(sample_dir, 'one_face.jpg')
    if not os.path.exists(path):
        pytest.skip(f"缺少单人脸样例: {path}")
    return path


@pytest.fixture(scope='session')
def no_face_path(sample_dir):
    """无人脸样例"""
    path = os.path.join(sample_dir, 'no_face.jpg')
    if not os.path.exists(path):
        pytest.skip(f"缺少无人脸样例: {path}")
    return path


@pytest.fixture(scope='session')
def multi_face_path(sample_dir, one_face_path):
    """
    多人脸样例: 已存在则直接使用;
    不存在时按 temp/TEST_SAMPLES_GUIDE.md 方案 C 合成(单脸图水平翻转后并排)。
    """
    path = os.path.join(sample_dir, 'multi_face.jpg')
    if not os.path.exists(path):
        img = imread_unicode(one_face_path)
        if img is None:
            pytest.skip(f"无法读取单人脸样例: {one_face_path}")
        flipped = cv2.flip(img, 1)                        # 水平翻转, 视作第二张"不同"人脸
        multi = np.hstack([cv2.resize(x, (320, 320)) for x in (img, flipped)])
        ok, buf = cv2.imencode('.jpg', multi)
        if not ok:
            pytest.skip("多人脸样例合成失败")
        with open(path, 'wb') as f:
            f.write(buf)
    return path


# --------------------------------------- 隔离特征库夹具 ---------------------------------------
@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """
    隔离的特征库(临时文件), 同步替换三处 face_db 引用:
      db_utils.face_db         (全局单例)
      face_recognition.face_db (识别模块 import 时绑定的引用)
      main.face_db             (CLI 模块 import 时绑定的引用)
    避免测试数据写入真实 backend/face_features.db。
    """
    import db_utils
    from db_utils import FaceFeatureDB

    db = FaceFeatureDB(str(tmp_path / 'isolated.db'))
    monkeypatch.setattr(db_utils, 'face_db', db, raising=False)

    try:
        import face_recognition as fr
        monkeypatch.setattr(fr, 'face_db', db, raising=False)
    except ImportError:                                  # 未安装 torch 时不阻塞其他用例
        pass

    try:
        import main as cli
        monkeypatch.setattr(cli, 'face_db', db, raising=False)
    except ImportError:
        pass

    yield db
    db.close()


# --------------------------------------- 接口集成夹具 ---------------------------------------
@pytest.fixture(scope='session')
def api_client(tmp_path_factory):
    """
    会话级 TestClient:

      - 整个 pytest 会话只触发一次 lifespan(FaceNet 只加载/预热一次);
      - RESULT_DIR 指向临时目录, 避免标注图写入 backend/result_output/;
      - 数据库仍由函数级 isolated_db 替换, 从而保证用例之间互不串数据。

    说明: 会话夹具内不能使用函数级 monkeypatch, 因此手动保存/恢复模块属性。
    """
    from fastapi.testclient import TestClient

    import api_server

    original_result_dir = api_server.RESULT_DIR
    api_server.RESULT_DIR = str(tmp_path_factory.mktemp('api_result'))
    try:
        with TestClient(api_server.app) as client:
            yield client
    finally:
        api_server.RESULT_DIR = original_result_dir


@pytest.fixture()
def seeded_name(api_client, isolated_db, one_face_path):
    """通过接口录入一个唯一姓名并返回它; 用例结束后由 isolated_db 丢弃数据"""
    name = f'测试_{uuid.uuid4().hex[:8]}'
    with open(one_face_path, 'rb') as fp:
        payload = api_client.post(
            '/api/face/register/image',
            files={'file': ('one_face.jpg', fp.read(), 'image/jpeg')},
            data={'name': name},
        ).json()
    assert payload['code'] == 0, payload
    return name
