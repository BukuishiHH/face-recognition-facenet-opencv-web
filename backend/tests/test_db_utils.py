"""
Step 2-3 SQLite 人脸特征库模块单元测试
被测模块: backend/db_utils.py
说明: 全部用例使用 tmp_path 临时库, 不触碰真实 backend/face_features.db。
"""

import numpy as np
import pytest

from db_utils import FaceFeatureDB

FEATURE_DIM = 512


def make_feature(seed: int = 0) -> np.ndarray:
    """生成一条 (1, 512) 的 float32 模拟特征"""
    rng = np.random.default_rng(seed)
    return rng.random((1, FEATURE_DIM)).astype(np.float32)


@pytest.fixture()
def db(tmp_path):
    """每个用例独立临时库, 结束后释放句柄(规避 Windows 文件占用)"""
    instance = FaceFeatureDB(str(tmp_path / 'unit_test.db'))
    yield instance
    instance.close()


# --------------------------------------- 建表 ---------------------------------------
class TestCreateTable:

    def test_table_created(self, db):
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='face_features'"
        )
        assert cursor.fetchone() is not None
        cursor.close()

    def test_init_is_idempotent(self, tmp_path):
        """重复初始化不报错、不丢数据"""
        path = str(tmp_path / 'idempotent.db')
        first = FaceFeatureDB(path)
        first.save_feature(make_feature(), '张三')
        first.close()

        second = FaceFeatureDB(path)                       # 再次初始化
        matrix, info = second.get_all_features()
        assert matrix is not None and len(info) == 1
        second.close()


# --------------------------------------- 保存与读取 ---------------------------------------
class TestSaveAndGet:

    def test_save_returns_record_id(self, db):
        record_id = db.save_feature(make_feature(), '李四')
        assert isinstance(record_id, int) and record_id > 0

    def test_blob_roundtrip_is_lossless(self, db):
        feature = make_feature(seed=7)
        db.save_feature(feature, '王五')

        matrix, info = db.get_all_features()
        assert matrix.shape == (1, FEATURE_DIM)
        assert matrix.dtype == np.float32
        assert np.allclose(matrix[0], feature[0])          # BLOB 存取无损
        assert info[0]['name'] == '王五'

    def test_multi_records_matrix_shape(self, db):
        for seed, name in enumerate(['A', 'B', 'C']):
            db.save_feature(make_feature(seed), name)
        matrix, info = db.get_all_features()
        assert matrix.shape == (3, FEATURE_DIM)
        assert [item['name'] for item in info] == ['A', 'B', 'C']

    def test_wrong_shape_returns_none(self, db):
        wrong = np.random.rand(2, FEATURE_DIM).astype(np.float32)
        assert db.save_feature(wrong, '形状错误') is None
        assert db.get_all_features() == (None, None)       # 库未被污染

    def test_non_ndarray_returns_none(self, db):
        assert db.save_feature([[0.0] * FEATURE_DIM], '非数组') is None

    def test_duplicate_name_skipped(self, db):
        assert db.save_feature(make_feature(seed=1), '同名') is not None
        assert db.save_feature(make_feature(seed=2), '同名') is None   # 同名被跳过
        matrix, _ = db.get_all_features()
        assert matrix.shape[0] == 1

    def test_empty_db_returns_none_pair(self, db):
        assert db.get_all_features() == (None, None)


# --------------------------------------- 同名查询 ---------------------------------------
class TestExistsName:

    def test_existing_name_returns_true(self, db):
        db.save_feature(make_feature(), '张三')
        assert db.exists_name('张三') is True

    def test_missing_name_returns_false(self, db):
        db.save_feature(make_feature(), '张三')
        assert db.exists_name('李四') is False

    def test_empty_db_returns_false(self, db):
        assert db.exists_name('张三') is False

    def test_exact_match_not_prefix(self, db):
        """只做精确匹配, 不能把「张三丰」误判为「张三」已存在"""
        db.save_feature(make_feature(), '张三丰')
        assert db.exists_name('张三') is False

    def test_query_is_read_only(self, db):
        """预检不得改变记录数与内容"""
        db.save_feature(make_feature(), '张三')
        before_matrix, before_info = db.get_all_features()

        assert db.exists_name('张三') is True

        after_matrix, after_info = db.get_all_features()
        assert after_matrix.shape == before_matrix.shape
        assert np.array_equal(after_matrix, before_matrix)
        assert after_info == before_info

    def test_after_delete_returns_false(self, db):
        db.save_feature(make_feature(), '待删除')
        assert db.exists_name('待删除') is True
        db.delete_by_name('待删除')
        assert db.exists_name('待删除') is False


# --------------------------------------- 删除 ---------------------------------------
class TestDelete:

    def test_delete_by_name_removes_records(self, db):
        db.save_feature(make_feature(), '删除对象')
        assert db.delete_by_name('删除对象') == 1
        assert db.get_all_features() == (None, None)

    def test_delete_missing_name_returns_zero(self, db):
        assert db.delete_by_name('不存在的人') == 0
