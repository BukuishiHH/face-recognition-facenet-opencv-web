"""
特征库模块(SQLite)
职责: 存储人员身份标签 + 对应的人脸特征向量, 供识别时比对
"""

import os
import time
import sqlite3
import threading
import numpy as np

# 模块级可重入锁: SQLite 单连接不接受并发写, 读操作同样串行化以保证行序一致
_DB_LOCK = threading.RLock()


class FaceFeatureDB:

    def __init__(self, db_path = None):
        # 数据库文件放在本文件同级目录
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_features.db')
        self.db_path = db_path
        self._closed = False
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # check_same_thread=False: FastAPI 同步路由在线程池执行, 需允许跨线程复用连接;
        # 并发写保护由模块级 _DB_LOCK 提供(见 M3-1.7).
        self._create_table()


    # --------------------------------------- 创建人脸特征表 ---------------------------------------
    def _create_table(self):
        """
        人脸特征表不存在时进行创建
        """

        with _DB_LOCK:
            cursor = self.conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature BLOB NOT NULL,               -- 人脸特征向量(二进制存储)
                timestamp DATETIME NOT NULL,         -- 录入时间
                face_id TEXT,                        -- 身份标签(姓名)
                feature_dim INTEGER NOT NULL         -- 特征维度(512)
            )
            ''')
            self.conn.commit()
            cursor.close()


    # --------------------------------------- 保存人脸特征 ---------------------------------------
    def save_feature(self, feature, face_id):
        """
        保存一条人脸特征
        :param feature: 特征向量, 形状(1, 512)
        :param face_id: 身份标签(如"獭獭") 
        :return: 记录ID, 失败则返回 None
        """

        with _DB_LOCK:
            try:
                if not isinstance(feature, np.ndarray):
                    raise ValueError("特征向量必须是NumPy数组")
                if feature.ndim != 2 or feature.shape[0] != 1 or feature.shape[1] != 512:
                    raise ValueError(f"特征形状应为(1, 512), 当前是{feature.shape}")

                # 1. NumPy 数组转二进制(float32 保证字节数一致)
                feature_blob = feature.astype(np.float32).tobytes()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                # 2. 同名特征已存在则跳过, 避免重复录入
                cursor = self.conn.cursor()
                cursor.execute('SELECT id FROM face_features WHERE face_id = ?', (face_id,))
                if cursor.fetchone():
                    print(f"Warning! [{face_id}]的特征已存在(库中有同名记录), 跳过录入.")
                    cursor.close()
                    return None
                cursor.close()

                # 3. 插入
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO face_features (feature, timestamp, face_id, feature_dim)
                VALUES (?, ?, ?, ?)
                ''', (feature_blob, timestamp, face_id, feature.shape[1]))
                self.conn.commit()
                record_id = cursor.lastrowid
                cursor.close()
                print(f"Success! 人脸特征已入库(姓名：{face_id}, 记录ID: {record_id})")
                return record_id
            except Exception as e:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                print(f"Error! 保存人脸特征失败: {e}")
                return None


    # --------------------------------------- 获取特征库所有特征 ---------------------------------------
    def get_all_features(self):
        """
        获取特征库所有特征
        :return: (特征矩阵 (N, 512), 信息列表 [{'id':.., 'name':..}, ...]), 库空返回 (None, None)
        """
        with _DB_LOCK:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, face_id, feature, feature_dim FROM face_features')
            records = cursor.fetchall()
            cursor.close()

            features, info_list = [], []
            for record_id, name, feature_blob, dim in records:
                try:
                    # 二进制转回特征向量
                    feature = np.frombuffer(feature_blob, dtype=np.float32).reshape(1, dim)
                    features.append(feature)
                    info_list.append({'id': record_id, 'name': name})
                except Exception as e:
                    print(f"Error! 恢复特征失败(ID: {record_id}): {e}")
                    continue

            if features:
                return np.vstack(features), info_list
            return None, None


    # --------------------------------------- 删除特征 ---------------------------------------
    def delete_by_name(self, face_id):
        """
        按姓名删除特征
        """
        with _DB_LOCK:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM face_features WHERE face_id = ?', (face_id,))
            self.conn.commit()
            deleted = cursor.rowcount
            cursor.close()
            print(f"已删除[{face_id}]的 {deleted} 条特征" if deleted else f"库中没有[{face_id}]的记录")
            return deleted


    # --------------------------------------- 关闭数据库 ---------------------------------------
    def close(self):
        """
        关闭数据库(幂等, 允许生命周期与测试夹具重复调用)
        """
        with _DB_LOCK:
            if self._closed:
                return
            self.conn.close()
            self._closed = True


# 全局数据库实例
face_db = FaceFeatureDB()