"""
FastAPI 接口层入口

职责:
  1. 创建 app 实例, 提供 /docs 与 /openapi.json;
  2. 定义公共常量与 Pydantic 数据模型;
  3. 管理检测器 / FaceNet / SQLite 的启动与关闭;
  4. 以 HTTP + JSON 暴露阶段 2 的算法能力.

边界:
  - 不复制人脸检测、特征提取、特征库存取等算法实现;
  - 不出现 input() / cv2.imshow() 等控制台交互.
"""

import logging
import os

from fastapi import FastAPI

# --------------------------------------- 日志 ---------------------------------------
logger = logging.getLogger('api_server')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# --------------------------------------- 公共常量 ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_DIM = 512                       # FaceNet 输出维度
DEFAULT_THRESHOLD = 0.6                 # 识别阈值默认值
THRESHOLD_MIN = 0.3                     # 阈值合法下限
THRESHOLD_MAX = 0.95                    # 阈值合法上限
MAX_UPLOAD_BYTES = 5 * 1024 * 1024      # 图片上传上限 5 MB
MAX_FRAME_SIDE = 1280                   # 推理前长边缩放上限
MAX_NAME_LEN = 32                       # 姓名长度上限
RESULT_DIR = os.path.join(BASE_DIR, 'result_output')
STATIC_PREFIX = '/static'               # 标注结果图静态资源前缀
VERSION = '1.0.0'

# --------------------------------------- 应用实例 ---------------------------------------
app = FastAPI(
    title='人脸检测识别系统 API',
    description='基于 FaceNet + OpenCV 的人脸检测、特征录入、识别与特征库管理接口',
    version=VERSION,
)
