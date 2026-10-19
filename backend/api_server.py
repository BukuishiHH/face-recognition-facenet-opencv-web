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
from typing import Generic, TypeVar

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

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

# --------------------------------------- 公共数据模型 ---------------------------------------
T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    """统一响应信封: code=0 成功, 非 0 为业务错误码"""
    code: int = 0
    message: str = 'ok'
    data: T | None = None


class FaceBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceResult(BaseModel):
    box: FaceBox                      # 原图坐标系下的人脸框
    name: str | None = None           # 匹配到的人名, 未匹配为 null
    similarity: float = 0.0           # 最高余弦相似度(0~1)
    matched: bool = False             # 是否达到阈值


class RegisterData(BaseModel):
    record_id: int                    # SQLite 记录 ID
    name: str
    face_count: int                   # 图中检测到的人脸总数
    selected: FaceBox                 # 实际入库的那张人脸框
    feature_dim: int = FEATURE_DIM


class FrameRegisterRequest(BaseModel):
    """摄像头帧入库请求体(docs/03-02)"""
    name: str = Field(..., description='姓名, 路由内 strip 后校验 1~32 字符')
    image_base64: str = Field(..., description='base64 或 dataURL, 路由内校验可解码')


class FrameRecognizeRequest(BaseModel):
    """摄像头帧识别请求体(docs/03-03)"""
    image_base64: str = Field(..., description='base64 或 dataURL, 路由内校验可解码')
    threshold: float = Field(DEFAULT_THRESHOLD, description='阈值 0.3~0.95, 路由内校验')


class RecognizeData(BaseModel):
    total_faces: int
    matched_count: int
    elapsed_ms: int
    library_empty: bool = False       # 库为空时置 True, 前端提示[请先录入]
    faces: list[FaceResult] = []
    result_image_url: str | None = None   # save_result=true 时返回 /static/xxx.jpg


class LibraryItem(BaseModel):
    id: int
    name: str
    timestamp: str                    # 'YYYY-MM-DD HH:MM:SS'
    feature_dim: int


class LibraryData(BaseModel):
    total: int
    items: list[LibraryItem] = []


class DeleteData(BaseModel):
    name: str
    deleted: int                      # 实际删除行数


class HealthData(BaseModel):
    status: str                       # 'ok'
    version: str
    device: str                       # 'cuda' / 'cpu'
    detector: str                     # 'haarcascade_frontalface_alt2.xml'
    feature_dim: int
    library_size: int


# --------------------------------------- 应用实例 ---------------------------------------
app = FastAPI(
    title='人脸检测识别系统 API',
    description='基于 FaceNet + OpenCV 的人脸检测、特征录入、识别与特征库管理接口',
    version=VERSION,
)

# --------------------------------------- OpenAPI 扩展 ---------------------------------------
_OPENAPI_MODELS = (
    ApiResponse, FaceBox, FaceResult, RegisterData, FrameRegisterRequest,
    FrameRecognizeRequest, RecognizeData, LibraryItem, LibraryData,
    DeleteData, HealthData,
)


def _build_openapi():
    """在自动生成的基础上补齐暂未被路由引用的公共模型 schema"""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    components = schema.setdefault('components', {}).setdefault('schemas', {})
    for model in _OPENAPI_MODELS:
        model_schema = model.model_json_schema(ref_template='#/components/schemas/{model}')
        for name, definition in model_schema.pop('$defs', {}).items():
            components.setdefault(name, definition)
        components.setdefault(model.__name__, model_schema)
    app.openapi_schema = schema
    return schema


app.openapi = _build_openapi

