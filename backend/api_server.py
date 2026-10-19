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
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import db_utils
from face_detection import FaceDetection
from face_recognition import FaceRecognition

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
DETECTOR_MODEL = 'haarcascade_frontalface_alt2.xml'
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


# --------------------------------------- 统一响应工具 ---------------------------------------
def ok(data=None, message: str = 'ok') -> dict:
    """成功信封: 路由统一返回 dict, 由 response_model 做二次校验"""
    return {'code': 0, 'message': message, 'data': data}


def fail(code: int, message: str, status_code: int = 200) -> JSONResponse:
    """失败信封: 直接返回 JSONResponse, 绕过 response_model"""
    return JSONResponse(status_code=status_code,
                        content={'code': code, 'message': message, 'data': None})


# --------------------------------------- 生命周期 ---------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化检测器 / 预热 FaceNet / 记录特征库规模, 关闭时释放数据库"""
    os.makedirs(RESULT_DIR, exist_ok=True)

    app.state.detector = FaceDetection(model_name=DETECTOR_MODEL)
    logger.info('检测器就绪: %s', DETECTOR_MODEL)

    if os.getenv('FACENET_WARMUP', '1') == '1':
        logger.info('FaceNet 预热中(首次运行需下载 casia-webface 权重)...')
        FaceRecognition._get_model()
        logger.info('FaceNet 设备: %s', FaceRecognition._device)
    else:
        logger.info('FACENET_WARMUP=0, FaceNet 将在首个识别请求时懒加载')

    try:
        _, info_list = db_utils.face_db.get_all_features()
        logger.info('特征库记录数: %s', 0 if info_list is None else len(info_list))
    except Exception as exc:                              # 记录数读取失败不应阻断服务启动
        logger.warning('读取特征库记录数失败: %s', exc)

    try:
        yield
    finally:
        db_utils.face_db.close()
        logger.info('数据库连接已关闭')


# --------------------------------------- 应用实例 ---------------------------------------
app = FastAPI(
    title='人脸检测识别系统 API',
    description='基于 FaceNet + OpenCV 的人脸检测、特征录入、识别与特征库管理接口',
    version=VERSION,
    lifespan=lifespan,
)

# --------------------------------------- CORS ---------------------------------------
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        'CORS_ORIGINS',
        'http://localhost:5173,http://127.0.0.1:5173,'
        'http://localhost:4173,http://127.0.0.1:4173',
    ).split(',')
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
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


# --------------------------------------- 统一异常处理 ---------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Pydantic 结构校验失败: 422 + 统一信封"""
    first = exc.errors()[0]
    field = '.'.join(str(x) for x in first.get('loc', []) if x != 'body')
    return JSONResponse(
        status_code=422,
        content={
            'code': 422,
            'message': f'参数校验失败: {field} {first.get("msg", "")}',
            'data': None,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """未捕获异常兜底: 1007 + 统一信封, 异常栈只进日志"""
    logger.exception('未捕获异常 %s %s: %s', request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={'code': 1007, 'message': '服务内部错误', 'data': None},
    )


# --------------------------------------- 系统接口 ---------------------------------------
@app.get('/api/health', response_model=ApiResponse[HealthData], tags=['system'])
def health():
    """健康检查: 服务状态 / 设备 / 检测器 / 特征维度 / 特征库记录数"""
    _, info_list = db_utils.face_db.get_all_features()
    return ok(HealthData(
        status='ok',
        version=VERSION,
        device=FaceRecognition._device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        detector=DETECTOR_MODEL,
        feature_dim=FEATURE_DIM,
        library_size=0 if info_list is None else len(info_list),
    ))

