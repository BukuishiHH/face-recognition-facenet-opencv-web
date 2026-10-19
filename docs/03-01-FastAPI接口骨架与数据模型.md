# Step 3-1 · FastAPI 接口骨架与数据模型

> **所属阶段**：阶段 3 · Web 后端接口层（FastAPI）
> **对应总纲**：`docs/development_steps.md` → Step 3-1（FastAPI 项目骨架与公共模型定义）
> **源码文件**：`backend/api_server.py`（新建）、`backend/db_utils.py`（线程安全适配）
> **文档状态**：✅ 已实现（2026-10-19，M3-1.1 ~ M3-1.8 全部完成）

---

## 一、功能背景

阶段 2 交付的四个模块（检测 / 特征 / 存储 / 编排）只能以 **CLI 菜单 + `input()`** 的形式使用，能力虽已闭环，但存在两个硬伤：

1. **调用方被限制在本机控制台**——浏览器、Vue 前端、桌面 GUI 都无法直接调用 Python 函数；
2. **业务逻辑与交互耦合**——`main.py` 的 `register_face_from_image()` 等函数内部直接 `input()` 取路径与姓名、直接 `cv2.imshow()` 预览，本质上无法被远程复用。

因此阶段 3 的核心动作是**服务化**：新增 `backend/api_server.py`，把阶段 2 的算法能力以 HTTP + JSON 暴露出去，供阶段 4 的 Vue 前端调用。接口层只做「协议适配 + 业务编排」，**不得复制算法实现**，也**不得出现 `input()` / `imshow()`**。

技术选型对比：

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **FastAPI** | Pydantic 自动校验；原生 OpenAPI，`/docs` 可直接点测；同步 `def` 路由自动丢线程池，适配 CPU 密集推理 | 需 uvicorn 部署 | ✅ 选用 |
| Flask | 极轻量、上手快 | 无内置数据校验与接口文档，模型定义要手写 | ✗ |
| Django REST Framework | 生态完整、权限体系强 | 组件重，与本项目「单服务单算法」定位不符 | ✗ |
| gRPC | 二进制传输、性能高 | 浏览器需 grpc-web 代理，前端接入成本高 | ✗ |

---

## 二、开发目标

| 目标 | 说明 |
|------|------|
| 服务骨架 | `backend/api_server.py` 内创建 `app`，可由 `uvicorn api_server:app` 直接启动 |
| 公共模型 | 用 Pydantic 定义「录入请求 / 识别结果 / 特征库列表项 / 统一响应」等全部出入参模型 |
| 生命周期 | 启动时初始化检测器、结果目录、数据库，并预热 FaceNet；关闭时释放数据库连接 |
| CORS | 放行 Vite dev server（5173）与 preview（4173），保证 Vue 跨域调用不被浏览器拦截 |
| 统一响应 | 全部业务接口返回统一信封 `{code, message, data}`，前端只需一套拦截逻辑 |
| 接口文档 | `http://127.0.0.1:8000/docs` 可访问，模型与接口分组清晰 |
| 线程安全 | 适配 FastAPI 线程池模型：SQLite 连接跨线程可用、单帧推理串行化 |
| 静态资源 | 可选挂载 `/static` → `backend/result_output/`，供前端预览标注结果图 |

---

## 三、功能职责

`backend/api_server.py` 是**唯一的 HTTP 入口**，承担 6 类职责：

| 分类 | 内容 | 职责 |
|------|------|------|
| 应用实例 | `app = FastAPI(title=..., version=..., lifespan=...)` | 挂载中间件、异常处理器、路由与静态目录 |
| 公共常量 | 特征维度、阈值区间、上传上限、缩放上限、版本号 | 全接口共用，避免散落魔法数字 |
| 数据模型 | `ApiResponse` / `FaceBox` / `FaceResult` / `RegisterData` / `FrameRegisterRequest` / `FrameRecognizeRequest` / `RecognizeData` / `LibraryItem` / `LibraryData` / `DeleteData` / `HealthData`（共 11 个） | 出入参契约，供 `/docs` 自动生成文档 |
| 解码工具 | `decode_base64_image()` / `decode_upload_image()` / `limit_max_side()` | 把「上传字节」与「base64 字符串」统一转成 BGR `np.ndarray` |
| 响应工具 | `ok()` / `fail()` | 生成统一信封，避免各路由重复拼字典 |
| 生命周期 | `lifespan()` | 加载 Haar、预热 FaceNet、关闭数据库 |

**不做的事**（职责边界）：

- 不做人脸检测算法（→ `face_detection.py`）；
- 不做特征提取与余弦比对（→ `face_recognition.py`）；
- 不做 Blob 存取（→ `db_utils.py`）；
- 不做画框、保存图片的中文渲染（→ 复用 `main.py` 的 `draw_results` / `save_result`）；
- 不做前端页面与 canvas 叠加绘制（→ 阶段 4）。

## 四、功能逻辑

### 4.1 应用骨架

```python
# backend/api_server.py （骨架示意，具体路由在 03-02 ~ 03-04 展开）
import base64
import binascii
import logging
import os
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Generic, TypeVar

import db_utils
from db_utils import FaceFeatureDB
from face_detection import FaceDetection
from face_recognition import FaceRecognition

logger = logging.getLogger("api_server")
```

> **导入约定**：数据库一律通过 `db_utils.face_db` **属性方式**访问，而不是 `from db_utils import face_db`。
> 原因见「七、异常点与坑点」第 6 条（`from ... import` 会在导入时绑定值，运行期替换单例不生效，测试与多 worker 场景难以隔离）。

### 4.2 公共常量

| 常量 | 取值 | 用途 |
|------|------|------|
| `FEATURE_DIM` | `512` | FaceNet 输出维度，用于响应与校验提示 |
| `DEFAULT_THRESHOLD` | `0.6` | 识别阈值默认值（与 `main.py` / `README` 一致） |
| `THRESHOLD_MIN` / `THRESHOLD_MAX` | `0.3` / `0.95` | 阈值合法区间，越界返回 `1001` |
| `MAX_UPLOAD_BYTES` | `5 * 1024 * 1024` | 图片上传上限 5 MB |
| `MAX_FRAME_SIDE` | `1280` | 推理前长边缩放上限（大图降采样，坐标按比例还原） |
| `MAX_NAME_LEN` | `32` | 姓名长度上限（去空格后校验） |
| `RESULT_DIR` | `backend/result_output/` | 标注结果图落盘目录 |
| `DETECTOR_MODEL` | `haarcascade_frontalface_alt2.xml` | 检测器文件名，生命周期与健康检查共用 |
| `VERSION` | `1.0.0` | 写入 `/api/health` 与 OpenAPI `version` |

### 4.3 生命周期（lifespan）

```text
uvicorn 启动
   │
   ├─ 创建 RESULT_DIR（makedirs exist_ok）
   ├─ app.state.detector = FaceDetection('haarcascade_frontalface_alt2.xml')
   │      └─ 失败 → RuntimeError（opencv 5.x / 模型缺失），进程启动即失败，避免带病运行
   ├─ FACENET_WARMUP=1（默认）→ FaceRecognition._get_model()
   │      ├─ 首次运行会下载 casia-webface 权重（数十 MB），启动耗时 10s~2min
   │      └─ 离线部署可置 FACENET_WARMUP=0，改为首个识别请求懒加载
   └─ 记录日志：设备(cuda/cpu)、特征维度、特征库记录数
        │
        ▼
    请求处理（阶段 3-2 / 3-3 / 3-4）
        │
        ▼
uvicorn 关闭 → db_utils.face_db.close() 释放 SQLite 句柄
```

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(RESULT_DIR, exist_ok=True)
    app.state.detector = FaceDetection(model_name='haarcascade_frontalface_alt2.xml')
    if os.getenv('FACENET_WARMUP', '1') == '1':
        FaceRecognition._get_model()          # 预热，消除首个请求 3~5s 延迟与懒加载竞态
    yield
    db_utils.face_db.close()
```

> **模型不在 import 时加载**：`main.py` 是「导入即加载 Haar + 连接数据库」，这对 CLI 单例是合理的；但接口层若沿用该写法，会导致 pytest 收集用例时就下载 FaceNet 权重、`uvicorn --reload` 每次重载重复占用显存。故接口层改为**生命周期内加载**。

### 4.4 CORS 配置

| 配置项 | 取值 | 说明 |
|--------|------|------|
| `allow_origins` | `CORS_ORIGINS` 环境变量，默认 `http://localhost:5173`、`http://127.0.0.1:5173`、`http://localhost:4173`、`http://127.0.0.1:4173` | Vite `dev` / `preview` 端口 |
| `allow_credentials` | `False` | 接口不使用 Cookie，关闭后避免「`*` + credentials」的浏览器限制 |
| `allow_methods` | `["*"]` | 需覆盖 `OPTIONS` 预检 |
| `allow_headers` | `["*"]` | 放行 `Content-Type` 等自定义头 |

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,'
                                            'http://localhost:4173,http://127.0.0.1:4173').split(','),
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)
```

### 4.5 数据模型清单

```python
T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    """统一响应信封：code=0 成功，非 0 为业务错误码"""
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
    name: str | None = None           # 匹配到的人名，未匹配为 null
    similarity: float = 0.0           # 最高余弦相似度（0~1）
    matched: bool = False             # 是否达到阈值


class RegisterData(BaseModel):
    record_id: int                    # SQLite 记录 ID
    name: str
    face_count: int                   # 图中检测到的人脸总数
    selected: FaceBox                 # 实际入库的那张人脸框
    feature_dim: int = 512


class FrameRegisterRequest(BaseModel):
    """摄像头帧入库请求体（docs/03-02）"""
    name: str = Field(..., description='姓名，路由内 strip 后校验 1~32 字符')
    image_base64: str = Field(..., description='base64 或 dataURL，路由内校验可解码')


class FrameRecognizeRequest(BaseModel):
    """摄像头帧识别请求体（docs/03-03）"""
    image_base64: str = Field(..., description='base64 或 dataURL，路由内校验可解码')
    threshold: float = Field(DEFAULT_THRESHOLD, description='阈值 0.3~0.95，路由内校验')


class RecognizeData(BaseModel):
    total_faces: int
    matched_count: int
    elapsed_ms: int
    library_empty: bool = False       # 库为空时置 True，前端提示「请先录入」
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
```

**模型设计要点**：

1. `FaceBox` 单独抽层，前端 canvas 叠加时直接读 `box.x / box.y / box.w / box.h`，无需记 `[x, y, w, h]` 下标；
2. `FaceResult.matched` 与 `name` 冗余存在——`name` 供展示，`matched` 供前端统计与样式分支（绿框/红框），避免前端再比较阈值；
3. `RecognizeData.elapsed_ms` 由服务端计时，便于阶段 5 性能验收与前端「耗时」展示；
4. 所有模型字段均可 JSON 序列化（`int` / `float` / `str` / `bool` / `null`），**不返回 ndarray**。
5. **校验分层**：Pydantic 只做结构校验（缺字段、类型错误 → `422`）；长度、取值范围、空串等**业务校验一律在路由内完成**（→ `1001`）。因此 `FrameRegisterRequest.name` / `image_base64` 与 `FrameRecognizeRequest.threshold` 不加 `min_length` / `ge` / `le`，否则同一类错误在图片接口返回 `1001`、帧接口返回 `422`，前端需写两套分支。

### 4.6 统一响应与错误码

**约定**：全部业务接口一律返回 HTTP 200 + `{code, message, data}`；前端只按 `code` 分支，HTTP 层异常仅保留「并发保护 429」与「框架校验 422」两种。

| code | HTTP | 含义 | 触发场景 | 建议前端处理 |
|------|------|------|----------|--------------|
| `0` | 200 | 成功 | 正常返回 | 取 `data` 渲染 |
| `1001` | 200 | 请求参数非法 | 姓名为空/超长、文件为空、base64 无法解码、图片损坏、阈值越界、文件超过 5 MB | 红字提示 `message` |
| `1002` | 200 | 未检测到人脸 | 图片/帧中无人脸 | 提示「请正对摄像头或更换图片」 |
| `1003` | 200 | 特征提取失败 | ROI 裁剪为空、模型前向异常 | 提示重试并记录日志 |
| `1004` | 200 | 同名特征已存在 | `save_feature` 查重命中 | 提示「先删除旧记录再录入」 |
| `1005` | 200 | 记录不存在 | 删除返回 0 行 | 提示并刷新列表 |
| `1007` | 500 | 服务内部错误 | 未捕获异常（兜底处理器） | 提示「服务异常」 |
| `1008` | 429 | 推理服务繁忙 | 实时帧接口并发超过 1 帧在算 | 前端丢帧，不提示用户 |
| `422` | 422 | 请求体校验失败 | Pydantic 解析失败（字段缺失/类型错误） | 提示 `message` 首条错误 |

```python
def ok(data=None, message: str = 'ok') -> dict:
    """成功信封：路由统一返回 dict，由 response_model 做二次校验"""
    return {'code': 0, 'message': message, 'data': data}


def fail(code: int, message: str, status_code: int = 200):
    return JSONResponse(status_code=status_code, content={'code': code, 'message': message, 'data': None})


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    first = exc.errors()[0]
    field = '.'.join(str(x) for x in first.get('loc', []) if x != 'body')
    return JSONResponse(status_code=422,
                       content={'code': 422, 'message': f'参数校验失败: {field} {first.get("msg", "")}',
                                'data': None})
```

> `ok()` 返回 dict 后仍会经过 `response_model` 校验；`fail()` 直接返回 `JSONResponse`，绕过 `response_model`，因此错误响应中的 `data` 固定为 `null`。
> 另需注册 `@app.exception_handler(Exception)` 兜底，返回 `1007`，避免异常栈泄露到前端。

### 4.7 图片解码与缩放工具

```python
def decode_base64_image(raw: str):
    """base64 / dataURL → BGR ndarray；失败返回 None"""
    if not raw:
        return None
    if ',' in raw and raw.strip().startswith('data:'):
        raw = raw.split(',', 1)[1]                  # 剥离 'data:image/jpeg;base64,' 前缀
    raw = ''.join(raw.split())                      # 去除换行/空格
    try:
        binary = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not binary:
        return None
    return cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_upload_image(raw: bytes):
    """上传文件字节 → BGR ndarray；失败返回 None"""
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def limit_max_side(image, max_side: int = MAX_FRAME_SIDE):
    """长边超过 max_side 时等比缩小，返回 (缩放后图像, 缩放系数 scale)"""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image, 1.0
    scale = max_side / long_side
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale
```

> **坐标契约**：响应中的 `box` 必须位于**原图坐标系**。内部缩放后检测到的框要除以 `scale` 还原（`x/scale`、`w/scale`），否则前端 canvas 叠加会整体偏移。

### 4.8 路由总表

| # | 方法 | 路径 | 请求体类型 | 响应模型 | 归属文档 |
|---|------|------|------------|----------|----------|
| 1 | `GET` | `/api/health` | — | `ApiResponse[HealthData]` | 本文档 |
| 2 | `POST` | `/api/face/register/image` | `multipart/form-data`（`file` + `name`） | `ApiResponse[RegisterData]` | `docs/03-02` |
| 3 | `POST` | `/api/face/register/camera-frame` | JSON（`name` + `image_base64`） | `ApiResponse[RegisterData]` | `docs/03-02` |
| 4 | `POST` | `/api/recognize/image` | `multipart/form-data`（`file` + `threshold` + `save_result`） | `ApiResponse[RecognizeData]` | `docs/03-03` |
| 5 | `POST` | `/api/recognize/frame` | JSON（`image_base64` + `threshold`） | `ApiResponse[RecognizeData]` | `docs/03-03` |
| 6 | `GET` | `/api/face/library` | — | `ApiResponse[LibraryData]` | `docs/03-04` |
| 7 | `DELETE` | `/api/face/library` | query（`name`） | `ApiResponse[DeleteData]` | `docs/03-04` |

接口分组（OpenAPI tags）：`system`（健康检查）、`face`（采集与库管理）、`recognize`（识别推理）。

`/api/health` 实现示意：

```python
@app.get('/api/health', response_model=ApiResponse[HealthData], tags=['system'])
def health():
    matrix, info = db_utils.face_db.get_all_features()
    return ok(HealthData(
        status='ok',
        version=VERSION,
        device=FaceRecognition._device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        detector='haarcascade_frontalface_alt2.xml',
        feature_dim=FEATURE_DIM,
        library_size=0 if info is None else len(info),
    ))
```

### 4.9 线程模型与数据库适配

FastAPI 中 **`def`（同步）路由会被丢进线程池执行**，与启动线程不是同一个线程；而 `sqlite3` 默认禁止跨线程复用连接：

```text
uvicorn 主线程 ──创建──► 全局 face_db（sqlite3 连接，创建线程 = 主线程）
      │
      ├─ 线程池 worker#1 ── SELECT/INSERT ──► sqlite3.ProgrammingError:
      └─ 线程池 worker#2 ── SELECT/INSERT        SQLite objects created in a thread
                                                 can only be used in that same thread
```

因此 Step 3-1 必须同时对 `db_utils.py` 做**最小改造**（不改变既有 CLI 行为）：

| 改造点 | 内容 | 原因 |
|--------|------|------|
| 连接参数 | `sqlite3.connect(self.db_path, check_same_thread=False)` | 允许线程池 worker 复用同一连接 |
| 写锁 | 模块级 `_DB_LOCK = threading.RLock()`，在 `save_feature` / `delete_by_name` 内加锁 | SQLite 单连接不接受并发写；`RLock` 允许同线程嵌套调用 |
| 读一致性 | `get_all_features` 读操作同样加锁（或使用同一把锁） | 避免「读一半 + 写」交叠导致行序错位 |

推理侧：`torch` 模型可被多线程只读调用，但**实时帧接口必须串行**（同一时刻只允许 1 帧在算），否则 GPU 显存/CPU 线程争抢会让每一帧都变慢，见 `docs/03-03` 的 `threading.Lock` 丢帧策略。

> **实施顺序**：M3-1.3 生命周期需要在非创建线程内执行 `close()`，因此先放开 `check_same_thread=False`；M3-1.7 再补齐模块级 `RLock` 与 20 并发验证。

部署约定：`uvicorn api_server:app --host 0.0.0.0 --port 8000`，**单 worker**（`--workers 1`）。多 worker 会让每个进程各加载一份 FaceNet（显存 × N）并各自持有 SQLite 连接，与单机演示定位不符。

---

## 五、里程碑计划

| 里程碑 | 交付物 | 判定标准 | 状态 |
|--------|--------|----------|------|
| M3-1.1 | 应用骨架 | `uvicorn api_server:app` 启动无异常，`/docs` 返回 200 | ✅ 已完成（2026-10-19，真实 `uvicorn` 启动成功，`/docs`、`/openapi.json` 均 200） |
| M3-1.2 | 公共数据模型 | 11 个模型落地，`/openapi.json` 中可见完整 schema | ✅ 已完成（2026-10-19，`components.schemas` 含 11 个公共模型，另含路由泛型实例 `ApiResponse_HealthData_`） |
| M3-1.3 | 生命周期管理 | 启动加载 Haar、预热 FaceNet、退出 `close()` 数据库 | ✅ 已完成（2026-10-19，`FACENET_WARMUP=1` 预热至 CUDA；`=0` 懒加载，启动/关闭正常） |
| M3-1.4 | CORS 配置 | 带 `Origin: http://localhost:5173` 的 `OPTIONS` 预检返回 200 且带 `access-control-allow-origin` | ✅ 已完成（2026-10-19，`localhost` / `127.0.0.1` 两种 origin 预检均通过） |
| M3-1.5 | 统一响应与异常处理 | 业务错误返回 `code != 0`；`422` 与未捕获异常均返回统一信封 | ✅ 已完成（2026-10-19，`/api/health` 返回 `code=0`；`422` 与 `1007` 均为 `{code,message,data}`） |
| M3-1.6 | 解码 / 缩放工具 | `base64`、`dataURL`、上传字节三路均可解码；缩放后坐标可按 `scale` 还原 | ✅ 已完成（2026-10-19，三路解码一致；2000×1000 缩放至 1280×640，坐标还原正确） |
| M3-1.7 | 数据库线程安全适配 | 并发 20 次读写无 `ProgrammingError`，数据无错行 | ✅ 已完成（2026-10-19，20 线程并发写入 20 条记录无异常，数量/姓名一致；`close()` 可重复调用） |
| M3-1.8 | 静态资源挂载 | `/static/<file>.jpg` 可访问 `result_output/` 内文件 | ✅ 已完成（2026-10-19，`/static/recognize_121224500.jpg` 返回 200 且 `content-type=image/jpeg`） |

### 5.1 实施记录（2026-10-19）

| 里程碑 | 提交 | 核心改动 |
|--------|------|----------|
| M3-1.1 | `24b9f04` | 创建 `api_server.py`，落地公共常量与 FastAPI 实例 |
| M3-1.2 | `933dc33` | 定义 11 个 Pydantic 模型；自定义 `_build_openapi` 补齐未被路由引用的 schema |
| M3-1.3 | `9239484` | `lifespan` 内加载 Haar、按 `FACENET_WARMUP` 预热 FaceNet、退出时关闭数据库 |
| M3-1.4 | `4a0ea07` | CORS 默认放行 5173 / 4173 的 `localhost` 与 `127.0.0.1`，支持环境变量覆盖 |
| M3-1.5 | `b667b5c` | `ok()` / `fail()` 统一信封、422 校验处理、1007 兜底、`GET /api/health` |
| M3-1.6 | `57d6865` | `decode_base64_image` / `decode_upload_image` / `limit_max_side` |
| M3-1.7 | `940bad9` | `db_utils` 增加模块级 `RLock`，读写与 `close()` 串行化，`close()` 幂等 |
| M3-1.8 | `aea1046` | 将 `result_output/` 挂载到 `/static` |

### 5.2 阶段 3-1 验证结果

| 验证项 | 结果 |
|--------|------|
| 全量回归 | `python -m pytest -q` → **54 passed**（`testpaths = tests`，含阶段 2 算法与 CLI 用例） |
| 真实启动（`FACENET_WARMUP=1`） | 日志：`检测器就绪` → `FaceNet 设备: cuda` → `特征库记录数: 2` → `Application startup complete` |
| `GET /api/health` | `200`，`code=0`，`device=cuda`，`feature_dim=512`，`library_size=2` |
| `/docs` / `/openapi.json` | 均 `200`；`components.schemas` 共 12 项（11 个公共模型 + 路由泛型实例 `ApiResponse_HealthData_`）；阶段 3-1 仅有 `GET /api/health` 路由 |
| CORS 预检 | `localhost:5173` 与 `127.0.0.1:5173` 的 `OPTIONS` 均 `200`，且回显 `access-control-allow-origin` |
| 静态资源 | `/static/recognize_121224500.jpg` → `200`、`image/jpeg`、15416 B |
| 数据库并发 | 20 线程并发写入 20 条记录无异常，数量/姓名一致；20 线程并发删除后剩余 0 条；`close()` 重复调用正常 |
| 失败信封 | `422` 与未捕获异常 `1007` 均为 `{code, message, data}`；`1001~1005`、`1008` 复用 `fail()`，随业务路由验证 |

> 阶段 3-1 只交付「骨架 + 公共能力」。`/api/face/*` 与 `/api/recognize/*` 路由分别由 `docs/03-02`、`docs/03-03`、`docs/03-04` 实现，届时 `/docs` 会增加 `face` / `library` / `recognize` 分组。

---

## 六、验收清单

- [x] `uvicorn api_server:app --host 0.0.0.0 --port 8000` 可正常启动，无 import 期副作用（实测 `127.0.0.1:8000` 启动成功）
- [x] `http://127.0.0.1:8000/docs` 可访问；阶段 3-1 仅 `system` 分组，`face` / `library` / `recognize` 随 `03-02 ~ 03-04` 接入
- [x] `GET /api/health` 返回 `code=0`，含 `device`、`feature_dim=512`、`library_size`
- [x] Pydantic 模型字段类型、默认值与 `docs/03-02 ~ 03-04` 示例逐字段核对一致
- [x] 启动日志明确输出「检测器就绪 / FaceNet 设备 / 特征库记录数」
- [x] `FACENET_WARMUP=0` 时启动不加载权重，首个识别请求才加载（M3-1.3 已验证）
- [x] CORS 预检通过，`localhost` 与 `127.0.0.1` 两种 origin 均被放行
- [x] 统一错误信封已落地：`422` 与 `1007` 已实测；`1001~1005`、`1008` 复用 `fail()`，随业务路由验证
- [ ] 传入 5 MB 以上文件返回 `1001`，不进入推理（依赖 `/api/face/register/image`，待 `03-02` 实现）
- [x] 数据库跨线程访问无异常，20 并发写入后记录数正确
- [x] `/static/` 挂载路径与 `RESULT_DIR` 一致，未挂载时不影响主流程

---

## 七、异常点与坑点

| # | 问题 | 现象 | 定位与处置 |
|---|------|------|------------|
| 1 | **SQLite 跨线程访问** | 同步路由在线程池执行，`face_db` 在主线程创建 → `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread` | `sqlite3.connect(..., check_same_thread=False)` + 模块级 `RLock` 包住读写；已在本文档 4.9 列为 Step 3-1 必做改造 |
| 2 | **FaceNet 懒加载竞态** | 两个并发识别请求同时看到 `cls._model is None`，各自加载一次权重，显存翻倍且日志重复 | 生命周期内预热（默认）；若关闭预热，则 `_get_model` 内加 `threading.Lock` 双检 |
| 3 | **导入即加载模型** | 沿用 `main.py` 的「模块级 `face_detector = FaceDetection()`」写法，pytest 收集用例、`--reload` 重载都会触发加载 | 接口层改为 `lifespan` 内初始化并挂到 `app.state` |
| 4 | **`/docs` 依赖 CDN** | 内网/离线环境下 Swagger UI 的 js/css 从 CDN 拉取，页面空白 | 调试时改用 `/openapi.json`；或将 Swagger 静态资源本地化后再挂载 |
| 5 | **CORS origin 精确匹配** | 只放行 `http://localhost:5173`，但浏览器访问的是 `http://127.0.0.1:5173` → 被拦 | 两个 host 各放行一份（默认已配置 4 条）；环境变量可覆盖 |
| 6 | **`from db_utils import face_db` 绑定值** | 导入时即绑定对象，运行期把 `db_utils.face_db` 换成测试库后，`face_recognition.py` / `api_server.py` 仍指向旧库 | 接口层统一 `import db_utils` + `db_utils.face_db.xxx`；测试只需替换一处（详见 `docs/03-05`） |
| 7 | **`Form` / `File` 需要 `python-multipart`** | 未安装时启动即报 `RuntimeError: Form data requires "python-multipart"` | `requirements.txt` 已声明，环境搭建阶段需确认可导入 |
| 8 | **dataURL 前缀未剥离** | 前端 `canvas.toDataURL()` 传入 `data:image/jpeg;base64,/9j/...`，直接 `b64decode` 抛 `binascii.Error` | `decode_base64_image()` 先按 `,` 切分剥离前缀，再 `validate=True` 解码 |
| 9 | **大图未降采样** | 4000×3000 照片 Haar 检测耗时数百 ms，实时帧接口超时堆积 | 推理前 `limit_max_side(1280)`；**框坐标必须除以 `scale` 还原到原图**，否则前端框整体偏小偏移 |
| 10 | **多 worker 重复加载模型** | `--workers 4` 时 4 份 FaceNet + 4 个 SQLite 连接，显存暴涨、库文件锁冲突 | 单 worker 部署；如需横向扩展，应把模型推理拆为独立服务 |
| 11 | **`--reload` 下的日志重复** | reloader 父进程 + 子进程各打印一次启动日志，误以为加载两次 | 仅子进程执行 lifespan，属正常现象；生产禁用 `--reload` |
| 12 | 时区/时间格式 | `db_utils` 存的是本地时间字符串，接口直接透出 | 与 CLI 保持一致；如需跨时区，见 `docs/02-03` 坑点 9 |

---

## 附：验证方法（阶段 3-1 冒烟）

```powershell
# 0. 进入后端目录并使用固定解释器
cd backend

# 1. 启动服务（单 worker；首次启动会下载 FaceNet 权重）
$env:FACENET_WARMUP = "1"
& 'D:\anaconda\envs\cv_demo\python.exe' -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# 2. 健康检查 + 接口文档
curl.exe http://127.0.0.1:8000/api/health
Start-Process http://127.0.0.1:8000/docs

# 3. CORS 预检（应返回 200，且响应头含 access-control-allow-origin）
curl.exe -i -X OPTIONS http://127.0.0.1:8000/api/health `
  -H "Origin: http://localhost:5173" `
  -H "Access-Control-Request-Method: GET"
```

```python
# 4. 进程内冒烟（pytest 同款方式，必须用 with 触发 lifespan）
from fastapi.testclient import TestClient
from api_server import app

with TestClient(app) as client:
    print(client.get('/api/health').json())          # code=0, feature_dim=512
    print(client.get('/openapi.json').status_code)   # 200
    # 422 信封验证：缺少必填字段
    print(client.post('/api/recognize/frame', json={}).json())
```

