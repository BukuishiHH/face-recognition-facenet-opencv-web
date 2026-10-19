# 人脸检测识别系统 · FaceRecognition (OpenCV + FaceNet)

> **基于 OpenCV Haar 级联 + FaceNet 的人脸检测识别系统**  
> 前端 Vue3 · 后端 Python FastAPI · 双运行模式（CLI / Web）

---

## 📋 项目简介

本系统完整实现了**人脸检测 → 人脸识别**的全链路流程：

| 阶段 | 说明 |
| --- | --- |
| **图像源获取** | 本地图片上传 / 摄像头视频流（WebRTC MediaDevices） |
| **图像预处理** | 灰度转换 → 高斯降噪 → 直方图光照均衡 |
| **人脸检测** | OpenCV Haar 级联分类器定位人脸坐标，ROI 裁剪人脸区域 |
| **人脸识别** | FaceNet 提取 512 维特征向量；SQLite 存储特征库；余弦相似度完成身份比对 |
| **结果可视化** | 绘制人脸框、身份标签、相似度分数 |
| **资源释放** | 摄像头、窗口、数据库连接自动关闭 |

### 三种运行方式

| 模式 | 入口 | 适用场景 |
| --- | --- | --- |
| **CLI 模式** | `backend/main.py` | 原有控制台菜单，用于算法调试与快速验证 |
| **Web 模式**（推荐） | FastAPI 后端 + Vue3 前端 | 浏览器完成全部业务功能 |
| **PyQt 桌面版**（备选） | 切换分支 `pyqt-desktop` | 桌面端本地应用 |

---

## 🛠 技术栈

### 后端（Python 3.11）

| 组件 | 用途 |
| --- | --- |
| `OpenCV-Python 4.8.x` | Haar 人脸检测（必须 4.8.x：5.x 移除了 CascadeClassifier，4.14.x 要求 NumPy 2） |
| `NumPy 1.26.x` | 数组运算（锁定 1.x，与 torch 2.2.2 / opencv 4.8.1 的 ABI 匹配） |
| `FaceNet-Pytorch` | 人脸特征提取（512 维） |
| `SQLite3` | 存储人脸身份标签与二进制特征向量 |
| `FastAPI` | 提供 HTTP 接口供 Vue 前端调用 |
| `Uvicorn` | ASGI 服务器 |

### 前端（Vue3 + Vite）

| 组件 | 用途 |
| --- | --- |
| `Vue3 + Vite` | 前端框架与构建工具 |
| `Axios` | 后端接口请求 |
| `MediaDevices` | Web 端摄像头采集 |

---

## 📁 目录结构

```
face-recognition-facenet-opencv-web/
├── backend/                    # Python 人脸算法后端
│   ├── sample_faces/           # 人脸样本集（建议 ≥5 张，覆盖不同光照/角度/距离）
│   ├── result_output/          # 识别结果输出图片
│   ├── tests/                  # pytest 自动化测试（8 个文件，185 项）
│   ├── pytest.ini              # pytest 配置（testpaths / api 标记）
│   ├── main.py                 # CLI 控制台调试入口
│   ├── api_server.py           # FastAPI Web 服务
│   ├── face_detection.py       # Haar 人脸检测模块
│   ├── face_recognition.py     # FaceNet 特征提取与比对模块
│   ├── db_utils.py             # SQLite 特征数据库模块
│   └── face_features.db        # 特征库（首次运行自动生成）
├── frontend/                   # Vue3 前端项目（阶段 4 待初始化，当前仅 .gitkeep）
├── docs/                       # 项目文档（总纲 + 阶段 1~3 共 11 份）
├── temp/                       # 本地临时指导文档（已忽略）
├── requirements.txt            # Python 依赖清单
├── .gitignore                  # 版本控制忽略规则
└── README.md                   # 项目入口说明
```

---

## 🚀 环境部署

### 1. 后端部署

```bash
# 创建并激活 conda 环境（与项目实测环境一致；也可使用 venv）
conda create -n cv_demo python=3.11 -y
conda activate cv_demo

# 安装依赖
pip install -r requirements.txt

# CUDA 版 torch 的安装/离线权重放置见 temp/TORCH_INSTALL_GUIDE.md
```

### 2. 启动后端

```bash
# 方式 A：CLI 控制台调试
cd backend
python main.py
```

CLI 菜单功能如下：

1. 通过图片录入人脸
2. 通过摄像头录入人脸
3. 查看特征库
4. 从图片识别人脸
5. 摄像头实时识别人脸
6. 删除已录入人脸
7. 退出系统（实际菜单项为 `0`）

```bash
# 方式 B：启动 FastAPI Web 后端服务
cd backend
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 3. 前端部署（阶段 4 初始化后）

```bash
cd frontend
npm install
npm run dev
```

### 4. 运行自动化测试

```bash
cd backend
python -m pytest -v
```

覆盖范围：人脸检测（`test_face_detection.py`）、SQLite 特征库（`test_db_utils.py`）、FaceNet 特征提取与比对（`test_face_recognition.py`）、CLI 业务流程（`test_main.py`）、采集 / 识别 / 特征库 / 集成接口（`test_api_register.py` / `test_api_recognize.py` / `test_api_library.py` / `test_api_server.py`）。当前共 **185 项**（阶段 2 单测 65 项 + 阶段 3 接口 120 项）。

- 测试样例位于 `backend/sample_faces/_test_samples/`（`one_face.jpg` / `no_face.jpg`，缺失时相关用例自动 `skip`）；
- 特征库测试使用 `tmp_path` 临时库，**不会污染** `face_features.db`；
- 摄像头相关的两个功能依赖物理设备，不纳入自动化测试，需人工验证。

---

## ⚙️ 核心参数说明

| 参数 | 说明 |
| --- | --- |
| **特征维度** | FaceNet 输出 512 维归一化人脸特征向量 |
| **比对算法** | 余弦相似度 |
| **识别阈值** | 默认 `0.6`，相似度 ≥ 阈值判定为同一人 |
| **人脸检测器** | Haar 级联 `haarcascade_frontalface_alt2.xml` |
| **数据库** | SQLite，特征以 `Blob` 二进制形式存储 |
| **预训练权重** | FaceNet 首次运行自动下载 `casia-webface` |

### 调参建议

- **阈值过低（< 0.5）**：不同人的误识别率上升
- **阈值过高（> 0.7）**：同一人多角度/光照下可能漏判
- 建议针对实际场景（样本质量、光照、摄像头位置）在 `0.5 ~ 0.7` 区间标定

---

## 📌 已知注意事项

1. **OpenCV 版本要求**：`opencv-python` 固定 **4.8.x**（实测 4.8.1.78）；5.x 已移除 `CascadeClassifier`，4.14.x 又要求 NumPy 2，均无法与本项目共存。
2. **NumPy 版本要求**：必须 **1.x**（实测 1.26.4）；升级到 2.x 会导致 `cv2` / `torch` 出现 NumPy ABI 报错。
3. **预训练权重下载**：FaceNet 首次运行会自动下载 `casia-webface` 预训练权重，需保持网络畅通。
4. **中文字体**：Windows 环境需 `simhei.ttf`（黑体），缺失会导致控制台绘图中文乱码。
5. **硬件加速**：GPU 环境自动启用 CUDA 加速；无 GPU 自动降级为 CPU 运行。

---

## 📄 快速使用流程

```
1. 部署后端     → 创建虚拟环境，安装依赖，启动 uvicorn
2. 录入人脸     → 通过 CLI/Web 上传图片或摄像头采集，生成特征向量入库
3. 识别人脸     → 上传待识别图片，系统定位人脸、比对特征库、输出身份与相似度
4. 查看结果     → 结果图片保存至 result_output/，Web 端可实时预览
```

---

## 📚 附：接口速览（FastAPI）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查（设备、检测器、特征维度、库记录数） |
| `POST` | `/api/face/register/image` | 上传图片 + 姓名，检测后取最大脸入库 |
| `POST` | `/api/face/register/camera-frame` | base64 / dataURL 摄像头帧入库 |
| `GET` | `/api/face/library` | 特征库元信息列表（ID、姓名、时间、维度） |
| `DELETE` | `/api/face/library?name=xxx` | 按姓名删除全部同名记录 |
| `POST` | `/api/recognize/image` | 上传图片识别，可保存标注图 |
| `POST` | `/api/recognize/frame` | base64 / dataURL 摄像头帧识别 |

> 统一响应信封为 `{code, message, data}`；完整参数、错误码（1001~1008、422）与示例见 `docs/03-01` ~ `docs/03-04`。Swagger UI：`http://127.0.0.1:8000/docs`。

---

## 🤝 贡献与说明

- 本项目用于教学与算法验证，人脸数据请遵循相关隐私合规要求采集使用。
- 如发现问题，欢迎提交 Issue 或 Pull Request。