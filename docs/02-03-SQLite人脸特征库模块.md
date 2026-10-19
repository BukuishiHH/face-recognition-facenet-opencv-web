# Step 2-3 · SQLite 人脸特征库模块

> **所属阶段**：阶段 2 · 底层核心算法模块开发、验证与文档沉淀
> **对应总纲**：`docs/development_steps.md` → Step 2-3（SQLite 人脸特征库模块开发与沉淀）
> **源码文件**：`backend/db_utils.py`
> **测试文件**：`backend/tests/test_db_utils.py`（22 项）
> **文档状态**：✅ 已完成（代码 + 文档 + 自动化测试；2026-10-19 按阶段 3 改造后复核）

---

## 一、功能背景

识别模块产出的是 512 维浮点向量，「这个向量属于谁」必须被持久化，否则每次启动都要重新录入。选型对比：

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **SQLite** | Python 内置 `sqlite3`，**零安装零服务**；单文件；原生支持 `BLOB` 存二进制 | 无并发写入优化（本场景单进程足够） | ✅ 选用 |
| CSV/JSON | 直观 | 浮点文本精度损失、无索引、无事务 | ✗ |
| `npy` 文件 | 读写极快 | 无法按姓名删除单条、无时间戳元数据 | ✗ |
| MySQL/PostgreSQL | 功能强 | 需部署服务，与「单机算法演示」定位不符 | ✗ |

**关键设计**：特征向量按 `float32` 序列化为 `BLOB`（512 × 4 = **2048 字节/条**），既无精度损失，也远小于文本形式。

---

## 二、开发目标

| 目标 | 说明 |
|------|------|
| 建表幂等 | 反复初始化不得报错、不得丢数据（`CREATE TABLE IF NOT EXISTS`） |
| 特征存取 | `(1,512)` `float32` 特征 ↔ `BLOB` 无损双向转换 |
| 全量读取 | 一次取回全部特征，拼成 `(N,512)` 矩阵供批量余弦比对 |
| 按姓名删除 | 支持删除某人全部记录，返回受影响行数 |
| 重复防护 | 同名记录已存在时跳过录入，返回 `None` 而非抛异常 |
| 同名预检 | `exists_name()` 只读精确查询，供接口层区分 `1004`（同名）与 `1003`（入库失败） |
| 元信息列表 | `list_records()` 只查 `id/name/timestamp/feature_dim`，不读取 2048 B 的 BLOB |
| 并发安全 | `check_same_thread=False` + 模块级 `RLock`，允许 FastAPI 线程池串行复用同一连接 |
| 资源可释放 | 提供幂等 `close()`，重复调用不报错（Windows 下文件被占用会导致删除失败） |

---

## 三、功能职责

`backend/db_utils.py` 暴露类 `FaceFeatureDB`，职责边界：

1. **表结构管理**：首次连接即建表，幂等；
2. **特征持久化**：校验形状 → `float32` 序列化 → 查重 → 插入；
3. **特征检索**：全量读取 → 反序列化 → 拼矩阵 + 元信息列表；另提供轻量 `list_records()`；
4. **记录删除**：按 `face_id`（姓名）删除；
5. **并发与生命周期**：所有数据库操作由模块级 `RLock` 串行化，`close()` 幂等。

**不做的事**：不做人脸检测与特征提取（→ 上游模块）；不做界面交互（→ `main.py` / `api_server.py`）。

---

## 四、功能逻辑

### 4.1 表结构

```sql
CREATE TABLE IF NOT EXISTS face_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    feature     BLOB    NOT NULL,   -- 512 维 float32 特征, 2048 字节
    timestamp   DATETIME NOT NULL,  -- 录入时间 'YYYY-MM-DD HH:MM:SS'
    face_id     TEXT,               -- 身份标签(姓名)
    feature_dim INTEGER NOT NULL    -- 特征维度, 固定 512
);
```

> `feature_dim` 单独存列，使 `get_all_features()` 反序列化时无需硬编码 512，为未来更换特征模型（如 128 维）留出扩展空间。

### 4.2 类结构

```python
_DB_LOCK = threading.RLock()          # 模块级可重入锁

class FaceFeatureDB:
    def __init__(self, db_path=None)      # 默认 backend/face_features.db; check_same_thread=False; 建表
    def _create_table(self)               # 幂等建表
    def save_feature(self, feature, face_id) -> int | None   # 保存, 返回记录ID; 同名/形状错误返回 None
    def exists_name(self, face_id) -> bool                   # 只读精确查询, 不改数据
    def list_records(self) -> list[dict]                     # id/name/timestamp/feature_dim, ORDER BY id ASC
    def get_all_features(self) -> (np.ndarray | None, list | None)
    def delete_by_name(self, face_id) -> int                 # 按姓名删除, 返回行数
    def close(self)                       # 关闭连接; 幂等
```

### 4.3 写入流程

```text
save_feature(feature, face_id)
   │
   ├─ 类型校验: isinstance(feature, np.ndarray)                    → 否 → ValueError
   ├─ 形状校验: ndim==2 and shape==(1, 512)                        → 否 → ValueError
   │
   ├─ 序列化  : feature.astype(np.float32).tobytes()   → 2048 字节
   ├─ 时间戳  : time.strftime('%Y-%m-%d %H:%M:%S')
   │
   ├─ 查重    : SELECT id WHERE face_id = ?            → 命中 → 打印 Warning, 返回 None
   ├─ 插入    : INSERT INTO face_features (...)
   ├─ 提交    : conn.commit()                          → 失败分支 conn.rollback()
   └─ 返回    : cursor.lastrowid
```

### 4.4 读取流程

```text
get_all_features()
   │
   ├─ SELECT id, face_id, feature, feature_dim FROM face_features
   ├─ 逐行反序列化: np.frombuffer(blob, dtype=np.float32).reshape(1, dim)
   │      └─ 单行失败 → 打印 Error 并 continue (坏数据不拖垮整库)
   ├─ 拼接: np.vstack(features)  → (N, 512)
   └─ 空库保护: if features → 返回 (matrix, info_list)
                else        → 返回 (None, None)
```

> 返回值 `info_list` 与矩阵**行序严格对应**（`[{'id':.., 'name':..}, ...]`），下游通过 `argmax` 的下标即可反查身份。

### 4.5 输入与输出

| 方法 | 输入 | 输出 | 异常/边界 |
|------|------|------|-----------|
| `__init__(db_path=None)` | 路径（可选） | 实例 | 路径不可写时 `sqlite3.OperationalError`；连接参数 `check_same_thread=False` |
| `save_feature(feature, face_id)` | `(1,512)` `np.ndarray` + 姓名 | 记录 ID（int） | 形状不符 → 打印 Error 并返回 `None`；同名已存在 → 打印 Warning 并返回 `None` |
| `exists_name(face_id)` | 姓名 | `True` / `False` | 精确匹配（`张三` 不会命中 `张三丰`）；只读 |
| `list_records()` | — | `[{id,name,timestamp,feature_dim}, ...]`，按 `id` 升序 | 空库返回 `[]`；`face_id IS NULL` 归一为 `''` |
| `get_all_features()` | — | `(N,512)` 矩阵 + 信息列表；空库 → `(None, None)` | 单条坏数据被跳过，不中断 |
| `delete_by_name(face_id)` | 姓名 | 删除行数（int） | 姓名不存在返回 `0` |
| `close()` | — | `None` | 幂等：`_closed` 标记，重复调用直接返回 |

### 4.6 存储开销

| 项 | 计算 | 结果 |
|----|------|------|
| 单条特征 | 512 × 4 字节 | **2048 B** |
| 1000 人 | 2048 B × 1000 | ≈ 2 MB |
| 对比 JSON 文本 | 512 × ~20 字符 | ≈ 10 KB/条（**5 倍以上**） |

### 4.7 并发与生命周期

| 机制 | 实现 | 目的 |
|------|------|------|
| 跨线程连接 | `sqlite3.connect(..., check_same_thread=False)` | FastAPI 同步路由在线程池执行，允许 worker 复用主线程创建的连接 |
| 串行化 | 模块级 `_DB_LOCK = threading.RLock()` 包住建表、写入、查询、删除、关闭 | 单连接不支持并发写；读也串行以保证行序稳定；`RLock` 允许同线程嵌套 |
| 幂等关闭 | `_closed` 布尔标记 + 锁内判断 | lifespan 与测试夹具可能重复 `close()`，不抛 `ProgrammingError` |
| 只读预检 | `exists_name()` 与 `list_records()` 均不写库 | 接口层的 `1004` 预检、列表查询不改变数据 |

---

## 五、里程碑计划

| 里程碑 | 交付物 | 判定标准 | 状态 |
|--------|--------|----------|------|
| M2-3.1 | 建表实现 | 重复初始化无异常，表结构一致 | ✅ |
| M2-3.2 | 特征保存实现 | 返回有效 `id`，`face_features.db` 生成 | ✅ |
| M2-3.3 | 特征读取实现 | `(N,512)` 矩阵与原值一致（`allclose`） | ✅ |
| M2-3.4 | 按姓名删除实现 | `delete_by_name` 返回受影响行数且记录消失 | ✅ |
| M2-3.5 | 重复录入防护 | 同名二次录入被跳过 | ✅ |
| M2-3.6 | 模块文档沉淀 | 本文档 | ✅ |
| M2-3.7 | 同名只读预检 | `exists_name()` 精确匹配、只读，供接口层使用 | ✅ |
| M2-3.8 | 元信息列表 | `list_records()` 按 `id ASC` 返回四项元信息，不读取 BLOB | ✅ |
| M2-3.9 | 线程安全与幂等关闭 | 20 线程并发读写无 `ProgrammingError`；`close()` 可重复调用 | ✅ |
| M2-3.10 | 自动化测试 | `backend/tests/test_db_utils.py` 22 项全部通过 | ✅ |

---

## 六、验收清单

- [x] **建表幂等**：多次 `FaceFeatureDB()` 不报错、不重置数据
- [x] **Blob 特征存取一致**：`save` → `get` 后 `np.allclose(原值, 还原值) == True`
- [x] **按姓名删除生效**：`delete_by_name('张三')` 后该姓名记录数为 0
- [x] 形状非 `(1,512)` 的特征被拒绝，不污染库
- [x] 空库调用 `get_all_features()` 返回 `(None, None)` 而非崩溃
- [x] `face_features.db` 在首次运行后于 `backend/` 目录自动生成
- [x] 文件句柄可通过 `close()` 释放，且重复调用不报错
- [x] `exists_name()` 精确匹配、只读，删除后返回 `False`
- [x] `list_records()` 空库返回 `[]`，按 `id` 升序，`name` 不为 `None`
- [x] 20 线程并发读写无 `ProgrammingError`，记录数与姓名一致
- [x] 自动化测试 22 项通过（`test_db_utils.py`）

---

## 七、异常点与坑点

| # | 问题 | 现象 | 定位与处置 |
|---|------|------|------------|
| 1 | **空库变量未定义（已修复）** | 原代码 `if feature:`（应为 `if features:`）——`feature` 是循环内局部变量，空库时从未赋值 → `NameError`；即使侥幸通过，`np.vstack([])` 也会抛 `ValueError` | 修正为 `if features:`；**空库必须显式返回 `(None, None)`**，因为下游 `main.py`/`api_server.py` 需要「库为空」这一状态 |
| 2 | **`float32` 与 `float64` 混用** | 上游若传入 `float64`，`tobytes()` 得到 4096 字节，`frombuffer(dtype=float32)` 还原后长度翻倍、数值错乱 | 写入统一 `astype(np.float32)`；读取固定 `dtype=np.float32`，形成端到端契约 |
| 3 | **Windows 下数据库文件被占用** | 不 `close()` 就删除 `face_features.db` → `PermissionError` | 所有使用方（`main.py`/`api_server.py`/测试）负责 `close()`；`close()` 已做幂等；测试用 `tmp_path` 独立库避免互相干扰 |
| 4 | **模块导入即建库的副作用** | 文件末尾 `face_db = FaceFeatureDB()` 会在 **import 时** 直接创建 `backend/face_features.db` | 属有意设计（CLI/接口层直接复用单例）；副作用已记录：单元测试应显式构造 `FaceFeatureDB(tmp_path/...)` 而非依赖全局单例 |
| 5 | 同名多人（重名）无法区分 | `face_id` 仅存姓名，库内以「同名唯一」为前提 | 当前业务模型为「同名即同一人」，重名场景需改用 `姓名+工号` 作为 `face_id`（阶段 3 可扩展） |
| 6 | 同名多条记录的历史遗留 | 早期版本可能已写入同名多行，导致 `get_all_features()` 返回重复身份 | `save_feature` 已加查重；如需修复历史数据，可 `delete_by_name` 后重新录入 |
| 7 | 用 `print` 而非 `logging` 输出 | 日志无法分级、无法重定向；在 FastAPI 下会混进 stdout | `db_utils` 仍保留 `print` 以兼容 CLI；接口层另用 `api_server` logger 记录业务结果与耗时，不依赖这些 `print` |
| 8 | 无索引下的按姓名删除 | 数据量大时 `DELETE WHERE face_id=?` 全表扫描 | 数据规模千级时无感知；如需优化可对 `face_id` 建索引 |
| 9 | 时间戳为本地字符串而非 UTC | 跨时区协作时时间含义不一致 | 单机演示场景可接受；如接入统一时区，建议改存 ISO8601 UTC |

---

## 附：验证方法（阶段 2-3 冒烟）

```powershell
cd backend
& 'D:\anaconda\envs\cv_demo\python.exe' -c @"
import numpy as np
from db_utils import FaceFeatureDB

db = FaceFeatureDB('_smoke.db')                     # 独立测试库
f = np.random.rand(1, 512).astype(np.float32)
rid = db.save_feature(f, '测试员')
mat, info = db.get_all_features()
print('id:', rid, 'shape:', mat.shape, 'same:', np.allclose(mat[0], f[0]))
print('delete:', db.delete_by_name('测试员'))
print('after  :', db.get_all_features())            # 期望 (None, None)
db.close()
"@
```