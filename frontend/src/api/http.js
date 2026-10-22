import axios from 'axios'

// baseURL 来自环境变量；未配置时回落到本地后端（见 docs/04-01 4.3 / 坑点 16）
const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api'

export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, '')                 // 去掉尾部斜杠，避免 //health
export const STATIC_ORIGIN = (
  import.meta.env.VITE_STATIC_ORIGIN || API_BASE_URL.replace(/\/api$/, '')
).replace(/\/+$/, '')

/**
 * 前端统一错误对象
 * code:  0 成功；>0 后端业务码；<0 前端侧（-1 网络 / -2 超时 / -3 取消）
 * status: HTTP 状态码，网络类错误为 0
 */
export class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }

  /** 推理繁忙：实时页静默丢帧，图片页提示稍后重试 */
  get isBusy() {
    return this.code === 1008
  }

  /** 主动取消：页面直接 return，不写提示 */
  get isCanceled() {
    return this.code === -3
  }

  /** 网络不可达 / 超时：首页显示重试入口 */
  get isNetwork() {
    return this.code === -1 || this.code === -2
  }
}

const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: { Accept: 'application/json' },
})

/**
 * 成功分支：把 {code, message, data} 信封解包后原样返回
 * 页面拿到的是信封本身（不是 AxiosResponse），业务数据取 res.data
 */
function unwrap(response) {
  const body = response.data
  if (!body || typeof body.code !== 'number') {
    throw new ApiError(-1, '响应格式异常：缺少 code 字段', response.status)
  }
  if (body.code !== 0) {
    throw new ApiError(body.code, body.message || '请求失败', response.status)
  }
  return body
}

http.interceptors.response.use(
  unwrap,
  (error) => {
    // 422 / 429 / 500 也由后端包装成信封，统一按业务码处理
    const body = error.response && error.response.data
    if (body && typeof body.code === 'number') {
      return Promise.reject(
        new ApiError(body.code, body.message || '请求失败', error.response.status),
      )
    }
    if (error.code === 'ERR_CANCELED') {
      return Promise.reject(new ApiError(-3, '请求已取消', 0))
    }
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new ApiError(-2, '请求超时：后端推理耗时较长或服务未响应', 0))
    }
    // 非信封响应（如未匹配路由的 404 Not Found、网关的 502）：按 HTTP 状态兜底，
    // 避免把「接口路径写错」误报成「无法连接后端服务」（docs/04-06 §4.4 #11）
    if (error.response) {
      const { status } = error.response
      return Promise.reject(
        new ApiError(status, `接口响应异常：HTTP ${status}，请检查请求路径与参数`, status),
      )
    }
    return Promise.reject(
      new ApiError(-1, '无法连接后端服务，请确认 uvicorn 已在 8000 端口启动', 0),
    )
  },
)

export default http

/** `/static/xxx.jpg` → 完整可访问 URL；已是绝对地址则原样返回 */
export function resolveStaticUrl(path) {
  if (!path) return ''
  if (/^https?:\/\//i.test(path)) return path
  return `${STATIC_ORIGIN}${path.startsWith('/') ? '' : '/'}${path}`
}
