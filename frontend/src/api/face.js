import http from './http'

// ------------------------------- 前端常量（与后端契约一致） -------------------------------
export const DEFAULT_THRESHOLD = 0.6
export const THRESHOLD_MIN = 0.3
export const THRESHOLD_MAX = 0.95
export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024
export const MAX_NAME_LEN = 32

/** 健康检查：服务状态 / 设备 / 检测器 / 特征维度 / 特征库人数 */
export function getHealth({ signal } = {}) {
  return http.get('/health', { timeout: 5000, signal })
}

/** 图片入库：multipart 字段固定为 file / name（docs/03-02） */
export function registerByImage(file, name, { onUploadProgress, signal } = {}) {
  const form = new FormData()
  form.append('file', file, file.name || 'upload.jpg')
  form.append('name', name.trim())
  return http.post('/face/register/image', form, { timeout: 30000, onUploadProgress, signal })
}

/** 摄像头帧入库：JSON name / image_base64（dataURL 可直接传） */
export function registerByFrame(name, imageBase64, { signal } = {}) {
  return http.post(
    '/face/register/camera-frame',
    { name: name.trim(), image_base64: imageBase64 },
    { timeout: 30000, signal },
  )
}

/** 图片识别：save_result 必须显式传字符串，避免布尔被序列化成空值 */
export function recognizeByImage(
  file,
  { threshold = DEFAULT_THRESHOLD, saveResult = false, signal } = {},
) {
  const form = new FormData()
  form.append('file', file, file.name || 'upload.jpg')
  form.append('threshold', String(threshold))
  form.append('save_result', saveResult ? 'true' : 'false')
  return http.post('/recognize/image', form, { timeout: 60000, signal })
}

/** 实时帧识别：超时即丢帧，不排队（docs/04-04） */
export function recognizeByFrame(imageBase64, { threshold = DEFAULT_THRESHOLD, signal } = {}) {
  return http.post(
    '/recognize/frame',
    { image_base64: imageBase64, threshold },
    { timeout: 5000, signal },
  )
}

/** 特征库列表：每次进入页面重新拉取，不做前端缓存 */
export function getLibrary({ signal } = {}) {
  return http.get('/face/library', { timeout: 10000, signal })
}

/** 删除特征库记录：params 传原始姓名，由 axios 负责 URL 编码（坑点 15） */
export function deleteLibraryItem(name, { signal } = {}) {
  return http.delete('/face/library', { params: { name }, timeout: 10000, signal })
}
