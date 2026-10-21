import { ApiError } from '../api/http'

/**
 * 错误码 → 兜底提示语（docs/04-01 4.8）
 * 业务错误的 message 由后端给出且更具体，优先使用；这里只在前端需要补充解释时兜底
 */
const HINTS = {
  1001: '请检查姓名、图片格式与文件大小（≤ 5 MB）',
  1002: '未检测到人脸，请更换图片或正对摄像头',
  1003: '人脸特征提取失败，请重试',
  1004: '该姓名已存在，请先到「特征库管理」删除旧记录',
  1005: '记录不存在，列表已刷新',
  1007: '服务内部错误，请查看后端日志',
  1008: '推理服务繁忙，请稍后重试',
  422: '请求参数不符合接口要求',
}

/**
 * 转成界面可展示的文案
 * - 主动取消返回空串（页面判空后不提示）
 * - 业务错误优先后端 message
 * - 网络 / 超时使用前端文案
 */
export function toUserMessage(error) {
  if (!(error instanceof ApiError)) return '发生未知错误'
  if (error.isCanceled) return ''
  if (error.code > 0) return error.message || HINTS[error.code] || '请求失败'
  return error.message
}

/** 按错误码取兜底提示；未知码返回空串 */
export function hintOf(code) {
  return HINTS[code] || ''
}
