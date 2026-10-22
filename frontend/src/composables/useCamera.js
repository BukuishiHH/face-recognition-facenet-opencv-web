import { onBeforeUnmount, ref, shallowRef } from 'vue'
import { scaleSize } from '../utils/image'

/** 浏览器异常名 → 中文处置建议（docs/04-02 4.4） */
const ERROR_TEXT = {
  NotAllowedError: '摄像头权限被拒绝，请点击地址栏的相机图标允许访问后重试',
  PermissionDeniedError: '摄像头权限被拒绝，请点击地址栏的相机图标允许访问后重试',
  NotFoundError: '未检测到可用摄像头设备',
  DevicesNotFoundError: '未检测到可用摄像头设备',
  NotReadableError: '摄像头被其他程序占用，请关闭会议 / 直播软件后重试',
  TrackStartError: '摄像头被其他程序占用，请关闭占用摄像头的应用后重试',
  OverconstrainedError: '当前摄像头不支持请求的分辨率，请切换设备后重试',
  SecurityError: '当前页面不是安全上下文，请通过 127.0.0.1 / localhost 或 HTTPS 访问',
}

/**
 * 摄像头生命周期：申请权限 → 绑定 <video> → 截帧 → 释放
 * 4-2（拍照入库）与 4-4（实时识别）共用，差异只在截图频率与后续请求
 */
export function useCamera({ width = 640, height = 480, facingMode = 'user' } = {}) {
  const videoRef = ref(null)
  const stream = shallowRef(null) // MediaStream 不需要深层响应
  const status = ref('idle') // idle | starting | ready | error
  const errorText = ref('')
  const devices = ref([])
  const activeDeviceId = ref('')

  async function start(deviceId = '') {
    stop()
    status.value = 'starting'
    errorText.value = ''

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      status.value = 'error'
      // 局域网 IP 是非 secure context，Chrome 会直接隐藏 mediaDevices，容易误判为「浏览器不支持」
      errorText.value =
        window.isSecureContext === false
          ? '当前地址不是安全上下文，请用 localhost 或 HTTPS 访问'
          : '当前浏览器不支持摄像头采集，请使用 Chrome / Edge 最新版'
      return false
    }

    const constraints = {
      audio: false,
      video: {
        width: { ideal: width },
        height: { ideal: height },
        ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode }),
      },
    }

    try {
      stream.value = await navigator.mediaDevices.getUserMedia(constraints)
    } catch (error) {
      status.value = 'error'
      errorText.value = ERROR_TEXT[error.name] || `摄像头启动失败：${error.message}`
      return false
    }

    if (videoRef.value) {
      videoRef.value.srcObject = stream.value
      try {
        await videoRef.value.play()
      } catch {
        /* 自动播放被策略拦截时，由用户点击 video 手动播放 */
      }
    }

    const track = stream.value.getVideoTracks()[0]
    activeDeviceId.value = deviceId || (track && track.getSettings().deviceId) || ''
    await refreshDevices()
    status.value = 'ready'
    return true
  }

  function stop() {
    if (stream.value) {
      // 仅置空 srcObject 不会释放设备，必须逐轨 stop()
      stream.value.getTracks().forEach((track) => track.stop())
      stream.value = null
    }
    if (videoRef.value) videoRef.value.srcObject = null
    if (status.value !== 'error') status.value = 'idle'
  }

  /** 首次授权前 label 为空，必须在 getUserMedia 成功后再枚举 */
  async function refreshDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return
    try {
      const list = await navigator.mediaDevices.enumerateDevices()
      devices.value = list.filter((item) => item.kind === 'videoinput')
    } catch {
      devices.value = []
    }
  }

  /**
   * 从当前帧截图；长边压到 maxWidth，返回 { dataUrl, width, height }
   * 必须用 videoWidth / videoHeight（解码分辨率），CSS 尺寸只影响显示
   */
  function capture({ maxWidth = 640, quality = 0.8 } = {}) {
    const video = videoRef.value
    if (!video || !video.videoWidth || !video.videoHeight) return null
    const size = scaleSize(video.videoWidth, video.videoHeight, maxWidth)
    const canvas = document.createElement('canvas')
    canvas.width = size.width
    canvas.height = size.height
    canvas.getContext('2d').drawImage(video, 0, 0, size.width, size.height)
    return { dataUrl: canvas.toDataURL('image/jpeg', quality), width: size.width, height: size.height }
  }

  onBeforeUnmount(stop)

  return {
    videoRef,
    stream,
    status,
    errorText,
    devices,
    activeDeviceId,
    start,
    stop,
    refreshDevices,
    capture,
  }
}
