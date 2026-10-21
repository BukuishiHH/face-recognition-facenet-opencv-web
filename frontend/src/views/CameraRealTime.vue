<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { DEFAULT_THRESHOLD, recognizeByFrame } from '../api/face'
import { useCamera } from '../composables/useCamera'
import { drawFaceBoxes } from '../utils/draw'
import { toUserMessage } from '../utils/errors'

const BASE_INTERVAL = 400 // 基础采样间隔（ms）
const MAX_INTERVAL = 800 // 慢链路上限：再慢说明后端过载，继续 2.5 QPS 只会加剧排队
const SLOW_RTT = 800 // 单次往返超过此值判定为慢，自动降频

const camera = useCamera({ width: 640, height: 480 })
const { videoRef } = camera
const overlayRef = ref(null)

const running = ref(false) // 摄像头已开启且识别循环运行中
const starting = ref(false)
const intervalMs = ref(BASE_INTERVAL) // 当前采样间隔（400 / 800）
const faces = ref([]) // 最近一帧的全部人脸
const frameSize = ref({ width: 0, height: 0 }) // 最近一次送检帧的像素尺寸（不是 videoWidth）
const updatedAt = ref('')
const lastMessage = ref('')
const errorText = ref('')
const stats = ref({ frames: 0, dropped: 0, errors: 0, latencySum: 0, startedAt: 0 })

let timer = null
let controller = null
let resizeObserver = null

const stateText = computed(() => {
  if (errorText.value) return '错误'
  if (starting.value) return '开启中'
  if (running.value) return '● 运行中'
  return '○ 未开启'
})
const stateTone = computed(() => {
  if (errorText.value) return 'danger'
  if (running.value) return 'ok'
  return 'muted'
})
const frameMeta = computed(() => {
  if (!frameSize.value.width) return ''
  return `${frameSize.value.width}×${frameSize.value.height}`
})

/** 唯一排程出口：清旧 timer 再排新，避免多处 setTimeout 泄漏 */
function scheduleNext(delay = intervalMs.value) {
  clearTimeout(timer)
  timer = null
  if (!running.value) return
  timer = setTimeout(tick, delay)
}

/** 终止级错误：继续轮询只会刷屏，必须停循环并提示用户手动恢复 */
function failLoop(err) {
  const message = toUserMessage(err)
  running.value = false
  clearTimeout(timer)
  timer = null
  if (controller) {
    controller.abort()
    controller = null
  }
  camera.stop()
  faces.value = []
  lastMessage.value = ''
  errorText.value = message
  clearOverlay()
}

/** 单飞节拍：上一帧请求完成（含失败）后才排下一帧 */
async function tick() {
  if (!running.value) return
  const shot = camera.capture({ maxWidth: 640, quality: 0.8 })
  if (!shot) {
    scheduleNext(200) // 首帧未就绪，稍后再试
    return
  }
  frameSize.value = { width: shot.width, height: shot.height }

  controller = new AbortController()
  const startedAt = performance.now()
  try {
    const res = await recognizeByFrame(shot.dataUrl, {
      threshold: DEFAULT_THRESHOLD,
      signal: controller.signal,
    })
    const rtt = performance.now() - startedAt
    if (!running.value) return // 请求返回时已被停止
    stats.value.frames += 1
    stats.value.latencySum += rtt
    // 慢链路自动降频，恢复后再回到基准间隔
    intervalMs.value = rtt > SLOW_RTT ? MAX_INTERVAL : BASE_INTERVAL
    faces.value = res.data.faces
    lastMessage.value = res.data.library_empty
      ? '特征库为空，请先到「人脸采集」录入'
      : res.message
    updatedAt.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    drawOverlay()
  } catch (err) {
    if (err.isCanceled) return
    if (err.code === 1002) {
      // 本帧无脸：清框，不当错误
      faces.value = []
      lastMessage.value = '当前画面未检测到人脸'
      drawOverlay()
    } else if (err.code === 1008 || err.code === -2) {
      // 1008 推理繁忙 / -2 超时：主动丢帧，下一拍继续，不重试当前帧
      stats.value.dropped += 1
      lastMessage.value = err.code === 1008 ? '推理繁忙，已丢弃本帧' : '本帧超时，已丢弃'
    } else if (err.code === -1 || err.code === 1001 || err.code === 1007) {
      failLoop(err)
    } else {
      stats.value.errors += 1
      lastMessage.value = toUserMessage(err)
    }
  } finally {
    controller = null
    if (running.value) scheduleNext()
  }
}

/** 覆盖层尺寸与视频显示区域严格一致；坐标按送检帧尺寸缩放 */
function drawOverlay() {
  const canvas = overlayRef.value
  const video = videoRef.value
  if (!canvas || !video || !video.videoWidth) return
  const rect = video.getBoundingClientRect()
  if (!rect.width || !rect.height) return

  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(rect.width * dpr)
  canvas.height = Math.round(rect.height * dpr)
  canvas.style.width = `${rect.width}px`
  canvas.style.height = `${rect.height}px`

  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0) // 高清屏 1:1 绘制
  ctx.clearRect(0, 0, rect.width, rect.height)
  // 框坐标属于「送检帧」坐标系（capture 可能把 1280 缩到 640），不是 videoWidth
  const srcWidth = frameSize.value.width || video.videoWidth
  drawFaceBoxes(ctx, faces.value, rect.width / srcWidth)
}

function clearOverlay() {
  const canvas = overlayRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

async function start() {
  if (starting.value || running.value) return
  starting.value = true
  errorText.value = ''
  lastMessage.value = '正在请求摄像头权限…'
  const ok = await camera.start()
  if (!ok) {
    starting.value = false
    errorText.value = camera.errorText.value || '摄像头启动失败'
    lastMessage.value = ''
    return
  }
  faces.value = []
  frameSize.value = { width: 0, height: 0 }
  updatedAt.value = ''
  intervalMs.value = BASE_INTERVAL
  stats.value = { frames: 0, dropped: 0, errors: 0, latencySum: 0, startedAt: Date.now() }
  running.value = true
  starting.value = false
  lastMessage.value = '已开启，正在识别…'
  timer = setTimeout(tick, 300) // 等首帧稳定后再截
}

function stop() {
  running.value = false
  starting.value = false
  clearTimeout(timer)
  timer = null
  if (controller) {
    controller.abort()
    controller = null
  }
  camera.stop()
  faces.value = []
  lastMessage.value = ''
  updatedAt.value = ''
  clearOverlay()
}

onMounted(() => {
  // 窗口缩放 / 侧栏折叠时重绘，避免框与画面错位
  if (videoRef.value && typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => drawOverlay())
    resizeObserver.observe(videoRef.value)
  }
})
onBeforeUnmount(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  stop()
})
// HMR 兜底：热更新不会触发完整卸载，需手动释放 MediaStream
if (import.meta.hot) {
  import.meta.hot.dispose(() => stop())
}
</script>

<template>
  <section class="page realtime">
    <h1 class="page__title">实时识别</h1>
    <p class="page__subtitle">
      摄像头逐帧识别，绿框命中 / 红框未命中；繁忙或超时时静默丢帧，不堆积请求
    </p>

    <div class="layout-two-col">
      <!-- 左列：视频 + 覆盖层 -->
      <div class="card realtime-preview">
        <div id="realtime-stage" class="realtime-stage" :class="{ 'realtime-stage--on': running }">
          <video
            ref="videoRef"
            id="realtime-video"
            class="realtime-stage__video"
            autoplay
            playsinline
            muted
          />
          <canvas ref="overlayRef" id="realtime-overlay" class="realtime-stage__overlay" />
          <div v-if="!running" class="realtime-stage__placeholder">
            点击「开启摄像头」开始实时识别
            <span>摄像头仅在 localhost / 127.0.0.1 等安全上下文可用</span>
          </div>
        </div>
        <p id="realtime-state" class="realtime-state" :class="`realtime-state--${stateTone}`">
          <span>{{ stateText }}</span>
          <span v-if="frameMeta" class="realtime-state__meta">· 送检帧 {{ frameMeta }}</span>
          <span v-if="updatedAt" class="realtime-state__meta">· 更新于 {{ updatedAt }}</span>
        </p>
      </div>

      <!-- 右列：控制面板 -->
      <div class="card realtime-panel">
        <h2 class="card__title">识别控制</h2>

        <div class="realtime-actions">
          <button
            v-if="!running"
            id="realtime-start"
            class="btn"
            :disabled="starting"
            @click="start"
          >
            {{ starting ? '正在开启…' : '开启摄像头' }}
          </button>
          <button v-else id="realtime-stop" class="btn btn--danger" @click="stop">停止</button>
        </div>

        <div v-if="errorText" id="realtime-error" class="alert alert--danger realtime-alert">
          {{ errorText }}
        </div>
        <p v-else-if="lastMessage" id="realtime-message" class="muted realtime-alert">
          {{ lastMessage }}
        </p>

        <dl id="realtime-metrics" class="realtime-metrics">
          <div class="realtime-metrics__item">
            <dt>采样间隔</dt>
            <dd><span id="realtime-interval">{{ intervalMs }}</span> ms</dd>
          </div>
          <div class="realtime-metrics__item">
            <dt>已丢弃</dt>
            <dd id="realtime-dropped">{{ stats.dropped }}</dd>
          </div>
        </dl>

        <p class="realtime-hint">
          开启后页面每 <strong>{{ intervalMs }} ms</strong> 采样一帧发送到
          <code>POST /api/recognize/frame</code>；单帧往返超过 800 ms 会自动降频，
          后端忙（<code>1008</code>）或超时（<code>-2</code>）直接丢帧，不重试。
        </p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.realtime-preview {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.realtime-stage {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 320px;
  padding: 16px;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg);
  overflow: hidden;
}

.realtime-stage--on {
  padding: 0;
  border-style: solid;
  background: #0f172a;
}

/* 不用 object-fit / 固定宽高比：canvas 覆盖层要与视频显示区域严格重合 */
.realtime-stage__video {
  display: block;
  width: 100%;
  height: auto;
  max-height: 62vh;
}

/* 覆盖层与视频显示区域完全重合：只接收绘制，不拦截鼠标事件 */
.realtime-stage__overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.realtime-stage__placeholder {
  color: var(--color-muted);
  font-size: 14px;
  text-align: center;
}

.realtime-stage__placeholder span {
  display: block;
  margin-top: 6px;
  font-size: 12px;
}

.realtime-state {
  margin: 0;
  font-size: 13px;
}

.realtime-state__meta {
  margin-left: 6px;
  color: var(--color-muted);
  font-size: 12px;
}

.realtime-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.realtime-metrics__item {
  padding: 8px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg);
}

.realtime-metrics dt {
  color: var(--color-muted);
  font-size: 12px;
}

.realtime-metrics dd {
  margin: 2px 0 0;
  font-variant-numeric: tabular-nums;
  font-size: 15px;
  font-weight: 600;
}

.realtime-state--ok {
  color: var(--color-success);
}

.realtime-state--muted {
  color: var(--color-muted);
}

.realtime-state--danger {
  color: var(--color-danger);
}

.realtime-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.realtime-panel .card__title {
  margin: 0;
}

.realtime-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.realtime-alert {
  margin: 0;
}

.realtime-hint {
  margin: 0;
  color: var(--color-muted);
  font-size: 12px;
  line-height: 1.6;
}
</style>
