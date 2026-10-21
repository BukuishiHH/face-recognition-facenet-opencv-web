<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { useCamera } from '../composables/useCamera'

const camera = useCamera({ width: 640, height: 480 })
const { videoRef } = camera

const running = ref(false) // 摄像头是否已开启（M4-4.3 起同时表示识别循环运行中）
const starting = ref(false)
const lastMessage = ref('')
const errorText = ref('')

const stateText = computed(() => {
  if (errorText.value) return '错误'
  if (starting.value) return '开启中'
  if (running.value) return '● 已开启'
  return '○ 未开启'
})
const stateTone = computed(() => {
  if (errorText.value) return 'danger'
  if (running.value) return 'ok'
  return 'muted'
})

async function start() {
  if (starting.value || running.value) return
  starting.value = true
  errorText.value = ''
  lastMessage.value = '正在请求摄像头权限…'
  const ok = await camera.start()
  starting.value = false
  if (!ok) {
    errorText.value = camera.errorText.value || '摄像头启动失败'
    lastMessage.value = ''
    return
  }
  running.value = true
  lastMessage.value = '摄像头已开启（M4-4.1：仅预览，识别循环待接入）'
}

function stop() {
  running.value = false
  starting.value = false
  camera.stop()
  lastMessage.value = ''
}

onBeforeUnmount(stop)
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
          <div v-if="!running" class="realtime-stage__placeholder">
            点击「开启摄像头」开始实时识别
            <span>摄像头仅在 localhost / 127.0.0.1 等安全上下文可用</span>
          </div>
        </div>
        <p id="realtime-state" class="realtime-state" :class="`realtime-state--${stateTone}`">
          {{ stateText }}
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

        <p class="realtime-hint">
          开启后页面每 <strong>400 ms</strong> 采样一帧发送到
          <code>POST /api/recognize/frame</code>；离开页面或热更新时自动释放摄像头。
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
