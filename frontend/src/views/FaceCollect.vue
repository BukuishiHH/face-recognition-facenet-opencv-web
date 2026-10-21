<script setup>
import { computed, ref } from 'vue'
import { MAX_NAME_LEN } from '../api/face'
import { useCamera } from '../composables/useCamera'

const mode = ref('upload') // upload | camera
const realName = ref('')
const composing = ref(false)
const status = ref('idle') // idle | previewing | submitting | receipt | error
const errorText = ref('')
const notice = ref('')
const captured = ref(null) // 摄像头截帧 { dataUrl, width, height }

const camera = useCamera({ width: 640, height: 480 })
const {
  videoRef,
  status: cameraStatus,
  errorText: cameraError,
  devices,
  activeDeviceId,
  start: startStream,
  stop: stopStream,
} = camera

const nameLength = computed(() => [...realName.value.trim()].length)
const nameTooLong = computed(() => nameLength.value > MAX_NAME_LEN)
const submitting = computed(() => status.value === 'submitting')
const cameraReady = computed(() => cameraStatus.value === 'ready')
const cameraHint = computed(() => {
  if (cameraStatus.value === 'starting') return '正在开启摄像头…'
  if (cameraStatus.value === 'error') return '摄像头启动失败，请查看下方提示'
  return '摄像头未开启'
})

/** 失焦 / IME 结束后去掉首尾空格；拼音输入过程中不打断候选词 */
function normalizeName() {
  if (composing.value) return
  realName.value = realName.value.trim()
}

async function startCamera(deviceId = activeDeviceId.value) {
  notice.value = ''
  errorText.value = ''
  const ok = await startStream(deviceId)
  if (!ok) {
    errorText.value = cameraError.value || '摄像头启动失败'
    status.value = 'error'
    return
  }
  status.value = captured.value ? 'previewing' : 'idle'
}

function stopCamera() {
  stopStream()
  errorText.value = ''
  status.value = captured.value ? 'previewing' : 'idle'
}

function onDeviceChange(event) {
  const deviceId = event.target.value
  if (deviceId) startCamera(deviceId)
}

/** 切换采集方式：另一模式的预览与提示必须一并失效 */
async function switchMode(next) {
  if (submitting.value || next === mode.value) return
  mode.value = next
  status.value = 'idle'
  notice.value = ''
  errorText.value = ''
  if (next === 'upload') {
    stopStream()
    captured.value = null
    return
  }
  if (!cameraReady.value) await startCamera()
}
</script>


<template>
  <section class="page">
    <h1 class="page__title">人脸采集</h1>
    <p class="page__subtitle">上传本地图片或摄像头拍照，填写姓名后写入特征库</p>

    <div class="collect-tabs" role="tablist" aria-label="采集方式">
      <button
        id="tab-upload"
        class="collect-tab"
        :class="{ 'collect-tab--active': mode === 'upload' }"
        type="button"
        role="tab"
        aria-controls="panel-upload"
        :aria-selected="mode === 'upload'"
        :disabled="submitting"
        @click="switchMode('upload')"
      >
        ① 本地图片
      </button>
      <button
        id="tab-camera"
        class="collect-tab"
        :class="{ 'collect-tab--active': mode === 'camera' }"
        type="button"
        role="tab"
        aria-controls="panel-camera"
        :aria-selected="mode === 'camera'"
        :disabled="submitting"
        @click="switchMode('camera')"
      >
        ② 摄像头拍照
      </button>
    </div>

    <div class="layout-two-col">
      <div class="card">
        <h2 class="card__title">预览</h2>
        <div
          v-show="mode === 'upload'"
          id="panel-upload"
          class="collect-preview"
          role="tabpanel"
          aria-labelledby="tab-upload"
        >
          <p class="collect-preview__empty">尚未选择图片</p>
        </div>
        <div
          v-show="mode === 'camera'"
          id="panel-camera"
          class="collect-preview collect-preview--camera"
          role="tabpanel"
          aria-labelledby="tab-camera"
        >
          <video
            v-show="!captured"
            ref="videoRef"
            class="collect-preview__media"
            autoplay
            playsinline
            muted
          ></video>
          <img v-if="captured" class="collect-preview__media" :src="captured.dataUrl" alt="已拍摄画面" />
          <p v-if="!cameraReady && !captured" class="collect-preview__empty">{{ cameraHint }}</p>
        </div>

        <div class="collect-actions">
          <template v-if="mode === 'camera'">
            <button
              v-if="!cameraReady"
              class="btn"
              type="button"
              :disabled="cameraStatus === 'starting' || submitting"
              @click="startCamera()"
            >
              {{ cameraStatus === 'starting' ? '正在开启…' : '开启摄像头' }}
            </button>
            <button v-else class="btn btn--ghost" type="button" :disabled="submitting" @click="stopCamera()">
              关闭摄像头
            </button>
            <select
              v-if="devices.length > 1"
              class="select collect-device"
              :value="activeDeviceId"
              :disabled="submitting || !cameraReady"
              aria-label="选择摄像头"
              @change="onDeviceChange"
            >
              <option v-for="(device, index) in devices" :key="device.deviceId" :value="device.deviceId">
                {{ device.label || `摄像头 ${index + 1}` }}
              </option>
            </select>
          </template>
        </div>
      </div>

      <div class="card">
        <h2 class="card__title">录入信息</h2>
        <div class="field">
          <label class="field__label" for="real-name">姓名</label>
          <input
            id="real-name"
            v-model="realName"
            class="input"
            type="text"
            maxlength="64"
            autocomplete="off"
            placeholder="例如：张三"
            :disabled="submitting"
            @compositionstart="composing = true"
            @compositionend="composing = false; normalizeName()"
            @blur="normalizeName"
          />
          <p class="collect-hint" :class="{ 'collect-hint--error': nameTooLong }">
            {{ nameLength }} / {{ MAX_NAME_LEN }} 字符
            <span v-if="nameTooLong">（超出上限，后端将返回 1001）</span>
          </p>
          <p class="muted">姓名为特征库唯一键，与其已有记录重名时会提示先删除旧记录。</p>
        </div>
      </div>
    </div>

    <div v-if="errorText" class="alert alert--danger collect-alert" role="alert">{{ errorText }}</div>
    <div v-if="notice" class="alert collect-alert" role="status">{{ notice }}</div>
  </section>
</template>

<style scoped>
.collect-tabs {
  display: inline-flex;
  gap: 4px;
  padding: 4px;
  margin-bottom: 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-surface);
}

.collect-tab {
  min-height: 32px;
  padding: 5px 16px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--color-muted);
  font-family: inherit;
  font-size: 13px;
  cursor: pointer;
}

.collect-tab:hover:not(:disabled) {
  color: var(--color-text);
}

.collect-tab--active {
  background: #eef4ff;
  color: var(--color-primary);
  font-weight: 600;
}

.collect-tab:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.collect-preview {
  display: flex;
  align-items: center;
  justify-content: center;
  aspect-ratio: 4 / 3;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg);
  overflow: hidden;
}

.collect-preview__empty {
  margin: 0;
  color: var(--color-muted);
  font-size: 13px;
}

.collect-preview--camera {
  background: #0f172a;
}

.collect-preview__media {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.collect-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
}

.collect-device {
  width: auto;
  max-width: 220px;
}

.collect-alert {
  margin-top: 12px;
}

.collect-hint {
  margin: 0;
  color: var(--color-muted);
  font-size: 12px;
}

.collect-hint--error {
  color: var(--color-danger);
}

.muted {
  margin: 0;
  color: var(--color-muted);
  font-size: 12px;
}
</style>
