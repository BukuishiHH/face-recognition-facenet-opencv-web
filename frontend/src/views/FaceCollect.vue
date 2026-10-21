<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { MAX_NAME_LEN, MAX_UPLOAD_BYTES, registerByFrame, registerByImage } from '../api/face'
import { useCamera } from '../composables/useCamera'
import { toUserMessage } from '../utils/errors'
import { compressImage, validateImageFile } from '../utils/image'

const mode = ref('upload') // upload | camera
const realName = ref('')
const composing = ref(false)
const status = ref('idle') // idle | previewing | submitting | receipt | error
const errorText = ref('')
const errorCode = ref(0)
const notice = ref('')
const captured = ref(null) // 摄像头截帧 { dataUrl, width, height }
const selectedFile = ref(null) // 压缩后的 File（上传给后端）
const previewUrl = ref('')
const receipt = ref(null)
const uploadPercent = ref(0)
const fileInput = ref(null)
let submitController = null

const camera = useCamera({ width: 640, height: 480 })
const {
  videoRef,
  status: cameraStatus,
  errorText: cameraError,
  devices,
  activeDeviceId,
  start: startStream,
  stop: stopStream,
  capture: captureFrame,
} = camera

const nameLength = computed(() => [...realName.value.trim()].length)
const nameTooLong = computed(() => nameLength.value > MAX_NAME_LEN)
const nameBlank = computed(() => realName.value.length > 0 && nameLength.value === 0)
const submitting = computed(() => status.value === 'submitting')
const cameraReady = computed(() => cameraStatus.value === 'ready')
const cameraHint = computed(() => {
  if (cameraStatus.value === 'starting') return '正在开启摄像头…'
  if (cameraStatus.value === 'error') return '摄像头启动失败，请查看下方提示'
  return '摄像头未开启'
})
const hasImage = computed(() =>
  mode.value === 'upload' ? !!selectedFile.value : !!captured.value,
)
const canSubmit = computed(
  () =>
    !submitting.value &&
    nameLength.value >= 1 &&
    !nameTooLong.value &&
    hasImage.value,
)
const previewMeta = computed(() => {
  if (mode.value === 'upload') {
    if (!selectedFile.value) return ''
    const kb = Math.max(1, Math.round(selectedFile.value.size / 1024))
    return `${selectedFile.value.name} · ${kb} KB`
  }
  if (captured.value) return `已拍摄 ${captured.value.width}×${captured.value.height}`
  return ''
})
const noticeKind = computed(() => (receipt.value ? 'alert--success' : ''))

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

/** 截当前帧为静态预览；画面未就绪时不产出黑图 */
function takePhoto() {
  const shot = captureFrame({ maxWidth: 640, quality: 0.8 })
  if (!shot) {
    errorText.value = '画面尚未就绪，请等待 1 秒后重试'
    errorCode.value = 0
    status.value = 'error'
    return
  }
  captured.value = shot
  errorText.value = ''
  notice.value = `已拍摄 ${shot.width}×${shot.height} 画面，确认无误后点击「确认入库」`
  status.value = 'previewing'
}

async function retake() {
  captured.value = null
  notice.value = ''
  errorText.value = ''
  // 页面隐藏时轨道已被释放，重拍需要先恢复实时预览
  if (!cameraReady.value) {
    await startCamera()
    return
  }
  status.value = 'idle'
}

/** 释放上一次预览的 Blob URL，避免反复选图内存上涨 */
function releasePreview() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
}

function pickFile() {
  fileInput.value?.click()
}

async function onFileChange(event) {
  const file = event.target.files && event.target.files[0]
  event.target.value = '' // 允许再次选择同一个文件
  if (!file) return
  const invalid = validateImageFile(file, MAX_UPLOAD_BYTES)
  if (invalid) {
    clearSelected() // 丢弃上一次的有效图，避免误提交旧图
    errorText.value = invalid
    errorCode.value = 0
    status.value = 'error'
    return
  }
  try {
    selectedFile.value = await compressImage(file)
  } catch (error) {
    clearSelected()
    errorText.value = error.message || '图片处理失败，请重试'
    errorCode.value = 0
    status.value = 'error'
    return
  }
  releasePreview()
  previewUrl.value = URL.createObjectURL(selectedFile.value)
  captured.value = null
  receipt.value = null
  errorText.value = ''
  errorCode.value = 0
  notice.value = ''
  status.value = 'previewing'
}

function clearSelected() {
  releasePreview()
  selectedFile.value = null
  notice.value = ''
  if (mode.value === 'upload') status.value = 'idle'
}

/** 姓名与图片前置校验 → 调接口 → 回执 / 错误；提交中按钮禁用防重复入库 */
async function submit() {
  const name = realName.value.trim()
  if (!name) {
    errorText.value = '请输入姓名'
    errorCode.value = 0
    status.value = 'error'
    return
  }
  if ([...name].length > MAX_NAME_LEN) {
    errorText.value = `姓名最多 ${MAX_NAME_LEN} 个字符`
    errorCode.value = 0
    status.value = 'error'
    return
  }
  if (mode.value === 'upload' && !selectedFile.value) {
    errorText.value = '请先选择图片'
    errorCode.value = 0
    status.value = 'error'
    return
  }
  if (mode.value === 'camera' && !captured.value) {
    errorText.value = '请先拍照'
    errorCode.value = 0
    status.value = 'error'
    return
  }

  if (submitController) submitController.abort() // 按钮已禁用，兜底
  submitController = new AbortController()
  status.value = 'submitting'
  errorText.value = ''
  errorCode.value = 0
  notice.value = ''
  receipt.value = null
  uploadPercent.value = 0

  try {
    const options = { signal: submitController.signal }
    const res =
      mode.value === 'upload'
        ? await registerByImage(selectedFile.value, name, {
            ...options,
            onUploadProgress: (event) => {
              if (event.total) uploadPercent.value = Math.round((event.loaded / event.total) * 100)
            },
          })
        : await registerByFrame(name, captured.value.dataUrl, options)

    receipt.value = res.data
    notice.value = res.data.face_count > 1
      ? `入库成功：${res.data.name}（record_id=${res.data.record_id}，检测到 ${res.data.face_count} 张人脸）`
      : `入库成功：${res.data.name}（record_id=${res.data.record_id}）`
    status.value = 'receipt'
    selectedFile.value = null // 必须重新选图才能再次提交，规避同名 1004
    releasePreview()
    captured.value = null
    stopStream()
  } catch (error) {
    if (error.isCanceled) {
      status.value = 'previewing'
      return
    }
    errorText.value = toUserMessage(error) || '入库失败，请重试'
    errorCode.value = error.code
    status.value = 'error'
  } finally {
    submitController = null
    uploadPercent.value = 0
  }
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

/** 页面隐藏（切标签 / 最小化）时暂停摄像头，回来后由用户点「开启摄像头」恢复 */
function onVisibilityChange() {
  if (document.hidden && cameraReady.value) {
    stopStream()
    status.value = captured.value ? 'previewing' : 'idle'
  }
}

onMounted(() => document.addEventListener('visibilitychange', onVisibilityChange))

onBeforeUnmount(() => {
  document.removeEventListener('visibilitychange', onVisibilityChange)
  if (submitController) submitController.abort() // 提交中离开页面：中断请求，避免回写已卸载组件
  stopStream()
  releasePreview()
})

// 开发期热更新兜底：HMR 不走完整卸载语义，轨道会一直占用
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    stopStream()
    releasePreview()
  })
}
</script>


<template>
  <section class="page">
    <h1 class="page__title">人脸采集</h1>
    <p class="page__subtitle">上传本地图片或摄像头拍照，填写姓名后写入特征库</p>

    <input
      id="image-file"
      ref="fileInput"
      class="collect-file"
      type="file"
      accept="image/jpeg,image/png,image/webp,image/bmp"
      @change="onFileChange"
    />

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
          <img v-if="previewUrl" class="collect-preview__media" :src="previewUrl" alt="已选图片预览" />
          <p v-else class="collect-preview__empty">尚未选择图片</p>
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
            <button class="btn" type="button" :disabled="!cameraReady || submitting" @click="takePhoto">拍照</button>
            <button v-if="captured" class="btn btn--ghost" type="button" :disabled="submitting" @click="retake">
              重拍
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
          <template v-else>
            <button class="btn" type="button" :disabled="submitting" @click="pickFile">选择图片</button>
            <button v-if="previewUrl" class="btn btn--ghost" type="button" :disabled="submitting" @click="clearSelected">
              清除
            </button>
          </template>
        </div>
        <p v-if="previewMeta" class="muted collect-meta">{{ previewMeta }}</p>
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
          <p v-if="nameBlank" class="collect-hint collect-hint--error">姓名不能只包含空格</p>
          <p class="muted">姓名为特征库唯一键，与其已有记录重名时会提示先删除旧记录。</p>
        </div>

        <div class="collect-submit">
          <button class="btn" type="button" :disabled="!canSubmit" @click="submit">
            {{ submitting ? '提交中…' : '确认入库' }}
          </button>
          <span v-if="submitting && mode === 'upload'" class="muted">上传中 {{ uploadPercent }}%</span>
          <span v-else-if="!hasImage" class="muted">
            {{ mode === 'upload' ? '请先选择图片' : '请先拍照' }}
          </span>
          <span v-else-if="!nameLength" class="muted">请输入姓名</span>
        </div>
      </div>
    </div>

    <div v-if="errorText" class="alert alert--danger collect-alert" role="alert">
      <span>{{ errorText }}</span>
      <RouterLink v-if="errorCode === 1004" class="collect-alert__link" to="/library">去特征库管理</RouterLink>
    </div>
    <div v-if="notice" class="alert collect-alert" :class="noticeKind" role="status">{{ notice }}</div>

    <div v-if="receipt" class="card collect-receipt">
      <div class="collect-receipt__head">
        <h2 class="card__title">入库回执</h2>
        <span class="badge badge--success">已入库</span>
      </div>
      <dl class="collect-receipt__list">
        <dt>姓名</dt>
        <dd>{{ receipt.name }}</dd>
        <dt>记录 ID</dt>
        <dd><code>{{ receipt.record_id }}</code></dd>
        <dt>人脸数</dt>
        <dd>{{ receipt.face_count }}</dd>
        <dt>特征维度</dt>
        <dd>{{ receipt.feature_dim }} 维</dd>
      </dl>
      <p v-if="receipt.face_count > 1" class="alert alert--warning">
        图中存在 {{ receipt.face_count }} 张人脸，后端已选面积最大的 1 张入库，建议使用单人正面照重新采集以提高识别率。
      </p>
      <RouterLink class="btn btn--ghost collect-receipt__link" to="/library">查看特征库</RouterLink>
    </div>
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

.collect-file {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  border: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

.collect-meta {
  margin-top: 8px;
}

.collect-submit {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.collect-device {
  width: auto;
  max-width: 220px;
}

.collect-alert {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
}

.collect-alert__link {
  font-weight: 600;
}

.collect-receipt {
  margin-top: 16px;
}

.collect-receipt__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.collect-receipt__head .card__title {
  margin: 0;
}

.collect-receipt__list {
  display: grid;
  grid-template-columns: 84px minmax(0, 1fr);
  gap: 8px 12px;
  margin: 16px 0;
  font-size: 13px;
}

.collect-receipt__list dt {
  color: var(--color-muted);
}

.collect-receipt__list dd {
  margin: 0;
  font-weight: 600;
  word-break: break-word;
}

.collect-receipt__link {
  margin-top: 4px;
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
