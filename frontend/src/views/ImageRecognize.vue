<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { RouterLink } from 'vue-router'
import FaceCanvas from '../components/FaceCanvas.vue'
import {
  DEFAULT_THRESHOLD,
  MAX_UPLOAD_BYTES,
  THRESHOLD_MAX,
  THRESHOLD_MIN,
  recognizeByImage,
} from '../api/face'
import { compressImage, validateImageFile } from '../utils/image'
import { toUserMessage } from '../utils/errors'

const threshold = ref(DEFAULT_THRESHOLD)
const saveResult = ref(false)
const status = ref('idle') // idle | ready | submitting | done | error
const fileInput = ref(null)
const file = ref(null) // 压缩后的 File（实际上传内容）
const previewUrl = ref('')
const imageSize = ref({ width: 0, height: 0 })
const preparing = ref(false)
const dragging = ref(false)
const errorText = ref('')
const notice = ref('')
const result = ref(null)
const appliedThreshold = ref(DEFAULT_THRESHOLD) // 本次结果实际使用的阈值
const clientMs = ref(0)
let controller = null

const submitting = computed(() => status.value === 'submitting')
const hasImage = computed(() => !!file.value)
const activeFace = ref(null) // 鼠标悬停命中的框，用于高亮表格行
const sortedFaces = computed(() => {
  if (!result.value) return []
  // box.x 升序 → box.y 升序 → 原顺序，保证同坐标时排序稳定
  return result.value.faces
    .map((face, index) => ({ face, index }))
    .sort((a, b) =>
      a.face.box.x - b.face.box.x || a.face.box.y - b.face.box.y || a.index - b.index)
    .map(({ face }) => face)
})
const hitRate = computed(() => {
  const total = result.value?.total_faces || 0
  return total ? Math.round((result.value.matched_count / total) * 100) : 0
})
const previewMeta = computed(() => {
  if (!file.value) return ''
  const kb = Math.max(1, Math.round(file.value.size / 1024))
  const { width, height } = imageSize.value
  const dims = width && height ? `${width}×${height} · ` : ''
  return `${file.value.name} · ${dims}${kb} KB`
})

function pickFile() {
  fileInput.value?.click()
}

/** 读取 objectURL 的真实像素尺寸；失败时返回 0，不影响预览 */
function readImageSize(url) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => resolve({ width: 0, height: 0 })
    img.src = url
  })
}

function resetResult() {
  result.value = null
  clientMs.value = 0
  errorText.value = ''
  notice.value = ''
  activeFace.value = null
}

function clearImage() {
  if (controller) {
    controller.abort()
    controller = null
  }
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
  file.value = null
  imageSize.value = { width: 0, height: 0 }
  status.value = 'idle'
  resetResult()
}

/** 统一入口：校验 → 压缩 → 生成预览 → 释放旧 URL */
async function loadImage(picked) {
  if (!picked) return
  const invalid = validateImageFile(picked, MAX_UPLOAD_BYTES)
  if (invalid) {
    errorText.value = invalid
    status.value = file.value ? 'ready' : 'error'
    return
  }

  preparing.value = true
  errorText.value = ''
  notice.value = ''
  try {
    const prepared = await compressImage(picked) // 长边 1600 / q0.85，与 4-2 同一实现
    const nextUrl = URL.createObjectURL(prepared)
    const size = await readImageSize(nextUrl)
    if (controller) {
      controller.abort() // 换图时取消上一次识别，旧结果作废
      controller = null
    }
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = nextUrl
    file.value = prepared
    imageSize.value = size
    resetResult()
    status.value = 'ready'
  } catch (err) {
    errorText.value = err?.message || '图片处理失败，请换一张图片'
    status.value = file.value ? 'ready' : 'error'
  } finally {
    preparing.value = false
  }
}

async function onFileChange(event) {
  const picked = event.target.files && event.target.files[0]
  event.target.value = '' // 同文件重选也能触发 change
  await loadImage(picked)
}

function onDragOver(event) {
  if (submitting.value || preparing.value) return
  event.preventDefault()
  dragging.value = true
}

function onDragLeave() {
  dragging.value = false
}

async function onDrop(event) {
  event.preventDefault()
  dragging.value = false
  if (submitting.value || preparing.value) return
  const picked = event.dataTransfer?.files && event.dataTransfer.files[0]
  await loadImage(picked)
}

async function recognize() {
  if (!file.value) {
    errorText.value = '请先选择图片'
    return
  }
  if (controller) controller.abort()
  const active = new AbortController()
  controller = active
  status.value = 'submitting'
  errorText.value = ''
  notice.value = ''
  result.value = null
  const startedAt = performance.now()

  try {
    const res = await recognizeByImage(file.value, {
      threshold: Number(threshold.value),
      saveResult: saveResult.value,
      signal: active.signal,
    })
    result.value = res.data
    appliedThreshold.value = Number(threshold.value)
    clientMs.value = Math.round(performance.now() - startedAt)
    status.value = 'done'
    notice.value = res.data.library_empty
      ? '特征库为空，仅返回人脸位置，未做身份比对'
      : res.message
  } catch (err) {
    if (err.isCanceled) {
      status.value = file.value ? 'ready' : 'idle'
      return
    }
    errorText.value = toUserMessage(err)
    status.value = 'error'
    result.value = null // 1002 等错误必须清空旧结果
    appliedThreshold.value = Number(threshold.value)
  } finally {
    if (controller === active) controller = null
  }
}

onBeforeUnmount(() => {
  if (controller) controller.abort()
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
})
</script>

<template>
  <section class="page recognize">
    <h1 class="page__title">图片识别</h1>
    <p class="page__subtitle">
      上传一张图片，后端检测全部人脸并与特征库比对；命中绿框、未命中红框，可导出标注图
    </p>

    <div class="layout-two-col">
      <!-- 左列：图片预览 + 覆盖层 -->
      <div class="card recognize-preview">
        <div
          id="recognize-preview"
          :class="['recognize-preview__stage', { 'recognize-preview__stage--filled': previewUrl, 'recognize-preview__stage--drag': dragging }]"
          @dragover="onDragOver"
          @dragleave="onDragLeave"
          @drop="onDrop"
        >
          <FaceCanvas
            v-if="previewUrl && result"
            :src="previewUrl"
            :faces="result.faces"
            class="recognize-preview__canvas"
            @select="activeFace = $event"
          />
          <img v-else-if="previewUrl" :src="previewUrl" class="recognize-preview__image" alt="待识别图片">
          <p v-else class="recognize-preview__empty">
            <template v-if="preparing">图片处理中…</template>
            <template v-else>
              尚未选择图片
              <span>支持 JPG / PNG / WEBP / BMP，≤ 5 MB；长边超过 1600 会自动压缩；也可直接把图片拖到这里</span>
            </template>
          </p>
        </div>
        <p id="recognize-meta" class="recognize-meta muted">
          {{ previewMeta || '文件名 / 尺寸 / 体积将显示在这里' }}
        </p>
      </div>

      <!-- 右列：控制面板 -->
      <div id="recognize-panel" class="card recognize-panel">
        <h2 class="card__title">识别参数</h2>

        <div class="field">
          <label class="field__label" for="threshold">
            相似度阈值 <strong id="threshold-value">{{ Number(threshold).toFixed(2) }}</strong>
          </label>
          <input
            id="threshold"
            v-model.number="threshold"
            class="recognize-range"
            type="range"
            :min="THRESHOLD_MIN"
            :max="THRESHOLD_MAX"
            step="0.01"
            aria-describedby="threshold-hint"
            :disabled="submitting"
          >
          <p id="threshold-hint" class="recognize-hint">
            范围 {{ THRESHOLD_MIN.toFixed(2) }}~{{ THRESHOLD_MAX.toFixed(2) }}，步长 0.01；
            默认 {{ DEFAULT_THRESHOLD }}。越高越严格（误认少、漏认多），越低越宽松，调整后需重新识别。
          </p>
        </div>

        <label class="recognize-checkbox">
          <input id="save-result" v-model="saveResult" type="checkbox" :disabled="submitting">
          <span>保存标注图（后端生成 <code>/static/*.jpg</code>，便于下载留档）</span>
        </label>

          <input
          id="recognize-file"
          ref="fileInput"
          class="recognize-file"
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          @click.stop
          @change="onFileChange"
        >

        <div class="recognize-actions">
          <button class="btn btn--ghost" type="button" :disabled="submitting || preparing" @click="pickFile">
            选择图片
          </button>
          <button
            v-if="hasImage"
            class="btn btn--ghost"
            type="button"
            :disabled="submitting || preparing"
            @click="clearImage"
          >
            清除
          </button>
          <button
            class="btn btn--primary"
            type="button"
            :disabled="!hasImage || submitting || preparing"
            @click="recognize"
          >
            {{ submitting ? '识别中…' : preparing ? '处理中…' : '开始识别' }}
          </button>
        </div>

        <p v-if="preparing" class="alert recognize-alert" role="status">正在校验并压缩图片，请稍候…</p>
        <p v-if="errorText" id="recognize-error" class="alert alert--danger recognize-alert" role="alert">
          {{ errorText }}
        </p>
        <p
          v-if="notice"
          id="recognize-notice"
          :class="['alert', result && result.library_empty ? 'alert--warning' : 'alert--success', 'recognize-alert']"
          role="status"
        >
          {{ notice }}
          <RouterLink
            v-if="result && result.library_empty"
            id="library-empty-link"
            class="recognize-notice__link"
            to="/collect"
          >
            去采集人脸
          </RouterLink>
        </p>

        <dl class="recognize-summary" aria-live="polite">
          <div>
            <dt>人脸数</dt>
            <dd id="summary-faces">{{ result ? result.total_faces : '—' }}</dd>
          </div>
          <div>
            <dt>命中数</dt>
            <dd id="summary-matched">{{ result ? result.matched_count : '—' }}</dd>
          </div>
          <div>
            <dt>后端耗时</dt>
            <dd id="summary-elapsed">{{ result ? `${result.elapsed_ms} ms` : '—' }}</dd>
          </div>
          <div>
            <dt>本次阈值</dt>
            <dd id="summary-threshold">{{ result ? appliedThreshold.toFixed(2) : '—' }}</dd>
          </div>
        </dl>

        <p class="recognize-hint">图片上限 {{ (MAX_UPLOAD_BYTES / 1024 / 1024).toFixed(0) }} MB；识别中按钮会禁用。</p>
      </div>
    </div>

    <!-- 结果表：按 x → y 稳定排序，与 canvas 框数一一对应 -->
    <section v-if="result" id="recognize-result" class="card recognize-result">
      <div class="recognize-result__head">
        <h2 class="card__title">识别结果</h2>
        <p class="recognize-result__summary muted">
          {{ result.total_faces }} 张人脸 · 命中 {{ result.matched_count }} / {{ result.total_faces }}（{{ hitRate }}%）
          · 本次阈值 {{ appliedThreshold.toFixed(2) }}
        </p>
      </div>

      <div class="recognize-result__scroll">
        <table id="result-table" class="table result-table">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">姓名</th>
              <th scope="col">相似度</th>
              <th scope="col">状态</th>
              <th scope="col">坐标 (x, y, w, h)</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(face, i) in sortedFaces"
              :key="`${face.box.x}-${face.box.y}-${i}`"
              :class="[
                'result-row',
                face.matched ? 'result-row--hit' : 'result-row--miss',
                { 'result-row--active': activeFace === face },
              ]"
            >
              <td>{{ i + 1 }}</td>
              <td>{{ face.name || '未知' }}</td>
              <td>{{ face.similarity.toFixed(4) }}</td>
              <td>
                <span :class="['badge', face.matched ? 'badge--success' : 'badge--danger']">
                  {{ face.matched ? '命中' : '未命中' }}
                </span>
              </td>
              <td><code>{{ face.box.x }}, {{ face.box.y }}, {{ face.box.w }}, {{ face.box.h }}</code></td>
            </tr>
            <tr v-if="!sortedFaces.length">
              <td colspan="5" class="recognize-result__empty">未检测到人脸</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p class="recognize-hint">按 x 从左到右、再按 y 从上到下排序；鼠标悬停图片中的人脸可高亮对应行。</p>
    </section>
  </section>
</template>

<style scoped>
.recognize-preview {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.recognize-preview__stage {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 320px;
  padding: 16px;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg);
  overflow: hidden;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.recognize-preview__stage--filled {
  padding: 0;
  border-style: solid;
  background: #0f172a;
}

.recognize-preview__stage--drag {
  border-color: var(--color-primary);
  background: #eef4ff;
}

.recognize-preview__image {
  display: block;
  max-width: 100%;
  max-height: 60vh;
  width: auto;
  height: auto;
  margin: 0 auto;
  object-fit: contain;
}

.recognize-preview__canvas {
  max-width: 100%;
}

.recognize-preview__empty {
  margin: 0;
  text-align: center;
  color: var(--color-muted);
  font-size: 14px;
}

.recognize-preview__empty span {
  display: block;
  margin-top: 6px;
  font-size: 12px;
}

.recognize-meta {
  margin: 0;
  font-size: 12px;
  word-break: break-all;
}

.recognize-alert {
  margin: 0;
}

.recognize-notice__link {
  margin-left: 8px;
  color: inherit;
  font-weight: 600;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.recognize-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.recognize-panel .card__title {
  margin: 0;
}

.recognize-range {
  width: 100%;
  accent-color: var(--color-primary);
}

.recognize-hint {
  margin: 0;
  color: var(--color-muted);
  font-size: 12px;
  line-height: 1.5;
}

.recognize-checkbox {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 13px;
  cursor: pointer;
}

.recognize-checkbox input {
  margin-top: 3px;
  accent-color: var(--color-primary);
}

.recognize-file {
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

.recognize-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.recognize-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
  padding: 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg);
}

.recognize-summary div {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.recognize-summary dt {
  color: var(--color-muted);
  font-size: 12px;
}

.recognize-summary dd {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
}

.recognize-result {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.recognize-result__head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 4px 16px;
}

.recognize-result__head .card__title {
  margin: 0;
}

.recognize-result__summary {
  margin: 0;
  font-size: 13px;
}

.recognize-result__scroll {
  overflow-x: auto;
}

.result-table td code {
  color: var(--color-muted);
  font-size: 12px;
  white-space: nowrap;
}

.result-row--hit {
  background: #f6fdf8;
}

.result-row--miss {
  background: #fffafa;
}

.result-row--active {
  outline: 2px solid var(--color-primary);
  outline-offset: -2px;
}

.recognize-result__empty {
  padding: 16px 10px;
  text-align: center;
  color: var(--color-muted);
}
</style>

