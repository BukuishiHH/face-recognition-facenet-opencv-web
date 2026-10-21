<script setup>
import { computed, ref } from 'vue'
import {
  DEFAULT_THRESHOLD,
  MAX_UPLOAD_BYTES,
  THRESHOLD_MAX,
  THRESHOLD_MIN,
} from '../api/face'

const threshold = ref(DEFAULT_THRESHOLD)
const saveResult = ref(false)
const status = ref('idle') // idle | ready | submitting | done | error
const fileInput = ref(null)

const submitting = computed(() => status.value === 'submitting')
const hasImage = computed(() => status.value !== 'idle')

function pickFile() {
  fileInput.value?.click()
}
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
        <div id="recognize-preview" class="recognize-preview__stage">
          <p class="recognize-preview__empty">
            尚未选择图片
            <span>支持 JPG / PNG / WEBP / BMP，≤ 5 MB；长边超过 1600 会自动压缩</span>
          </p>
        </div>
        <p id="recognize-meta" class="recognize-meta muted">文件名 / 尺寸 / 体积将显示在这里</p>
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
            :disabled="submitting"
          >
          <p class="recognize-hint">
            默认 {{ DEFAULT_THRESHOLD }}；越高越严格（误认少、漏认多），越低越宽松。调整后需重新识别。
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
        >

        <div class="recognize-actions">
          <button class="btn btn--ghost" type="button" :disabled="submitting" @click="pickFile">
            选择图片
          </button>
          <button class="btn" type="button" :disabled="!hasImage || submitting">
            开始识别
          </button>
        </div>

        <dl class="recognize-summary" aria-live="polite">
          <div>
            <dt>人脸数</dt>
            <dd id="summary-faces">—</dd>
          </div>
          <div>
            <dt>命中数</dt>
            <dd id="summary-matched">—</dd>
          </div>
          <div>
            <dt>后端耗时</dt>
            <dd id="summary-elapsed">—</dd>
          </div>
          <div>
            <dt>本次阈值</dt>
            <dd id="summary-threshold">—</dd>
          </div>
        </dl>

        <p class="recognize-hint">图片上限 {{ (MAX_UPLOAD_BYTES / 1024 / 1024).toFixed(0) }} MB；识别中按钮会禁用。</p>
      </div>
    </div>
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
</style>

