<script setup>
import { computed, ref } from 'vue'
import { MAX_NAME_LEN } from '../api/face'

const mode = ref('upload') // upload | camera
const realName = ref('')
const composing = ref(false)
const status = ref('idle') // idle | previewing | submitting | receipt | error

const nameLength = computed(() => [...realName.value.trim()].length)
const nameTooLong = computed(() => nameLength.value > MAX_NAME_LEN)
const submitting = computed(() => status.value === 'submitting')

/** 失焦 / IME 结束后去掉首尾空格；拼音输入过程中不打断候选词 */
function normalizeName() {
  if (composing.value) return
  realName.value = realName.value.trim()
}

/** 切换采集方式：另一模式的预览与提示必须一并失效（停轨在 M4-2.2/4-2.4 接入） */
function switchMode(next) {
  if (submitting.value || next === mode.value) return
  mode.value = next
  status.value = 'idle'
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
          class="collect-preview"
          role="tabpanel"
          aria-labelledby="tab-camera"
        >
          <p class="collect-preview__empty">摄像头未开启</p>
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
