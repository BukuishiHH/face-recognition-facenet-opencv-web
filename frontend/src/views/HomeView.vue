<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { getHealth } from '../api/face'
import { hintOf, toUserMessage } from '../utils/errors'

const entries = [
  { to: '/collect', title: '人脸采集', desc: '上传图片或摄像头拍照，录入姓名与特征' },
  { to: '/recognize/image', title: '图片识别', desc: '上传图片识别，可保存标注结果图' },
  { to: '/recognize/camera', title: '实时识别', desc: '摄像头实时截帧识别，叠加人脸框' },
  { to: '/library', title: '特征库管理', desc: '查看已录入人员与删除指定记录' },
]

const health = ref(null)
const loading = ref(false)
const errorMessage = ref('')
const errorHint = ref('')
const checkedAt = ref('')
let controller = null

const deviceLabel = computed(() => {
  const device = health.value?.device
  if (!device) return '—'
  if (device === 'cuda') return 'GPU（CUDA）'
  if (device === 'cpu') return 'CPU'
  return device
})

const librarySize = computed(() => {
  const size = health.value?.library_size
  return typeof size === 'number' ? size : '—'
})

// 重复点击「刷新」或快速重试时先中止上一次请求，避免旧响应覆盖新结果
async function loadHealth() {
  controller?.abort()
  const current = new AbortController()
  controller = current
  loading.value = true
  try {
    const { data } = await getHealth({ signal: current.signal })
    if (controller !== current) return
    health.value = data
    errorMessage.value = ''
    errorHint.value = ''
    checkedAt.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  } catch (error) {
    if (current.signal.aborted) return
    health.value = null
    errorMessage.value = toUserMessage(error) || '健康检查失败，请重试'
    // 网络层失败补一句可操作的排查提示；业务错误用错误码兜底文案
    errorHint.value = [
      error?.isNetwork ? '请确认后端已启动：uvicorn api_server:app --port 8000' : hintOf(error?.code),
    ]
      .filter(Boolean)
      .join(' · ')
  } finally {
    if (controller === current) loading.value = false
  }
}

onMounted(loadHealth)
onBeforeUnmount(() => controller?.abort())
</script>

<template>
  <section class="page">
    <h1 class="page__title">人脸检测识别系统</h1>
    <p class="page__subtitle">OpenCV Haar 检测 · FaceNet 512 维特征 · FastAPI + Vue3</p>

    <div class="card status-card" aria-live="polite">
      <div class="status-card__head">
        <h2 class="card__title">系统状态</h2>
        <button class="btn btn--ghost btn--sm" type="button" :disabled="loading" @click="loadHealth">
          {{ loading ? '检测中…' : '刷新' }}
        </button>
      </div>

      <div v-if="health" class="status-grid">
        <div class="status-item">
          <span class="status-item__label">服务状态</span>
          <span :class="['badge', health.status === 'ok' ? 'badge--success' : 'badge--danger']">
            {{ health.status || '未知' }}
          </span>
        </div>
        <div class="status-item">
          <span class="status-item__label">运行设备</span>
          <span class="status-item__value">{{ deviceLabel }}</span>
        </div>
        <div class="status-item">
          <span class="status-item__label">特征维度</span>
          <span class="status-item__value">{{ health.feature_dim ?? '—' }} 维</span>
        </div>
        <div class="status-item">
          <span class="status-item__label">特征库人数</span>
          <span class="status-item__value">{{ librarySize }}</span>
        </div>
      </div>

      <p v-else-if="loading" class="muted">正在检测服务状态…</p>

      <div v-else class="alert alert--danger status-card__error">
        <div>
          <strong>{{ errorMessage || '健康检查失败' }}</strong>
          <span v-if="errorHint" class="status-card__hint">{{ errorHint }}</span>
        </div>
        <button class="btn btn--sm" type="button" @click="loadHealth">重试</button>
      </div>

      <p v-if="health" class="muted status-card__meta">
        检测器 {{ health.detector || '—' }} · 版本 {{ health.version || '—' }}
        <template v-if="checkedAt"> · 检测于 {{ checkedAt }}</template>
      </p>
    </div>

    <div class="entry-grid">
      <RouterLink v-for="entry in entries" :key="entry.to" :to="entry.to" class="entry-card">
        <h3>{{ entry.title }}</h3>
        <p>{{ entry.desc }}</p>
        <span class="entry-card__arrow">进入 →</span>
      </RouterLink>
    </div>
  </section>
</template>

<style scoped>
.status-card {
  margin-bottom: 16px;
}

.status-card__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.status-card__head .card__title {
  margin: 0;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.status-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg);
}

.status-item__label {
  color: var(--color-muted);
  font-size: 12px;
}

.status-item__value {
  font-size: 16px;
  font-weight: 600;
}

.status-card__error {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.status-card__hint {
  display: block;
  margin-top: 4px;
  color: var(--color-muted);
  font-weight: 400;
}

.status-card__meta {
  margin: 12px 0 0;
}

.entry-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}

.entry-card {
  display: block;
  padding: 18px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  color: var(--color-text);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.entry-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12);
  text-decoration: none;
}

.entry-card h3 {
  margin: 0 0 6px;
  font-size: 16px;
  color: var(--color-primary);
}

.entry-card p {
  margin: 0 0 12px;
  color: var(--color-muted);
  font-size: 13px;
}

.entry-card__arrow {
  font-size: 13px;
  color: var(--color-primary);
}
</style>
