<script setup>
import { onMounted, ref } from 'vue'
import { getLibrary } from '../api/face'
import { toUserMessage } from '../utils/errors'

// 与 backend/facenet_encoder.py 的 EMBEDDING_DIM 一致；维度自检用（docs/04-05 4.2）
const EXPECTED_FEATURE_DIM = 512

const items = ref([]) // GET /api/face/library → data.items（逐条特征记录）
const loading = ref(false)
const errorText = ref('')
const loadedAt = ref('')

/**
 * 拉取特征库列表（docs/04-05 4.3）
 * - 单飞：loading 为真时直接返回，连点刷新不会产生并发请求
 * - 失败清空 items：避免用户对「过期列表」执行删除
 */
async function load() {
  if (loading.value) return
  loading.value = true
  errorText.value = ''
  try {
    const res = await getLibrary() // 已解包：{ code, message, data }
    items.value = res.data?.items || [] // 字段是 items，不是 records
    loadedAt.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  } catch (err) {
    if (err.isCanceled) return
    errorText.value = toUserMessage(err)
    items.value = []
    loadedAt.value = ''
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <section class="page">
    <header class="library-head">
      <div>
        <h1 class="page__title">特征库管理</h1>
        <p class="page__subtitle">
          共 <strong id="library-total">{{ items.length }}</strong> 条特征 /
          <strong id="library-people">{{ items.length }}</strong> 人
          <span v-if="loadedAt" id="library-loaded-at" class="library-head__time">更新于 {{ loadedAt }}</span>
        </p>
      </div>
      <button id="library-refresh" class="btn btn--ghost" type="button" :disabled="loading" @click="load()">
        {{ loading ? '加载中…' : '刷新' }}
      </button>
    </header>

    <div v-if="errorText" id="library-error" class="state state--error" role="alert">
      <p>{{ errorText }}</p>
      <button class="btn" type="button" @click="load()">重试</button>
    </div>

    <div v-else-if="loading && !items.length" id="library-loading" class="state">正在加载特征库…</div>

    <div v-else-if="!items.length" id="library-empty" class="state state--empty">
      <p>特征库还是空的。</p>
    </div>

    <ul v-else id="library-raw-list" class="library-raw">
      <li v-for="rec in items" :key="rec.id">
        <span class="mono">#{{ rec.id }}</span>
        <strong>{{ rec.name }}</strong>
        <span class="muted">{{ rec.feature_dim }} 维</span>
        <span class="mono muted">{{ rec.timestamp }}</span>
      </li>
    </ul>

    <p class="library-note muted">特征维度应为 {{ EXPECTED_FEATURE_DIM }} 维。</p>
  </section>
</template>

<style scoped>
.library-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.library-head__time {
  margin-left: 12px;
  color: var(--color-muted);
  font-size: 12px;
}

.library-raw {
  display: grid;
  gap: 6px;
  margin: 0;
  padding: 12px 16px;
  list-style: none;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  font-size: 13px;
}

.library-raw li {
  display: flex;
  align-items: center;
  gap: 10px;
}

.library-note {
  margin-top: 12px;
}

.state {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 12px;
  padding: 20px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  font-size: 13px;
}

.state p {
  margin: 0;
}

.state--error {
  border-left: 4px solid var(--color-danger);
  background: #fef2f2;
}

.state--empty {
  border-left: 4px solid var(--color-primary);
}

.mono {
  font-family: var(--font-mono);
}

.muted {
  color: var(--color-muted);
}
</style>
