<script setup>
import { computed, onMounted, ref } from 'vue'
import { getLibrary } from '../api/face'
import { toUserMessage } from '../utils/errors'

// 与 backend/facenet_encoder.py 的 EMBEDDING_DIM 一致；维度自检用（docs/04-05 4.2）
const EXPECTED_FEATURE_DIM = 512

const items = ref([]) // GET /api/face/library → data.items（逐条特征记录）
const loading = ref(false)
const errorText = ref('')
const loadedAt = ref('')

/**
 * 逐条记录 → 按人聚合（docs/04-05 4.2）
 * 同一姓名的多条特征只出一行；维度用 Number 归一后 Set 去重，避免 512 与 "512" 并存
 */
const people = computed(() => {
  const map = new Map()
  items.value.forEach((rec) => {
    const person = map.get(rec.name) || {
      name: rec.name,
      count: 0,
      dims: new Set(),
      latest: '',
      ids: [],
    }
    person.count += 1
    person.dims.add(Number(rec.feature_dim))
    person.ids.push(rec.id)
    // timestamp 形如 "2026-10-18 15:02:11"，同为 UTC 字符串，直接比较即可定位最新
    const stamp = String(rec.timestamp || '')
    if (stamp > person.latest) person.latest = stamp
    map.set(rec.name, person)
  })
  return [...map.values()]
    .map((person) => ({ ...person, dims: [...person.dims].sort((a, b) => a - b) }))
    .sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
})

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
          <strong id="library-people">{{ people.length }}</strong> 人
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

    <div v-else class="card library-card">
      <table id="library-people-table" class="table">
        <caption class="sr-only">按人聚合的特征库列表</caption>
        <thead>
          <tr>
            <th scope="col">姓名</th>
            <th scope="col">特征条数</th>
            <th scope="col">维度</th>
            <th scope="col">最近录入 (UTC)</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="person in people" :key="person.name">
            <td>{{ person.name }}</td>
            <td>{{ person.count }}</td>
            <td>
              <span class="badge">{{ person.dims.join(' / ') }}</span>
            </td>
            <td class="mono">{{ person.latest || '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>

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

.library-card {
  padding: 0;
  overflow-x: auto;
}

.library-note {
  margin-top: 12px;
}

.sr-only {
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
