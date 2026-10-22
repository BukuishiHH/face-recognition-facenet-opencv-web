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
const keyword = ref('')
const view = ref('people') // people | items
const pendingName = ref('') // 待删除姓名
const pendingCount = ref(0) // 该姓名下的特征条数（写进确认文案）
const dialogRef = ref(null)

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

// 搜索是纯前端过滤：不 watch keyword，也不因此发请求（docs/04-05 4.3 注）
const kw = computed(() => keyword.value.trim().toLowerCase())
const filteredPeople = computed(() =>
  kw.value ? people.value.filter((p) => p.name.toLowerCase().includes(kw.value)) : people.value,
)
const filteredItems = computed(() =>
  kw.value ? items.value.filter((r) => String(r.name).toLowerCase().includes(kw.value)) : items.value,
)
const viewRows = computed(() => (view.value === 'people' ? filteredPeople.value : filteredItems.value))
const filteredEmpty = computed(() => items.value.length > 0 && viewRows.value.length === 0)
// 数据不可信（加载中 / 加载失败已清空）时禁止发起删除，避免对未知数据做破坏性操作
const deleteBlocked = computed(() => loading.value || !!errorText.value)

/** 打开确认框：只填状态与弹窗，真正的删除在用户确认后进行（docs/04-05 4.4） */
function askDelete(person) {
  if (deleteBlocked.value) return
  pendingName.value = person.name
  pendingCount.value = person.count
  dialogRef.value?.showModal() // 原生模态：焦点陷阱 + ESC + ::backdrop
}

function resetPending() {
  pendingName.value = ''
  pendingCount.value = 0
}

function closeDialog() {
  dialogRef.value?.close()
  resetPending()
}

function cancelDelete() {
  closeDialog()
}

async function confirmDelete() {
  // M4-5.5 接入 DELETE 请求与刷新；本里程碑先关闭弹窗
  closeDialog()
}

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

    <template v-else>
      <section class="toolbar">
        <div class="segmented" role="group" aria-label="视图切换">
          <button
            id="library-view-people"
            type="button"
            class="segmented__btn"
            :class="{ 'is-active': view === 'people' }"
            :aria-pressed="view === 'people'"
            @click="view = 'people'"
          >
            按人
          </button>
          <button
            id="library-view-items"
            type="button"
            class="segmented__btn"
            :class="{ 'is-active': view === 'items' }"
            :aria-pressed="view === 'items'"
            @click="view = 'items'"
          >
            明细
          </button>
        </div>
        <label class="search">
          <span class="sr-only">按姓名筛选</span>
          <input id="library-search" v-model="keyword" class="input" type="search" placeholder="按姓名筛选" />
        </label>
      </section>

      <div v-if="filteredEmpty" id="library-no-match" class="state">
        <p>没有匹配「{{ keyword.trim() }}」的用户。</p>
        <button class="link-btn" type="button" @click="keyword = ''">清空搜索</button>
      </div>

      <div v-else class="card library-card">
        <table v-if="view === 'people'" id="library-people-table" class="table">
          <caption class="sr-only">按人聚合的特征库列表</caption>
          <thead>
            <tr>
              <th scope="col">姓名</th>
              <th scope="col">特征条数</th>
              <th scope="col">维度</th>
              <th scope="col">最近录入 (UTC)</th>
              <th scope="col">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="person in filteredPeople" :key="person.name">
              <td>{{ person.name }}</td>
              <td>{{ person.count }}</td>
              <td>
                <span class="badge">{{ person.dims.join(' / ') }}</span>
              </td>
              <td class="mono">{{ person.latest || '—' }}</td>
              <td>
                <button
                  class="btn btn--danger btn--sm"
                  type="button"
                  :aria-label="`删除 ${person.name}`"
                  :disabled="deleteBlocked"
                  @click="askDelete(person)"
                >
                  删除
                </button>
              </td>
            </tr>
          </tbody>
        </table>

        <table v-else id="library-items-table" class="table">
          <caption class="sr-only">特征记录明细</caption>
          <thead>
            <tr>
              <th scope="col">ID</th>
              <th scope="col">姓名</th>
              <th scope="col">维度</th>
              <th scope="col">录入时间 (UTC)</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="rec in filteredItems" :key="rec.id">
              <td class="mono">{{ rec.id }}</td>
              <td>{{ rec.name }}</td>
              <td>{{ rec.feature_dim }}</td>
              <td class="mono">{{ rec.timestamp }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <p class="library-note muted">特征维度应为 {{ EXPECTED_FEATURE_DIM }} 维。</p>

    <dialog ref="dialogRef" id="library-dialog" class="library-dialog" aria-labelledby="library-dialog-title" @close="resetPending">
      <form method="dialog" @submit.prevent>
        <h2 id="library-dialog-title" class="library-dialog__title">删除确认</h2>
        <p id="library-dialog-text" class="library-dialog__text">
          将删除「{{ pendingName }}」的 {{ pendingCount }} 条特征，此操作不可撤销。
        </p>
        <menu class="library-dialog__actions">
          <button class="btn btn--ghost" type="button" @click="cancelDelete">取消</button>
          <button class="btn btn--danger" type="button" @click="confirmDelete">删除</button>
        </menu>
      </form>
    </dialog>
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

/* ------------------------------- 工具栏 ------------------------------- */
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.segmented {
  display: inline-flex;
  gap: 4px;
  padding: 4px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-surface);
}

.segmented__btn {
  min-height: 30px;
  padding: 4px 16px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--color-muted);
  font-family: inherit;
  font-size: 13px;
  cursor: pointer;
}

.segmented__btn:hover {
  color: var(--color-text);
}

.segmented__btn.is-active {
  background: #eef4ff;
  color: var(--color-primary);
  font-weight: 600;
}

.search .input {
  width: 220px;
}

.search .input:focus {
  outline: 2px solid rgba(37, 99, 235, 0.25);
}

.link-btn {
  padding: 0;
  border: none;
  background: none;
  color: var(--color-primary);
  font-family: inherit;
  font-size: 13px;
  cursor: pointer;
  text-decoration: underline;
}

/* ------------------------------- 删除确认框 ------------------------------- */
.library-dialog {
  width: min(420px, calc(100vw - 32px));
  padding: 20px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: 0 20px 40px rgba(15, 23, 42, 0.25);
  color: var(--color-text);
}

.library-dialog::backdrop {
  background: rgba(15, 23, 42, 0.45);
}

.library-dialog__title {
  margin: 0 0 8px;
  font-size: 16px;
}

.library-dialog__text {
  margin: 0;
  font-size: 13px;
  word-break: break-word;
}

.library-dialog__actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin: 20px 0 0;
  padding: 0;
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
