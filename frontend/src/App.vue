<script setup>
// M4-1.1 仅用于验证 Vite 工程可启动；M4-1.6 起替换为「顶部导航 + 路由视图」全局布局
const milestones = [
  { id: 'M4-1.1', name: 'Vite 工程初始化', done: true },
  { id: 'M4-1.2', name: '依赖与脚本', done: false },
  { id: 'M4-1.3', name: '环境变量与构建配置', done: false },
  { id: 'M4-1.4', name: 'axios 封装', done: false },
  { id: 'M4-1.5', name: '接口封装', done: false },
  { id: 'M4-1.6', name: '路由与布局', done: false },
  { id: 'M4-1.7', name: '首页状态卡', done: false },
  { id: 'M4-1.8', name: '静态结果图 URL', done: false },
]
</script>

<template>
  <main class="boot">
    <h1>人脸检测识别系统</h1>
    <p class="boot__hint">Vite + Vue3 工程已初始化（阶段 4-1 · M4-1.1）</p>
    <ul class="boot__list">
      <li v-for="item in milestones" :key="item.id" :class="{ 'boot__item--done': item.done }">
        <span>{{ item.id }}</span>
        <span>{{ item.name }}</span>
        <span>{{ item.done ? '✅' : '⬜' }}</span>
      </li>
    </ul>
  </main>
</template>

<style scoped>
.boot {
  max-width: 640px;
  margin: 64px auto;
  padding: 24px;
  background: var(--color-surface);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.boot h1 {
  margin: 0 0 8px;
  font-size: 22px;
}

.boot__hint {
  margin: 0 0 20px;
  color: var(--color-muted);
  font-size: 14px;
}

.boot__list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 6px;
}

.boot__list li {
  display: grid;
  grid-template-columns: 72px 1fr 24px;
  align-items: center;
  padding: 8px 12px;
  border-radius: var(--radius);
  background: var(--color-bg);
  color: var(--color-muted);
  font-size: 13px;
}

.boot__item--done {
  color: var(--color-text);
  background: #eef4ff !important;
}
</style>
