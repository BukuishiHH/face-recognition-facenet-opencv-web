<script setup>
import AppNav from './components/AppNav.vue'
</script>

<template>
  <div class="app-shell">
    <AppNav />
    <main class="app-main">
      <router-view />
    </main>
    <footer class="app-footer">FaceNet + OpenCV · 阶段 4 Web 前端</footer>
  </div>
</template>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-bg);
}

.app-main {
  flex: 1;
  width: min(1200px, 100%);
  margin: 0 auto;
  padding: 20px 16px 32px;
}

.app-footer {
  padding: 12px;
  text-align: center;
  color: var(--color-muted);
  font-size: 12px;
}
</style>
