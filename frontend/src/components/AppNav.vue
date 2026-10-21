<script setup>
import { RouterLink } from 'vue-router'

const links = [
  { to: '/', label: '首页' },
  { to: '/collect', label: '人脸采集' },
  { to: '/recognize/image', label: '图片识别' },
  { to: '/recognize/camera', label: '实时识别' },
  { to: '/library', label: '特征库管理' },
]
</script>

<template>
  <header class="nav">
    <div class="nav__inner">
      <RouterLink class="nav__brand" to="/">
        <span class="nav__logo">FR</span>
        <span>人脸检测识别系统</span>
      </RouterLink>
      <nav class="nav__links">
        <RouterLink v-for="link in links" :key="link.to" :to="link.to" class="nav__link">
          {{ link.label }}
        </RouterLink>
      </nav>
    </div>
  </header>
</template>

<style scoped>
.nav {
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  box-shadow: var(--shadow);
  position: sticky;
  top: 0;
  z-index: 10;
}

.nav__inner {
  width: min(1200px, 100%);
  margin: 0 auto;
  padding: 0 16px;
  min-height: 56px;
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap;
}

.nav__brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--color-text);
  font-weight: 600;
  font-size: 15px;
  white-space: nowrap;
}

.nav__brand:hover {
  text-decoration: none;
}

.nav__logo {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: var(--color-primary);
  color: #fff;
  font-size: 12px;
  letter-spacing: 0.5px;
}

.nav__links {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}

.nav__link {
  padding: 6px 12px;
  border-radius: var(--radius);
  color: var(--color-muted);
  font-size: 14px;
  transition: background 0.15s ease, color 0.15s ease;
}

.nav__link:hover {
  background: var(--color-bg);
  color: var(--color-text);
  text-decoration: none;
}

.nav__link.router-link-active {
  background: #eef4ff;
  color: var(--color-primary);
  font-weight: 600;
}

@media (max-width: 720px) {
  .nav__inner {
    padding: 8px 12px;
    gap: 8px;
  }

  .nav__link {
    padding: 4px 8px;
    font-size: 13px;
  }
}
</style>
