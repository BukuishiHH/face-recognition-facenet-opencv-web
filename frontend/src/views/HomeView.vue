<script setup>
import { RouterLink } from 'vue-router'

const entries = [
  { to: '/collect', title: '人脸采集', desc: '上传图片或摄像头拍照，录入姓名与特征' },
  { to: '/recognize/image', title: '图片识别', desc: '上传图片识别，可保存标注结果图' },
  { to: '/recognize/camera', title: '实时识别', desc: '摄像头实时截帧识别，叠加人脸框' },
  { to: '/library', title: '特征库管理', desc: '查看已录入人员与删除指定记录' },
]
</script>

<template>
  <section class="page">
    <h1 class="page__title">人脸检测识别系统</h1>
    <p class="page__subtitle">OpenCV Haar 检测 · FaceNet 512 维特征 · FastAPI + Vue3</p>

    <div class="entry-grid">
      <RouterLink v-for="entry in entries" :key="entry.to" :to="entry.to" class="entry-card">
        <h3>{{ entry.title }}</h3>
        <p>{{ entry.desc }}</p>
        <span class="entry-card__arrow">进入 →</span>
      </RouterLink>
    </div>

    <p class="muted" style="margin-top: 20px">
      系统状态卡（设备 / 特征维度 / 特征库人数）将在 M4-1.7 接入 <code>GET /api/health</code>。
    </p>
  </section>
</template>

<style scoped>
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
