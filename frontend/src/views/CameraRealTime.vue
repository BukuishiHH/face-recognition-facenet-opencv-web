<script setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <section class="page">
    <h1 class="page__title">实时识别</h1>
    <p class="page__subtitle">摄像头逐帧识别，叠加人脸框与相似度；繁忙时静默丢帧</p>

    <div class="card">
      <div class="alert alert--warning">
        页面骨架已就绪（M4-1.6），业务逻辑将在阶段 4-4 按 <code>docs/04-04</code> 实现：
        <code>POST /api/recognize/frame</code>，帧间隔 400 ms、超时 5 s、<code>1008</code> 静默丢弃。
      </div>
      <p class="muted">摄像头仅在 <code>localhost</code> / <code>127.0.0.1</code> 等安全上下文可用。</p>
      <RouterLink class="btn" to="/">返回首页</RouterLink>
    </div>
  </section>
</template>
