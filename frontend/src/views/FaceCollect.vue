<script setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <section class="page">
    <h1 class="page__title">人脸采集</h1>
    <p class="page__subtitle">上传图片或摄像头拍照，填写姓名后写入特征库</p>

    <div class="card">
      <div class="alert alert--warning">
        页面骨架已就绪（M4-1.6），业务逻辑将在阶段 4-2 按 <code>docs/04-02</code> 实现：
        <code>POST /api/face/register/image</code> 与 <code>POST /api/face/register/camera-frame</code>。
      </div>
      <p class="muted">接口函数已封装在 <code>src/api/face.js</code>，页面不直接调用 axios。</p>
      <RouterLink class="btn" to="/">返回首页</RouterLink>
    </div>
  </section>
</template>
