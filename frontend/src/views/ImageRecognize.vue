<script setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <section class="page">
    <h1 class="page__title">图片识别</h1>
    <p class="page__subtitle">上传图片，返回人脸框、姓名与相似度，可保存标注结果图</p>

    <div class="card">
      <div class="alert alert--warning">
        页面骨架已就绪（M4-1.6），业务逻辑将在阶段 4-3 按 <code>docs/04-03</code> 实现：
        <code>POST /api/recognize/image</code> 与 <code>resolveStaticUrl()</code> 结果图展示。
      </div>
      <p class="muted">阈值区间 0.3 ~ 0.95，默认 0.6；图片上传上限 5 MB。</p>
      <RouterLink class="btn" to="/">返回首页</RouterLink>
    </div>
  </section>
</template>
