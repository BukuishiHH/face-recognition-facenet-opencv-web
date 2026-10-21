<script setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <section class="page">
    <h1 class="page__title">特征库管理</h1>
    <p class="page__subtitle">查看已录入人员与特征维度，删除指定姓名记录</p>

    <div class="card">
      <div class="alert alert--warning">
        页面骨架已就绪（M4-1.6），业务逻辑将在阶段 4-5 按 <code>docs/04-05</code> 实现：
        <code>GET /api/face/library</code> 与 <code>DELETE /api/face/library?name=</code>。
      </div>
      <p class="muted">每次进入页面重新拉取列表，不做前端缓存。</p>
      <RouterLink class="btn" to="/">返回首页</RouterLink>
    </div>
  </section>
</template>
