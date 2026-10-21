import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'home', component: () => import('../views/HomeView.vue'), meta: { title: '首页' } },
  { path: '/collect', name: 'collect', component: () => import('../views/FaceCollect.vue'), meta: { title: '人脸采集' } },
  { path: '/recognize/image', name: 'recognize-image', component: () => import('../views/ImageRecognize.vue'), meta: { title: '图片识别' } },
  { path: '/recognize/camera', name: 'recognize-camera', component: () => import('../views/CameraRealTime.vue'), meta: { title: '实时识别' } },
  { path: '/library', name: 'library', component: () => import('../views/FaceLibrary.vue'), meta: { title: '特征库管理' } },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

// 不使用 keep-alive：采集 / 实时页持有 MediaStream，缓存组件会让摄像头在离开后继续占用设备
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.afterEach((to) => {
  document.title = to.meta.title ? `${to.meta.title} · 人脸检测识别系统` : '人脸检测识别系统'
})

export default router
