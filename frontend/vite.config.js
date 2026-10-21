import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 端口固定：dev 5173 / preview 4173，与后端 CORS 放行列表一致
// strictPort = true：端口占用直接失败，避免换端口后预检被拒
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  // 可选：内网禁止直连 8000 时启用代理，并把 VITE_API_BASE_URL 改为 /api
  // （直连与代理二选一，见 docs/04-01 坑点 13）
  // server: { proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } } },
})
