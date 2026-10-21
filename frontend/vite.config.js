import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 端口 / 构建输出等配置在里程碑 M4-1.3 补齐（docs/04-01 4.3）
export default defineConfig({
  plugins: [vue()],
})
