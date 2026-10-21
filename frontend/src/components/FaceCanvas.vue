<!-- 人脸框覆盖层：坐标基于原图像素，绘制时按显示宽度等比缩放（docs/04-03 4.4） -->
<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  src: { type: String, default: '' },
  faces: { type: Array, default: () => [] }, // [{ box:{x,y,w,h}, name, similarity, matched }]
  showLabels: { type: Boolean, default: true },
  maxHeight: { type: String, default: '60vh' },
})
const emit = defineEmits(['select'])

const wrapRef = ref(null)
const canvasRef = ref(null)
const imgRef = ref(null)
const activeIndex = ref(-1)
let observer = null

function draw() {
  const canvas = canvasRef.value
  const wrap = wrapRef.value
  const img = imgRef.value
  if (!canvas || !wrap || !img || !img.naturalWidth) return
  const rect = wrap.getBoundingClientRect()
  if (!rect.width || !rect.height) return

  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(rect.width * dpr)
  canvas.height = Math.round(rect.height * dpr)
  canvas.style.width = `${rect.width}px`
  canvas.style.height = `${rect.height}px`

  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0) // 高清屏 1:1 绘制
  ctx.clearRect(0, 0, rect.width, rect.height)

  const scale = rect.width / img.naturalWidth // wrap 完全贴合 img，宽高比一致
  props.faces.forEach((face, index) => {
    const { x, y, w, h } = face.box
    const color = face.matched ? '#16a34a' : '#dc2626'
    const left = x * scale
    const top = y * scale
    const width = w * scale
    const height = h * scale
    ctx.lineWidth = activeIndex.value === index ? 4 : 2
    ctx.strokeStyle = color
    ctx.strokeRect(left, top, width, height)

    if (!props.showLabels) return
    const label = face.matched
      ? `${face.name} ${(face.similarity * 100).toFixed(1)}%`
      : `未知 ${(face.similarity * 100).toFixed(1)}%`
    ctx.font = '13px system-ui, -apple-system, "Segoe UI", sans-serif'
    const tagW = ctx.measureText(label).width + 8
    const tagH = 20
    const tagTop = top - tagH >= 0 ? top - tagH : top
    ctx.fillStyle = color
    ctx.fillRect(left, tagTop, tagW, tagH)
    ctx.fillStyle = '#fff'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, left + 4, tagTop + tagH / 2)
  })
}

function onMove(event) {
  const img = imgRef.value
  if (!img || !img.naturalWidth) return
  const rect = img.getBoundingClientRect()
  const scale = rect.width / img.naturalWidth
  const px = (event.clientX - rect.left) / scale
  const py = (event.clientY - rect.top) / scale
  const index = props.faces.findIndex(
    ({ box }) => px >= box.x && px <= box.x + box.w && py >= box.y && py <= box.y + box.h,
  )
  if (index !== activeIndex.value) {
    activeIndex.value = index
    if (index >= 0) emit('select', props.faces[index])
    draw()
  }
}

function onLeave() {
  if (activeIndex.value !== -1) {
    activeIndex.value = -1
    draw()
  }
}

watch(() => [props.faces, props.src], draw, { deep: true })
onMounted(() => {
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(() => draw())
    if (wrapRef.value) observer.observe(wrapRef.value)
  }
  window.addEventListener('resize', draw)
})
onBeforeUnmount(() => {
  observer && observer.disconnect()
  window.removeEventListener('resize', draw)
})

defineExpose({ redraw: draw })
</script>

<template>
  <div ref="wrapRef" class="face-canvas" @mousemove="onMove" @mouseleave="onLeave">
    <img ref="imgRef" :src="src" :style="{ maxHeight }" alt="待识别图片" @load="draw">
    <canvas ref="canvasRef" class="face-canvas__layer" />
  </div>
</template>

<style scoped>
.face-canvas {
  position: relative;
  display: inline-block;
  line-height: 0;
  max-width: 100%;
}

.face-canvas img {
  display: block;
  max-width: 100%;
  height: auto;
  border-radius: 6px;
}

.face-canvas__layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
</style>
