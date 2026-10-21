<!-- 人脸框覆盖层：坐标基于原图像素，绘制时按显示宽度等比缩放（docs/04-03 4.4） -->
<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { drawFaceBoxes } from '../utils/draw'

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
  drawFaceBoxes(ctx, props.faces, scale, {
    activeIndex: activeIndex.value,
    showLabels: props.showLabels,
  })
}

/** 悬停切换：命中 emit(face)，移出/空白处 emit(null)，便于页面同步取消表格高亮 */
function setActive(index) {
  if (index === activeIndex.value) return
  activeIndex.value = index
  emit('select', index >= 0 ? props.faces[index] : null)
  draw()
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
  setActive(index)
}

function onLeave() {
  setActive(-1)
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
