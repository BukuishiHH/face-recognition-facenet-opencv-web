/**
 * 人脸框绘制规则（docs/04-03 4.4 / docs/04-04 4.4）
 * 4-3 静态图（FaceCanvas.vue）与 4-4 视频覆盖层共用同一份实现，保证两页视觉一致：
 * 命中绿色 / 未命中红色、2px 描边（选中 4px）、标签含姓名与相似度。
 */

export const FACE_BOX_COLORS = {
  hit: '#16a34a',
  miss: '#dc2626',
}

export const FACE_LABEL_FONT = '13px system-ui, -apple-system, "Segoe UI", sans-serif'
export const FACE_LABEL_HEIGHT = 20
export const FACE_BOX_LINE_WIDTH = 2
export const FACE_BOX_ACTIVE_LINE_WIDTH = 4

/**
 * 在 canvas 2d 上下文里绘制人脸框与标签
 * @param {CanvasRenderingContext2D} ctx  已按 DPR setTransform
 * @param {Array} faces [{ box:{x,y,w,h}, name, similarity, matched }]
 * @param {number} scale 原图坐标 → 画布 CSS 像素的比例
 * @param {{ activeIndex?: number, showLabels?: boolean }} options
 */
export function drawFaceBoxes(ctx, faces, scale, { activeIndex = -1, showLabels = true } = {}) {
  if (!ctx || !Array.isArray(faces) || !scale) return

  faces.forEach((face, index) => {
    if (!face || !face.box) return
    const { x, y, w, h } = face.box
    const color = face.matched ? FACE_BOX_COLORS.hit : FACE_BOX_COLORS.miss
    const left = x * scale
    const top = y * scale

    ctx.lineWidth = index === activeIndex ? FACE_BOX_ACTIVE_LINE_WIDTH : FACE_BOX_LINE_WIDTH
    ctx.strokeStyle = color
    ctx.strokeRect(left, top, w * scale, h * scale)

    if (!showLabels) return
    const label = face.matched
      ? `${face.name} ${(face.similarity * 100).toFixed(1)}%`
      : `未知 ${(face.similarity * 100).toFixed(1)}%`
    ctx.font = FACE_LABEL_FONT
    const tagW = ctx.measureText(label).width + 8
    const tagH = FACE_LABEL_HEIGHT
    const tagTop = top - tagH >= 0 ? top - tagH : top // 顶部空间不足时标签画在框内
    ctx.fillStyle = color
    ctx.fillRect(left, tagTop, tagW, tagH)
    ctx.fillStyle = '#fff'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, left + 4, tagTop + tagH / 2)
  })
}
