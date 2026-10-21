export const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

/** 返回错误文案；合法返回 null。以后端 5 MB 上限为唯一口径 */
export function validateImageFile(file, maxBytes = 5 * 1024 * 1024) {
  if (!file) return '请选择图片文件'
  if (!ALLOWED_TYPES.includes(file.type)) return '仅支持 JPG / PNG / WEBP / BMP 图片'
  if (file.size === 0) return '图片为空，请重新选择'
  if (file.size > maxBytes) return `图片 ${(file.size / 1024 / 1024).toFixed(1)} MB，超过 5 MB 上限`
  return null
}

/** 等比缩放到长边 maxSide，返回 { width, height, scale }；只缩不放 */
export function scaleSize(width, height, maxSide) {
  const longSide = Math.max(width, height)
  if (longSide <= maxSide) return { width, height, scale: 1 }
  const scale = maxSide / longSide
  return { width: Math.round(width * scale), height: Math.round(height * scale), scale }
}

/** canvas → JPEG dataURL，质量 0~1 */
export function canvasToJpeg(canvas, quality = 0.8) {
  return canvas.toDataURL('image/jpeg', quality)
}

/** 长边 > maxSide 时压缩为 JPEG；不超过 threshold 的文件原样返回，避免无谓重编码 */
export async function compressImage(
  file,
  { maxSide = 1600, quality = 0.85, threshold = 800 * 1024, maxBytes = 5 * 1024 * 1024 } = {},
) {
  if (file.size <= threshold) return file
  let source
  let width
  let height
  if (window.createImageBitmap) {
    source = await createImageBitmap(file) // 默认按 EXIF 方向解码
    width = source.width
    height = source.height
  } else {
    source = await new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('图片解码失败'))
      img.src = URL.createObjectURL(file)
    })
    width = source.naturalWidth
    height = source.naturalHeight
  }
  const size = scaleSize(width, height, maxSide)
  const canvas = document.createElement('canvas')
  canvas.width = size.width
  canvas.height = size.height
  canvas.getContext('2d').drawImage(source, 0, 0, size.width, size.height)
  source.close?.()
  if (!window.createImageBitmap && source.src) URL.revokeObjectURL(source.src)
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality))
  if (!blob || blob.size > maxBytes) throw new Error('图片压缩失败或仍超过 5 MB')
  return new File([blob], `upload_${Date.now()}.jpg`, { type: 'image/jpeg' })
}
