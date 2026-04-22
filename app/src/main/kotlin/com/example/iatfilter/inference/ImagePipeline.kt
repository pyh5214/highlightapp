package com.example.iatfilter.inference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImagePipeline {
    const val SIZE = 512

    data class Letterbox(
        val padX: Int,
        val padY: Int,
        val contentW: Int,
        val contentH: Int,
    )

    fun letterbox(src: Bitmap): Pair<Bitmap, Letterbox> {
        val scale = minOf(SIZE.toFloat() / src.width, SIZE.toFloat() / src.height)
        val contentW = (src.width * scale).toInt().coerceAtLeast(1)
        val contentH = (src.height * scale).toInt().coerceAtLeast(1)
        val padX = (SIZE - contentW) / 2
        val padY = (SIZE - contentH) / 2

        val out = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.BLACK)
            drawBitmap(
                src,
                null,
                Rect(padX, padY, padX + contentW, padY + contentH),
                Paint(Paint.FILTER_BITMAP_FLAG),
            )
        }
        return out to Letterbox(padX, padY, contentW, contentH)
    }

    fun toInputBuffer(bmp: Bitmap): ByteBuffer {
        require(bmp.width == SIZE && bmp.height == SIZE)
        val buf = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(SIZE * SIZE)
        bmp.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        for (px in pixels) {
            buf.putFloat(((px shr 16) and 0xff) / 255f)
            buf.putFloat(((px shr 8) and 0xff) / 255f)
            buf.putFloat((px and 0xff) / 255f)
        }
        buf.rewind()
        return buf
    }

    fun fromOutputBuffer(buf: ByteBuffer): Bitmap {
        buf.rewind()
        val pixels = IntArray(SIZE * SIZE)
        for (i in pixels.indices) {
            val r = (buf.float.coerceIn(0f, 1f) * 255f).toInt()
            val g = (buf.float.coerceIn(0f, 1f) * 255f).toInt()
            val b = (buf.float.coerceIn(0f, 1f) * 255f).toInt()
            pixels[i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    fun unletterbox(processed: Bitmap, box: Letterbox, originalW: Int, originalH: Int): Bitmap {
        val cropped = Bitmap.createBitmap(processed, box.padX, box.padY, box.contentW, box.contentH)
        return Bitmap.createScaledBitmap(cropped, originalW, originalH, true)
    }

    /**
     * Gray-world white balance. Normalizes per-channel means to the overall mean to
     * neutralize color casts (IAT enhance tends to push warm/magenta on phone JPEGs).
     */
    fun grayWorldWhiteBalance(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        var sumR = 0L; var sumG = 0L; var sumB = 0L
        for (px in pixels) {
            sumR += (px shr 16) and 0xff
            sumG += (px shr 8) and 0xff
            sumB += px and 0xff
        }
        val n = pixels.size.toDouble()
        val meanR = (sumR / n).coerceAtLeast(1.0)
        val meanG = (sumG / n).coerceAtLeast(1.0)
        val meanB = (sumB / n).coerceAtLeast(1.0)
        val gray = (meanR + meanG + meanB) / 3.0
        val gR = gray / meanR
        val gG = gray / meanG
        val gB = gray / meanB

        for (i in pixels.indices) {
            val px = pixels[i]
            val a = (px ushr 24) and 0xff
            val r = (((px shr 16) and 0xff) * gR).toInt().coerceIn(0, 255)
            val g = (((px shr 8) and 0xff) * gG).toInt().coerceIn(0, 255)
            val b = ((px and 0xff) * gB).toInt().coerceIn(0, 255)
            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }

    /**
     * Apply IAT's illumination change to the full-res original without upscale blur.
     *
     * The model runs at 512×512 and its direct output is soft after bilinear upscale.
     * Instead, compute the low-res per-pixel gain (enhanced / original), bilinear-upscale
     * the gain map, and multiply with the full-res original. IAT's correction is almost
     * all low-frequency (illumination + color matrix + gamma), so the gain map is
     * low-frequency and upscales cleanly while the original keeps its edges.
     */
    fun gainMapEnhance(originalFull: Bitmap, enhancedLowRes: Bitmap): Bitmap {
        val W = originalFull.width
        val H = originalFull.height
        val cw = enhancedLowRes.width
        val ch = enhancedLowRes.height

        val origLowRes = Bitmap.createScaledBitmap(originalFull, cw, ch, true)
        val enhFull = Bitmap.createScaledBitmap(enhancedLowRes, W, H, true)
        val origBlurred = Bitmap.createScaledBitmap(origLowRes, W, H, true)

        val enhPx = IntArray(W * H)
        val blurPx = IntArray(W * H)
        val fullPx = IntArray(W * H)
        enhFull.getPixels(enhPx, 0, W, 0, 0, W, H)
        origBlurred.getPixels(blurPx, 0, W, 0, 0, W, H)
        originalFull.getPixels(fullPx, 0, W, 0, 0, W, H)

        val out = IntArray(W * H)
        val maxGain = 20f
        for (i in out.indices) {
            val oR = ((blurPx[i] shr 16) and 0xff).coerceAtLeast(1).toFloat()
            val oG = ((blurPx[i] shr 8) and 0xff).coerceAtLeast(1).toFloat()
            val oB = (blurPx[i] and 0xff).coerceAtLeast(1).toFloat()
            val eR = ((enhPx[i] shr 16) and 0xff).toFloat()
            val eG = ((enhPx[i] shr 8) and 0xff).toFloat()
            val eB = (enhPx[i] and 0xff).toFloat()
            val gR = (eR / oR).coerceIn(0f, maxGain)
            val gG = (eG / oG).coerceIn(0f, maxGain)
            val gB = (eB / oB).coerceIn(0f, maxGain)

            val fR = (fullPx[i] shr 16) and 0xff
            val fG = (fullPx[i] shr 8) and 0xff
            val fB = fullPx[i] and 0xff
            val a = (fullPx[i] ushr 24) and 0xff
            val rOut = (fR * gR).toInt().coerceIn(0, 255)
            val gOut = (fG * gG).toInt().coerceIn(0, 255)
            val bOut = (fB * gB).toInt().coerceIn(0, 255)
            out[i] = (a shl 24) or (rOut shl 16) or (gOut shl 8) or bOut
        }
        return Bitmap.createBitmap(out, W, H, Bitmap.Config.ARGB_8888)
    }

    fun cropContent(src: Bitmap, box: Letterbox): Bitmap =
        Bitmap.createBitmap(src, box.padX, box.padY, box.contentW, box.contentH)

    /** alpha-blend enhanced over original (both same size). intensity in [0,1]. */
    fun blend(original: Bitmap, enhanced: Bitmap, intensity: Float): Bitmap {
        val out = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawBitmap(original, 0f, 0f, null)
        val paint = android.graphics.Paint().apply {
            alpha = (intensity.coerceIn(0f, 1f) * 255f).toInt()
        }
        canvas.drawBitmap(enhanced, 0f, 0f, paint)
        return out
    }
}
