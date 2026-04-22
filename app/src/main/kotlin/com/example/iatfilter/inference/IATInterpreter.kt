package com.example.iatfilter.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class IATInterpreter(context: Context) : AutoCloseable {

    val delegateName: String
    private val interpreter: Interpreter
    private val delegate: AutoCloseable?
    private val outBuffer: ByteBuffer

    init {
        val modelBuffer = loadModel(context, MODEL_ASSET)
        val options = Interpreter.Options()

        val compat = CompatibilityList()
        if (compat.isDelegateSupportedOnThisDevice) {
            val gpu = GpuDelegate(compat.bestOptionsForThisDevice)
            options.addDelegate(gpu)
            delegate = gpu
            delegateName = "GPU"
        } else {
            options.setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            options.setUseXNNPACK(true)
            delegate = null
            delegateName = "CPU/XNNPACK"
        }

        interpreter = Interpreter(modelBuffer, options)
        Log.i(TAG, "IAT interpreter ready on $delegateName")

        val shape = interpreter.getOutputTensor(0).shape()
        val bytes = shape.fold(1) { acc, d -> acc * d } * 4
        outBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
    }

    fun runOnBitmap(letterboxed: Bitmap): Bitmap {
        val input = ImagePipeline.toInputBuffer(letterboxed)
        outBuffer.rewind()
        val t0 = System.nanoTime()
        interpreter.run(input, outBuffer)
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "inference: ${elapsedMs}ms ($delegateName)")
        outBuffer.rewind()
        return ImagePipeline.fromOutputBuffer(outBuffer)
    }

    override fun close() {
        interpreter.close()
        delegate?.close()
    }

    private fun loadModel(context: Context, name: String): ByteBuffer {
        val fd = context.assets.openFd(name)
        FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    companion object {
        private const val TAG = "IATInterpreter"
        private const val MODEL_ASSET = "iat_enhance.tflite"
    }
}
