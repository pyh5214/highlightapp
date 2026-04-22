package com.example.iatfilter.ui

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.example.iatfilter.R
import com.example.iatfilter.inference.IATInterpreter
import com.example.iatfilter.inference.ImagePipeline
import com.example.iatfilter.io.ImageIO
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private data class Enhanced(val original: Bitmap, val enhanced: Bitmap)

@Composable
fun ResultScreen(
    uri: Uri,
    onNewPhoto: () -> Unit,
) {
    val context = LocalContext.current
    var state by remember { mutableStateOf<Enhanced?>(null) }
    var error by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(uri) {
        try {
            state = enhance(context, uri)
        } catch (t: Throwable) {
            error = t.message ?: "처리 실패"
        }
    }

    Box(Modifier.fillMaxSize()) {
        when {
            error != null -> ErrorView(error!!, onNewPhoto)
            state == null -> LoadingView()
            else -> ResultView(state!!, onNewPhoto, context)
        }
    }
}

@Composable
private fun LoadingView() {
    Column(
        Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(24.dp))
        Text(stringResource(R.string.processing))
    }
}

@Composable
private fun ErrorView(message: String, onNewPhoto: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.error,
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onNewPhoto) { Text(stringResource(R.string.new_photo)) }
    }
}

@Composable
private fun ResultView(
    result: Enhanced,
    onNewPhoto: () -> Unit,
    context: Context,
) {
    var intensity by remember { mutableFloatStateOf(0.75f) }
    val originalBmp = remember(result.original) { result.original.asImageBitmap() }
    val enhancedBmp = remember(result.enhanced) { result.enhanced.asImageBitmap() }

    Column(Modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color.Black),
            contentAlignment = Alignment.Center,
        ) {
            Image(
                bitmap = originalBmp,
                contentDescription = "original",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
            Image(
                bitmap = enhancedBmp,
                contentDescription = "enhanced",
                modifier = Modifier.fillMaxSize().alpha(intensity),
                contentScale = ContentScale.Fit,
            )
        }

        Column(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = stringResource(R.string.intensity),
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    text = "${(intensity * 100).toInt()}%",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            Slider(
                value = intensity,
                onValueChange = { intensity = it },
                valueRange = 0f..1f,
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedButton(
                onClick = onNewPhoto,
                modifier = Modifier.weight(1f),
            ) { Text(stringResource(R.string.new_photo)) }
            Button(
                onClick = {
                    val blended = ImagePipeline.blend(result.original, result.enhanced, intensity)
                    val saved = ImageIO.saveToGallery(context, blended)
                    Toast.makeText(
                        context,
                        if (saved != null) context.getString(R.string.saved)
                        else context.getString(R.string.save_failed),
                        Toast.LENGTH_SHORT,
                    ).show()
                },
                modifier = Modifier.weight(1f),
            ) { Text(stringResource(R.string.save)) }
        }
    }
}

private suspend fun enhance(context: Context, uri: Uri): Enhanced =
    withContext(Dispatchers.Default) {
        val original = ImageIO.loadBitmap(context, uri)
        val (boxed, box) = ImagePipeline.letterbox(original)
        IATInterpreter(context).use { iat ->
            val processed = iat.runOnBitmap(boxed)
            val enhancedContent = ImagePipeline.cropContent(processed, box)
            val wbLowRes = ImagePipeline.grayWorldWhiteBalance(enhancedContent)
            val sharp = ImagePipeline.gainMapEnhance(original, wbLowRes)
            Enhanced(original = original, enhanced = sharp)
        }
    }
