package com.example.iatfilter

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.example.iatfilter.ui.HomeScreen
import com.example.iatfilter.ui.ResultScreen
import com.example.iatfilter.ui.theme.IATFilterTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            IATFilterTheme {
                Surface(modifier = Modifier.windowInsetsPadding(WindowInsets.safeDrawing)) {
                    var pickedUri by remember { mutableStateOf<Uri?>(null) }
                    val current = pickedUri
                    if (current == null) {
                        HomeScreen(onPhotoPicked = { pickedUri = it })
                    } else {
                        ResultScreen(
                            uri = current,
                            onNewPhoto = { pickedUri = null },
                        )
                    }
                }
            }
        }
    }
}
