package com.mikul17.gesturerecognition

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.Image
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Załaduj model TensorFlow Lite
        interpreter = Interpreter(loadModelFile(this, "model.tflite"))

        // Ustaw jednowątkowy executor do obsługi kamerki
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Prośba o uprawnienia do kamerki
        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            if (!isGranted) {
                Log.e("MainActivity", "Brak uprawnień do kamerki!")
            }
        }
        requestPermissionLauncher.launch(Manifest.permission.CAMERA)

        // Ustaw interfejs w Jetpack Compose
        setContent {
            CameraScreen(
                interpreter = interpreter,
                cameraExecutor = cameraExecutor,
                lifecycleOwner = this
            )
        }

    }

    private fun loadModelFile(context: Context, modelFileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        interpreter.close()
    }
}

@Composable
fun CameraScreen(
    interpreter: Interpreter,
    cameraExecutor: ExecutorService,
    lifecycleOwner: LifecycleOwner
) {
    val context = LocalContext.current

    // Remember the PreviewView
    val previewView = remember { androidx.camera.view.PreviewView(context) }

    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    val predictionResult = remember { mutableStateOf("") }

    LaunchedEffect(cameraProviderFuture) {
        val cameraProvider = cameraProviderFuture.get()
        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(224, 224))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalysis.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
            processImageProxy(interpreter, imageProxy, predictionResult)
        })

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
        } catch (exc: Exception) {
            Log.e("CameraScreen", "Use case binding failed", exc)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        Text(
            text = predictionResult.value,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary
        )
    }
}

private fun processImageProxy(
    interpreter: Interpreter,
    imageProxy: ImageProxy,
    predictionResult: MutableState<String>
) {
    val image = imageProxy.image ?: run {
        imageProxy.close()
        return
    }

    val bitmap = imageProxyToBitmap(imageProxy)

    // Pobierz oczekiwany kształt danych wejściowych
    val inputShape = interpreter.getInputTensor(0).shape()
    val height = inputShape[1]
    val width = inputShape[2]

    // Zmień rozmiar bitmapy do wymaganego rozmiaru
    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)

    // Użyj TensorImage do przetworzenia obrazu
    val tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(resizedBitmap)

    // Jeśli konieczne, dodaj przetwarzanie (np. normalizację)
    val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(0f, 255f)) // Dostosuj do wymagań modelu
        .build()

    val processedImage = imageProcessor.process(tensorImage)
    val inputBuffer = processedImage.buffer

    // Przygotowanie bufora wyjściowego
    val outputShape = interpreter.getOutputTensor(0).shape()
    val outputDataType = interpreter.getOutputTensor(0).dataType()
    val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

    // Wykonanie inferencji
    interpreter.run(inputBuffer, outputBuffer.buffer.rewind())

    // Przetwarzanie wyniku
    val outputArray = outputBuffer.floatArray
    Log.d("ModelOutput", "Output values: ${outputArray.contentToString()}")

    val maxIndex = outputArray.indices.maxByOrNull { outputArray[it] } ?: -1
    predictionResult.value = "Rozpoznano gest: $maxIndex"

    imageProxy.close()
}


@OptIn(ExperimentalGetImage::class)
private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
    val image = imageProxy.image ?: return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

    val nv21 = yuv420888ToNv21(image)
    val yuvImage = android.graphics.YuvImage(
        nv21,
        android.graphics.ImageFormat.NV21,
        image.width,
        image.height,
        null
    )
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(
        android.graphics.Rect(0, 0, yuvImage.width, yuvImage.height),
        100,
        out
    )
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

private fun yuv420888ToNv21(image: Image): ByteArray {
    val yBuffer = image.planes[0].buffer // Y
    val uBuffer = image.planes[1].buffer // U
    val vBuffer = image.planes[2].buffer // V

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    return nv21
}
