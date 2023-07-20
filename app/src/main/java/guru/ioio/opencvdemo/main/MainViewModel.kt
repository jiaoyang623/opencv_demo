package guru.ioio.opencvdemo.main

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import guru.ioio.opencvdemo.App
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import java.io.FileOutputStream

class MainViewModel : ViewModel() {
    val age: MutableLiveData<String> = MutableLiveData()
    val gender: MutableLiveData<String> = MutableLiveData()

    private var ageNet: Net? = null
    private var genderNet: Net? = null

    private suspend fun getAgeNet(): Net {
        ageNet?.let { return it }
        ageNet = withContext(Dispatchers.IO) {
            val modelPath = copyFile("age_net.caffemodel")
            val configPath = copyFile("age_deploy.prototxt")
            Dnn.readNetFromCaffe(configPath, modelPath)
        }
        return ageNet!!
    }

    private suspend fun getGenderNet(): Net {
        genderNet?.let { return it }
        genderNet = withContext(Dispatchers.IO) {
            val modelPath = copyFile("gender_net.caffemodel")
            val configPath = copyFile("gender_deploy.prototxt")
            Dnn.readNetFromCaffe(configPath, modelPath)
        }
        return genderNet!!
    }

    // copy file from asset to cache dir
    private suspend fun copyFile(fileName: String): String =
        withContext(Dispatchers.IO) {
            val dir = App.ins.cacheDir
            val file = File(dir, fileName)
            if (!file.exists()) {
                file.createNewFile()
                App.ins.assets.open(fileName).use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            file.absolutePath
        }

    private val imageSize = Size(227.0, 227.0)

    fun getAgeAndGender(resId: Int) {
        viewModelScope.launch {
            predictAgeAndGender(BitmapFactory.decodeResource(App.ins.resources, resId))
        }
    }

    private suspend fun predictAgeAndGender(bitmap: Bitmap) {
        withContext(Dispatchers.IO) {
            val src = Mat()
            Utils.bitmapToMat(bitmap, src)
            Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2RGB)
            Imgproc.resize(src, src, imageSize)
            val blob = Dnn.blobFromImage(
                src,
                1.0,
                imageSize,
                Core.mean(src)
            )
            getAgeNet().run {
                setInput(blob)
                forward()
            }.let { age.postValue(getAgeFromPredictions(it)) }

            getGenderNet().run {
                setInput(blob)
                forward()
            }.let {
                gender.postValue(getGenderFromPredictions(it))
            }
        }
    }

    private fun getGenderFromPredictions(predictions: Mat): String {
        var maxGenderIndex = 0
        var maxGenderProbability = 0f
        for (i in 0 until predictions.cols()) {
            val genderProbabily = predictions.get(0, i)[0]
            Log.d("MainViewModel", "gender: $i -> $genderProbabily")
            if (genderProbabily > maxGenderProbability) {
                maxGenderProbability = genderProbabily.toFloat()
                maxGenderIndex = i
            }
        }
        Log.d("MainViewModel", "gender: $maxGenderIndex -> $maxGenderProbability")
        return if (maxGenderIndex == 1) "男" else "女"
    }

    private fun getAgeFromPredictions(predictions: Mat): String {
        var maxAgeIndex = 0
        var maxAgeProbability = 0f
        for (i in 0 until predictions.cols()) {
            val ageProbabily = predictions.get(0, i)[0]
            Log.d("MainViewModel", "age: $i -> $ageProbabily")
            if (ageProbabily > maxAgeProbability) {
                maxAgeProbability = ageProbabily.toFloat()
                maxAgeIndex = i
            }
        }
        Log.d("MainViewModel", "age: $maxAgeIndex -> $maxAgeProbability")
        return mapAge(maxAgeIndex)
    }

    private val ageMap = mapOf(
        0 to "0-2",
        1 to "4-6",
        2 to "8-12",
        3 to "15-20",
        4 to "25-32",
        5 to "38-43",
        6 to "48-53",
        7 to "60-100"
    )

    private fun mapAge(index: Int): String = ageMap[index] ?: "unknown"
}