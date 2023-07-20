package guru.ioio.opencvdemo.main

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import guru.ioio.opencvdemo.App
import guru.ioio.opencvdemo.OpenCVUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc

class MainViewModel : ViewModel() {
    val age: MutableLiveData<String> = MutableLiveData()
    val gender: MutableLiveData<String> = MutableLiveData()

    val faceImage: MutableLiveData<Bitmap> = MutableLiveData()

    private var ageNet: Net? = null
    private var genderNet: Net? = null

    private suspend fun getAgeNet(): Net {
        ageNet?.let { return it }
        ageNet = withContext(Dispatchers.IO) {
            val modelPath = OpenCVUtils.copyOpenCVFile("age_net.caffemodel")
            val configPath = OpenCVUtils.copyOpenCVFile("age_deploy.prototxt")
            Dnn.readNetFromCaffe(configPath, modelPath)
        }
        return ageNet!!
    }

    private suspend fun getGenderNet(): Net {
        genderNet?.let { return it }
        genderNet = withContext(Dispatchers.IO) {
            val modelPath = OpenCVUtils.copyOpenCVFile("gender_net.caffemodel")
            val configPath = OpenCVUtils.copyOpenCVFile("gender_deploy.prototxt")
            Dnn.readNetFromCaffe(configPath, modelPath)
        }
        return genderNet!!
    }

    // copy file from asset to cache dir

    private val imageSize = Size(227.0, 227.0)

    fun getAgeAndGender(resId: Int) {
        viewModelScope.launch {
            predictAgeAndGender(BitmapFactory.decodeResource(App.ins.resources, resId))
        }
    }

    private suspend fun predictAgeAndGender(bitmap: Bitmap) {
        withContext(Dispatchers.IO) {
            val src = OpenCVUtils.resize(bitmap, imageSize)
            val blob = Dnn.blobFromImage(src, 1.0, imageSize, Core.mean(src))
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

    fun detectFaces(context: Context, resId: Int) {
        viewModelScope.launch {
            val img = BitmapFactory.decodeResource(context.resources, resId)
            val faces = FaceDetector.detectFace(img)
            drawFace(img, faces)
            faceImage.postValue(img)
            Log.d("MainViewModel", "faces: ${faces.size}")
        }
    }

    fun drawFace(src: Bitmap, faces: Array<FaceDetector.FaceInfo>) {
        val dst = Mat()
        Utils.bitmapToMat(src, dst)
        for (face in faces) {
            Imgproc.rectangle(
                dst,
                org.opencv.core.Point(face.x.toDouble(), face.y.toDouble()),
                org.opencv.core.Point(
                    (face.x + face.width).toDouble(),
                    (face.y + face.height).toDouble()
                ),
                Scalar(255.0, 0.0, 0.0),
                4
            )
        }
        Utils.matToBitmap(dst, src)
    }
}