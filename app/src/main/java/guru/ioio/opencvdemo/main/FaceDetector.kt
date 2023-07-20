package guru.ioio.opencvdemo.main

import android.graphics.Bitmap
import android.util.Log
import guru.ioio.opencvdemo.OpenCVUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.objdetect.FaceDetectorYN

object FaceDetector {
    private const val TAG = "FaceDetector"

    private val imageSize = Size(600.0, 600.0)

    private val faceDetector by lazy {
        FaceDetectorYN.create(
            OpenCVUtils.copyOpenCVFile("face_detection_yunet_2023mar.onnx"),
            "",
            imageSize
        )
    }

    suspend fun detectFace(img: Bitmap): Array<FaceInfo> {
        return withContext(Dispatchers.IO) {
            val faces = Mat()
            val t = System.currentTimeMillis()
            faceDetector.detect(OpenCVUtils.resize(img, imageSize), faces)
            Log.i(TAG, "detectFace: cost ${System.currentTimeMillis() - t}ms")
            /**
             * @param faces detection results stored in a 2D cv::Mat of shape [num_faces, 15]
             * - 0-1: x, y of bbox top left corner
             * - 2-3: width, height of bbox
             * - 4-5: x, y of right eye (blue point in the example image)
             * - 6-7: x, y of left eye (red point in the example image)
             * - 8-9: x, y of nose tip (green point in the example image)
             * - 10-11: x, y of right corner of mouth (pink point in the example image)
             * - 12-13: x, y of left corner of mouth (yellow point in the example image)
             * - 14: face score
             * */

            Log.i(
                TAG,
                "detectFace: ${faces.rows()} ${faces.cols()} ${faces.channels()} ${faces.type()}"
            )
            val faceInfos = mutableListOf<FaceInfo>()
            for (i in 0 until faces.rows()) {
                val face = faces.row(i)
                val x = face.get(0, 0)[0] / imageSize.width * img.width
                val y = face.get(0, 1)[0] / imageSize.height * img.height
                val width = face.get(0, 2)[0] / imageSize.width * img.width
                val height = face.get(0, 3)[0] / imageSize.height * img.height
                val score = face.get(0, 14)[0].toFloat()
                faceInfos.add(FaceInfo(x.toInt(), y.toInt(), width.toInt(), height.toInt(), score))
                Log.i(TAG, "detectFace: ${img.width}-${img.height} $x $y $width $height $score")
            }

            return@withContext faceInfos.toTypedArray()
        }
    }

    data class FaceInfo(
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val score: Float
    )
}