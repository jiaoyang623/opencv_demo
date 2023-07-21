package guru.ioio.opencvdemo.main

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
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

    suspend fun detectFace(img: Bitmap, cutFace: Boolean = false): Array<FaceInfo> {
        return withContext(Dispatchers.IO) {
            val faces = Mat()
            val t = System.currentTimeMillis()
            faceDetector.detect(OpenCVUtils.resize(img, imageSize), faces)
            Log.i(TAG, "detectFace: cost ${System.currentTimeMillis() - t}ms")

            val faceInfos = mutableListOf<FaceInfo>()
            for (i in 0 until faces.rows()) {
                faceInfos.add(toFaceInfo(faces.row(i), img, cutFace).apply {
                    Log.i(TAG, "detectFace: $this")
                })
            }

            return@withContext faceInfos.toTypedArray()
        }
    }

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
    private fun toFaceInfo(face: Mat, img: Bitmap, cutFace: Boolean): FaceInfo {
        val x = (face.get(0, 0)[0] / imageSize.width * img.width).toFloat()
        val y = (face.get(0, 1)[0] / imageSize.height * img.height).toFloat()
        val w = (face.get(0, 2)[0] / imageSize.width * img.width).toFloat()
        val h = (face.get(0, 3)[0] / imageSize.height * img.height).toFloat()

        return FaceInfo(
            RectF(x, y, x + w, y + h),
            PointF(
                (face.get(0, 6)[0] / imageSize.width * img.width).toFloat(),
                (face.get(0, 7)[0] / imageSize.height * img.height).toFloat()
            ),
            PointF(
                (face.get(0, 4)[0] / imageSize.width * img.width).toFloat(),
                (face.get(0, 5)[0] / imageSize.height * img.height).toFloat()
            ),
            PointF(
                (face.get(0, 8)[0] / imageSize.width * img.width).toFloat(),
                (face.get(0, 9)[0] / imageSize.height * img.height).toFloat()
            ),
            PointF(
                (face.get(0, 12)[0] / imageSize.width * img.width).toFloat(),
                (face.get(0, 13)[0] / imageSize.height * img.height).toFloat()
            ),
            PointF(
                (face.get(0, 10)[0] / imageSize.width * img.width).toFloat(),
                (face.get(0, 11)[0] / imageSize.height * img.height).toFloat()
            ),
            face.get(0, 14)[0].toFloat(),
            if (cutFace) cutFace(img, x.toInt(), y.toInt(), w.toInt(), h.toInt()) else null
        )
    }

    private fun cutFace(src: Bitmap, x: Int, y: Int, w: Int, h: Int): Bitmap {
        val x0 = x + w / 2
        val y0 = y + h / 2
        val r = w.coerceAtLeast(h) / 2
        val left = (x0 - r).coerceAtLeast(0)
        val top = (y0 - r).coerceAtLeast(0)
        val right = (x0 + r).coerceAtMost(src.width)
        val bottom = (y0 + r).coerceAtMost(src.height)
        return Bitmap.createBitmap(src, left, top, right - left, bottom - top)
    }

    data class FaceInfo(
        val rect: RectF,
        val leftEye: PointF,
        val rightEye: PointF,
        val nose: PointF,
        val mouthLeft: PointF,
        val mouthRight: PointF,
        val score: Float,
        val image: Bitmap? = null
    )
}