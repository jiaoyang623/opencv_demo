package guru.ioio.opencvdemo

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream

object OpenCVUtils {
    fun copyOpenCVFile(fileName: String) = copyFile(fileName, "opencv")

    fun copyFile(fileName: String, dirName: String): String {
        val dir = App.ins.cacheDir.absolutePath + "/$dirName"
        Log.i("OpenCVUtils", "copyFile: $dir")
        if (!File(dir).exists()) {
            File(dir).mkdirs()
        }
        val file = File(dir, fileName)
        if (!file.exists()) {
            file.createNewFile()
            App.ins.assets.open("$dirName/$fileName").use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    fun resize(bitmap: Bitmap, imageSize: org.opencv.core.Size): Mat {
        val dst = Mat()
        Utils.bitmapToMat(bitmap, dst)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_RGBA2RGB)
        Imgproc.resize(dst, dst, imageSize)
        return dst
    }
}