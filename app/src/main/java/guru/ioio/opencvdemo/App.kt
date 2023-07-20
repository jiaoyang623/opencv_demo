package guru.ioio.opencvdemo

import android.app.Application
import org.opencv.android.OpenCVLoader

class App : Application() {
    init {
        ins = this
    }

    companion object {
        lateinit var ins: App
    }

    override fun onCreate() {
        super.onCreate()
        OpenCVLoader.initDebug()
    }
}