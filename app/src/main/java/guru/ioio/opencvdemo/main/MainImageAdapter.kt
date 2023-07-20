package guru.ioio.opencvdemo.main

import android.widget.ImageView
import com.chad.library.adapter.base.BaseQuickAdapter
import com.chad.library.adapter.base.viewholder.BaseViewHolder
import guru.ioio.opencvdemo.R

class MainImageAdapter : BaseQuickAdapter<Int, BaseViewHolder>(R.layout.item_image) {
    override fun convert(holder: BaseViewHolder, item: Int) {
        holder.getView<ImageView>(R.id.image).setImageResource(item)
    }
}