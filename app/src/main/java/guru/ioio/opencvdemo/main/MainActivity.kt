package guru.ioio.opencvdemo.main

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.recyclerview.widget.LinearLayoutManager
import guru.ioio.opencvdemo.R
import guru.ioio.opencvdemo.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private val mViewModel: MainViewModel by lazy {
        ViewModelProvider(this)[MainViewModel::class.java].apply {
            age.observe(this@MainActivity) {
                mBinding.age.text = "年龄：$it"
            }
            gender.observe(this@MainActivity) {
                mBinding.gender.text = "性别：$it"
            }
        }
    }
    private val mBinding: ActivityMainBinding by lazy {
        ActivityMainBinding.inflate(layoutInflater).apply {
            recycler.apply {
                adapter = mAdapter
                layoutManager =
                    LinearLayoutManager(this@MainActivity, LinearLayoutManager.HORIZONTAL, false)
            }
        }
    }

    private val mAdapter = MainImageAdapter().apply {
        setOnItemClickListener { _, _, position ->
            val resid = getItem(position)
            mViewModel.getAgeAndGender(resid)
            mBinding.img.setImageResource(resid)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(mBinding.root)
        mAdapter.setList(
            listOf(
                R.drawable.jay,
                R.drawable.bjt,
                R.drawable.zjm,
                R.drawable.zzf
            )
        )
    }
}