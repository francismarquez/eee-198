package com.example.ecgclassifier

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.ecgclassifier.ml.Fcn1dQuantized
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.QuantizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.util.*


open class MainActivity : AppCompatActivity() {
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Utils().verifyStoragePermissions(this)

        findViewById<Button>(R.id.analyze).setOnClickListener(View.OnClickListener {
            val addView: View = LayoutInflater.from(this).inflate(R.layout.ecg_charts, null)
            setContentView(addView)
            println("asd")

            val inputCSV: String = readCSV()
            Utils().plotECG(this, inputCSV)

            analyze(this, inputCSV)
        })
    }


    @RequiresApi(Build.VERSION_CODES.N)
    fun onClickAnalayzeBtn(v: View) {
//        val addView: View = LayoutInflater.from(this).inflate(R.layout.ecg_charts, null)
        setContentView(R.layout.ecg_charts)

        val inputCSV: String = readCSV()
        Utils().plotECG(this, inputCSV)
        analyze(this, inputCSV)
    }

    fun onClickClearBtn(v: View) {
        setContentView(R.layout.activity_main)
    }


    private var selectedFile: Uri? = null

    fun onClickOpenCSV(v: View) {
        val intent = Intent().setType("*/*").setAction(Intent.ACTION_GET_CONTENT)
        startActivityForResult(Intent.createChooser(intent, "Select a file"), 111)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 111 && resultCode == RESULT_OK) {
            val selectedFile = data?.data //The uri with the location of the file
            this.selectedFile = selectedFile
        }
    }

    @RequiresApi(Build.VERSION_CODES.N)
    private fun analyze(ctx: Context, input: String) {
        var ecgData = input.split(",").map { it.toFloat() }


        var (mean, std) = Utils().getMeanAndStd(ecgData.toFloatArray())
        val res = Utils().standardScaler(ecgData.toFloatArray(), mean, std)

        val ecgModel = Fcn1dQuantized.newInstance(ctx)
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1000, 12), DataType.FLOAT32)

        var byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(1000 * 12 * 4)

        for ((_, value) in res.withIndex()) {
            byteBuffer.putFloat(value)
        }

        inputFeature0.loadBuffer(byteBuffer)

        val probabilityProcessor = TensorProcessor.Builder().add(NormalizeOp(127.5f, 127.5f)).add(
                QuantizeOp(
                        128.0f,
                        1 / 128.0f
                )
        ).build()

        val analyzeStart = System.currentTimeMillis()
        val outputs = ecgModel.process(probabilityProcessor.process(inputFeature0))
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val latency = System.currentTimeMillis() - analyzeStart
        findViewById<TextView>(R.id.tvResultsTitle).text = "Results (Latency: $latency ms)"

        val labelsList = Constants().diagnosisTaskLabels
//        val labelsList = Constants().subDiagnosisTaskLabels
//        val labelsList = Constants().superDiagnosisTaskLabels

        val tensorLabels = TensorLabel(labelsList, outputFeature0).mapWithFloatValue
        val floatMap: Map<String, Float> = tensorLabels

        val sortedLabels = floatMap.filterValues { it > 0.55f }
        val topResults = sortedLabels.toList().sortedBy { (_, value) -> value }.reversed().toMap()
        var finalResults = topResults.keys.toString().replace("[", "- ").replace("]", "").replace(", ", "\n" + "- ")
        findViewById<TextView>(R.id.tvResultsContent).text = finalResults.toString()
        println(finalResults)
        // Releases model resources if no longer used.
        ecgModel.close()
    }

    private fun readCSV(): String {
//        val filename = "/data/data/com.example.ecgclassifier/files/raw/x_val_0.csv"
//        val file: File = File(filename)
        val file: File = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "/${File(this.selectedFile?.path).name.toString()}")
        return csvReader().readAll(file).flatten().toString().replace("[", "").replace("]", "")
    }
}

