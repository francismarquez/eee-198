package com.example.ecgclassifier

import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import androidx.core.app.ActivityCompat
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.AxisBase
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.formatter.ValueFormatter
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.TransformType
import java.text.DecimalFormat
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class Utils {
    // Storage Permissions
    private val REQUEST_EXTERNAL_STORAGE = 1
    private val PERMISSIONS_STORAGE = arrayOf<String>(
            android.Manifest.permission.READ_EXTERNAL_STORAGE,
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE
    )

    /**
     * Checks if the app has permission to write to device storage
     *
     * If the app does not has permission then the user will be prompted to grant permissions
     *
     * @param activity
     */
    fun verifyStoragePermissions(activity: Activity?) {
        // Check if we have write permission
        val permission = ActivityCompat.checkSelfPermission(activity!!, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            )
        }
    }

    fun getMeanAndStd(inputArray: FloatArray): Array<Double> {
        var sum = 0.0
        var standardDev = 0.0

        for ((_, value) in inputArray.withIndex()) {
            sum += value
        }

        var mean = sum / inputArray.size

        for ((_, value) in inputArray.withIndex()) {
            standardDev += (abs(value - mean)).pow(2.0)
        }

        standardDev = sqrt(standardDev / inputArray.size)

        return arrayOf(mean, standardDev)
    }

    fun standardScaler(inputArray: FloatArray, mean: Double, std: Double): List<Float> {
        var res = FloatArray(1000 * 12)
        for ((index, value) in inputArray.withIndex()) {
            res[index] = (value - mean.toFloat()) / std.toFloat()
        }

        return res.toList()
    }

    fun plotECG(act: Activity, input: String) {
        var csv = input.split(",").map { it.toFloat() }

        var chartLeadI: LineChart = act.findViewById(R.id.chartLeadI)
        var chartLeadII: LineChart = act.findViewById(R.id.chartLeadII)
        var chartLeadIII: LineChart = act.findViewById(R.id.chartLeadIII)
        var chartLeadaVR: LineChart = act.findViewById(R.id.chartLeadaVR)
        var chartLeadaVL: LineChart = act.findViewById(R.id.chartLeadaVL)
        var chartLeadaVF: LineChart = act.findViewById(R.id.chartLeadaVF)
        var chartLeadV1: LineChart = act.findViewById(R.id.chartLeadV1)
        var chartLeadV2: LineChart = act.findViewById(R.id.chartLeadV2)
        var chartLeadV3: LineChart = act.findViewById(R.id.chartLeadV3)
        var chartLeadV4: LineChart = act.findViewById(R.id.chartLeadV4)
        var chartLeadV5: LineChart = act.findViewById(R.id.chartLeadV5)
        var chartLeadV6: LineChart = act.findViewById(R.id.chartLeadV6)

        var leadI = csv.slice(IntRange(0, 999))
        val leadII = csv.slice(IntRange(1000, 1999))
        val leadIII = csv.slice(IntRange(2000, 2999))
        val leadaVR = csv.slice(IntRange(4000, 4999))
        val leadaVL = csv.slice(IntRange(3000, 3999))
        val leadaVF = csv.slice(IntRange(5000, 5999))
        val leadV1 = csv.slice(IntRange(6000, 6999))
        val leadV2 = csv.slice(IntRange(7000, 7999))
        val leadV3 = csv.slice(IntRange(8000, 8999))
        val leadV4 = csv.slice(IntRange(9000, 9999))
        val leadV5 = csv.slice(IntRange(10000, 10999))
        val leadV6 = csv.slice(IntRange(11000, 11999))

        val mappedLeadI = ArrayList(leadI).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadI
            )[index]
            )
        }
        val mappedLeadII = ArrayList(leadII).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadII
            )[index]
            )
        }
        val mappedLeadIII = ArrayList(leadIII).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadIII
            )[index]
            )
        }
        val mappedLeadaVR = ArrayList(leadaVR).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadaVR
            )[index]
            )
        }
        val mappedLeadaVL = ArrayList(leadaVL).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadaVL
            )[index]
            )
        }
        val mappedLeadaVF = ArrayList(leadaVF).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadaVF
            )[index]
            )
        }
        val mappedLeadV1 = ArrayList(leadV1).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadV1
            )[index]
            )
        }
        val mappedLeadV2 = ArrayList(leadV2).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadV2
            )[index]
            )
        }
        val mappedLeadV3 = ArrayList(leadV3).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadV3
            )[index]
            )
        }
        val mappedLeadV4 = ArrayList(leadV4).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadV4
            )[index]
            )
        }
        val mappedLeadV5 = ArrayList(leadV5).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadV5
            )[index]
            )
        }
        val mappedLeadV6 = ArrayList(leadV6).mapIndexed { index, _ ->
            Entry(
                    index.toFloat(), ArrayList(
                    leadV6
            )[index]
            )
        }

        class MyXAxisFormatter : ValueFormatter() {
            override fun getAxisLabel(value: Float, axis: AxisBase?): String {
                return (value / 100).toInt().toString() + 's'
            }
        }

        class MyValueFormatter : ValueFormatter() {
            private val format = DecimalFormat("###,##0.0")
            override fun getAxisLabel(value: Float, axis: AxisBase?): String {
                return format.format(value) + 'V'
            }
        }

        val lineDataSet1 = LineDataSet(mappedLeadI, "I")
        lineDataSet1.color = Color.RED
        lineDataSet1.setDrawValues(true)
        lineDataSet1.setDrawCircles(false)
        lineDataSet1.cubicIntensity = 0.01f
        lineDataSet1.lineWidth = 1.5f

        chartLeadI.axisRight.setDrawLabels(false)
        chartLeadI.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadI.xAxis.axisMaximum = 1000f
        chartLeadI.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadI.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadI.description.isEnabled = false
        chartLeadI.legend.isEnabled = true

        val lineDataSet2 = LineDataSet(mappedLeadII, "II")
        lineDataSet2.color = Color.RED
        lineDataSet2.setDrawValues(true)
        lineDataSet2.setDrawCircles(false)
        lineDataSet2.cubicIntensity = 0.01f
        lineDataSet2.lineWidth = 1.5f

        chartLeadII.axisRight.setDrawLabels(false)
        chartLeadII.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadII.xAxis.axisMaximum = 1000f
        chartLeadII.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadII.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadII.description.isEnabled = false
        chartLeadII.legend.isEnabled = true

        val lineDataSet3 = LineDataSet(mappedLeadIII, "III")
        lineDataSet3.color = Color.RED
        lineDataSet3.setDrawValues(true)
        lineDataSet3.setDrawCircles(false)
        lineDataSet3.cubicIntensity = 0.01f
        lineDataSet3.lineWidth = 1.5f

        chartLeadIII.axisRight.setDrawLabels(false)
        chartLeadIII.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadIII.xAxis.axisMaximum = 1000f
        chartLeadIII.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadIII.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadIII.description.isEnabled = false
        chartLeadIII.legend.isEnabled = true

        val lineDataSet5 = LineDataSet(mappedLeadaVR, "aVR")
        lineDataSet5.color = Color.RED
        lineDataSet5.setDrawValues(true)
        lineDataSet5.setDrawCircles(false)
        lineDataSet5.cubicIntensity = 0.01f
        lineDataSet5.lineWidth = 1.5f

        chartLeadaVR.axisRight.setDrawLabels(false)
        chartLeadaVR.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadaVR.xAxis.axisMaximum = 1000f
        chartLeadaVR.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadaVR.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadaVR.description.isEnabled = false
        chartLeadaVR.legend.isEnabled = true

        val lineDataSet4 = LineDataSet(mappedLeadaVL, "aVL")
        lineDataSet4.color = Color.RED
        lineDataSet4.setDrawValues(true)
        lineDataSet4.setDrawCircles(false)
        lineDataSet4.cubicIntensity = 0.01f
        lineDataSet4.lineWidth = 1.5f

        chartLeadaVL.axisRight.setDrawLabels(false)
        chartLeadaVL.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadaVL.xAxis.axisMaximum = 1000f
        chartLeadaVL.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadaVL.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadaVL.description.isEnabled = false
        chartLeadaVL.legend.isEnabled = true

        val lineDataSet6 = LineDataSet(mappedLeadaVF, "aVF")
        lineDataSet6.color = Color.RED
        lineDataSet6.setDrawValues(true)
        lineDataSet6.setDrawCircles(false)
        lineDataSet6.cubicIntensity = 0.01f
        lineDataSet6.lineWidth = 1.5f

        chartLeadaVF.axisRight.setDrawLabels(false)
        chartLeadaVF.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadaVF.xAxis.axisMaximum = 1000f
        chartLeadaVF.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadaVF.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadaVF.description.isEnabled = false
        chartLeadaVF.legend.isEnabled = true

        val lineDataSet7 = LineDataSet(mappedLeadV1, "V1")
        lineDataSet7.color = Color.RED
        lineDataSet7.setDrawValues(true)
        lineDataSet7.setDrawCircles(false)
        lineDataSet7.cubicIntensity = 0.01f
        lineDataSet7.lineWidth = 1.5f

        chartLeadV1.axisRight.setDrawLabels(false)
        chartLeadV1.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadV1.xAxis.axisMaximum = 1000f
        chartLeadV1.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadV1.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadV1.description.isEnabled = false
        chartLeadV1.legend.isEnabled = true

        val lineDataSet8 = LineDataSet(mappedLeadV2, "V2")
        lineDataSet8.color = Color.RED
        lineDataSet8.setDrawValues(true)
        lineDataSet8.setDrawCircles(false)
        lineDataSet8.cubicIntensity = 0.01f
        lineDataSet8.lineWidth = 1.5f

        chartLeadV2.axisRight.setDrawLabels(false)
        chartLeadV2.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadV2.xAxis.axisMaximum = 1000f
        chartLeadV2.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadV2.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadV2.description.isEnabled = false
        chartLeadV2.legend.isEnabled = true

        val lineDataSet9 = LineDataSet(mappedLeadV3, "V3")
        lineDataSet9.color = Color.RED
        lineDataSet9.setDrawValues(true)
        lineDataSet9.setDrawCircles(false)
        lineDataSet9.cubicIntensity = 0.01f
        lineDataSet9.lineWidth = 1.5f

        chartLeadV3.axisRight.setDrawLabels(false)
        chartLeadV3.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadV3.xAxis.axisMaximum = 1000f
        chartLeadV3.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadV3.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadV3.description.isEnabled = false
        chartLeadV3.legend.isEnabled = true

        val lineDataSet10 = LineDataSet(mappedLeadV4, "V4")
        lineDataSet10.color = Color.RED
        lineDataSet10.setDrawValues(true)
        lineDataSet10.setDrawCircles(false)
        lineDataSet10.cubicIntensity = 0.01f
        lineDataSet10.lineWidth = 1.5f

        chartLeadV4.axisRight.setDrawLabels(false)
        chartLeadV4.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadV4.xAxis.axisMaximum = 1000f
        chartLeadV4.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadV4.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadV4.description.isEnabled = false
        chartLeadV4.legend.isEnabled = true

        val lineDataSet11 = LineDataSet(mappedLeadV5, "V5")
        lineDataSet11.color = Color.RED
        lineDataSet11.setDrawValues(true)
        lineDataSet11.setDrawCircles(false)
        lineDataSet11.cubicIntensity = 0.01f
        lineDataSet11.lineWidth = 1.5f

        chartLeadV5.axisRight.setDrawLabels(false)
        chartLeadV5.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadV5.xAxis.axisMaximum = 1000f
        chartLeadV5.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadV5.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadV5.description.isEnabled = false
        chartLeadV5.legend.isEnabled = true

        val lineDataSet12 = LineDataSet(mappedLeadV6, "V6")
        lineDataSet12.color = Color.RED
        lineDataSet12.setDrawValues(true)
        lineDataSet12.setDrawCircles(false)
        lineDataSet12.cubicIntensity = 0.01f
        lineDataSet12.lineWidth = 1.5f

        chartLeadV6.axisRight.setDrawLabels(false)
        chartLeadV6.xAxis.position = XAxis.XAxisPosition.BOTTOM
        chartLeadV6.xAxis.axisMaximum = 1000f
        chartLeadV6.xAxis.valueFormatter = MyXAxisFormatter()
        chartLeadV6.axisLeft.valueFormatter = MyValueFormatter()
        chartLeadV6.description.isEnabled = false
        chartLeadV6.legend.isEnabled = true

        chartLeadI.data = LineData(lineDataSet1)
        chartLeadII.data = LineData(lineDataSet2)
        chartLeadIII.data = LineData(lineDataSet3)
        chartLeadaVL.data = LineData(lineDataSet4)
        chartLeadaVR.data = LineData(lineDataSet5)
        chartLeadaVF.data = LineData(lineDataSet6)
        chartLeadV1.data = LineData(lineDataSet7)
        chartLeadV2.data = LineData(lineDataSet8)
        chartLeadV3.data = LineData(lineDataSet9)
        chartLeadV4.data = LineData(lineDataSet10)
        chartLeadV5.data = LineData(lineDataSet11)
        chartLeadV6.data = LineData(lineDataSet12)

        chartLeadI.invalidate()
        chartLeadI.notifyDataSetChanged()
        chartLeadII.invalidate()
        chartLeadII.notifyDataSetChanged()
        chartLeadIII.invalidate()
        chartLeadIII.notifyDataSetChanged()
        chartLeadaVR.invalidate()
        chartLeadaVR.notifyDataSetChanged()
        chartLeadaVL.invalidate()
        chartLeadaVL.notifyDataSetChanged()
        chartLeadaVF.invalidate()
        chartLeadaVF.notifyDataSetChanged()
        chartLeadV1.invalidate()
        chartLeadV1.notifyDataSetChanged()
        chartLeadV2.invalidate()
        chartLeadV2.notifyDataSetChanged()
        chartLeadV3.invalidate()
        chartLeadV3.notifyDataSetChanged()
        chartLeadV4.invalidate()
        chartLeadV4.notifyDataSetChanged()
        chartLeadV5.invalidate()
        chartLeadV5.notifyDataSetChanged()
        chartLeadV6.invalidate()
        chartLeadV6.notifyDataSetChanged()

    }

}