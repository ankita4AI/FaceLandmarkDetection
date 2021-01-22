package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import java.util.Random;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bitmap bitmap = null;
        Bitmap new_bitmap = null;
        Module module = null;
        Tensor imageTensor = null;
        Integer batch_size =  32;
        try {

            bitmap = BitmapFactory.decodeStream(getAssets().open("image_004_1.jpg"));
            new_bitmap = toGrayscale(bitmap);
            new_bitmap = Bitmap.createScaledBitmap(new_bitmap, 224, 224, false);

            
            float[] new_array = ArrayGenerator(batch_size);
            long[] shapeArray = new long[]{batch_size, 1, 224, 224};
            imageTensor = Tensor.fromBlob(new_array, shapeArray);

            String modelFile = assetFilePath(this, "mobile_model_ts.pt");
            module = Module.load(modelFile);

        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(new_bitmap);

//        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(new_bitmap,
//                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final Tensor torchTensor = imageTensor;
        long startTime = System.currentTimeMillis();

//        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final Tensor outputTensor = module.forward(IValue.from(imageTensor)).toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();

        Toast.makeText(this, String.valueOf(System.currentTimeMillis() - startTime), Toast.LENGTH_SHORT).show();
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal) {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public float[] ArrayGenerator(Integer batch_size) {
        float[] arr = new float[batch_size];
        Random randNum = new Random();
        for (int i = 0; i < batch_size; i++) {
            arr[i] = (float) randNum.nextGaussian();
        }
        return arr;
    }


}
