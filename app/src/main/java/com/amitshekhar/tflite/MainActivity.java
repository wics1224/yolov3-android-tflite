package com.amitshekhar.tflite;

import android.content.ContentUris;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject;
    private Bitmap resized_image;
    private Handler mHandler = new Handler();
    private ImageView mContentView;

    public static final int CHOOSE_PHOTO = 2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textViewResult = findViewById(R.id.textViewResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        btnDetectObject = findViewById(R.id.btnDetectObject);
        btnDetectObject.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent("android.intent.action.GET_CONTENT");
                intent.setType("image/*");
                startActivityForResult(intent,CHOOSE_PHOTO);//打开相册
            }
        });
        mContentView = (ImageView) findViewById(R.id.content);

        initTensorFlowAndLoadModel();

    }

    private class DetectThread extends Thread{
        private String mImagePath;

        public DetectThread(String path) {
            mImagePath = path;
        }

        @Override
        public void run() {
            if (!TextUtils.isEmpty(mImagePath)) {
                Log.d("wangmin", "start detect, input size: " + classifier.getInputSize());
                Bitmap bitmap = BitmapFactory.decodeFile(mImagePath);

                resized_image = processBitmap(bitmap,classifier.getInputSize());

                ArrayList<Classifier.Recognition> results = null;

                results = classifier.RecognizeImage(resized_image);

                resized_image = Bitmap.createBitmap(resized_image);
                final Canvas canvas = new Canvas(resized_image);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2.0f);

                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= 0.1) {
                        canvas.drawRect(location, paint);
                    }
                }

                mHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        mContentView.setImageBitmap(resized_image);
                    }
                });

                Log.d("wangmin", "detect end");
            }
        }
    }

    /**
     * Returns a transformation matrix from one reference frame into another.
     * Handles cropping (if maintaining aspect ratio is desired) and rotation.
     *
     * @param srcWidth Width of source frame.
     * @param srcHeight Height of source frame.
     * @param dstWidth Width of destination frame.
     * @param dstHeight Height of destination frame.
     * @param applyRotation Amount of rotation to apply from one frame to another.
     *  Must be a multiple of 90.
     * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
     * cropping the image if necessary.
     * @return The transformation fulfilling the desired requirements.
     */
    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }

    public static Bitmap processBitmap(Bitmap source,int size){

        int image_height = source.getHeight();
        int image_width = source.getWidth();

        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);

        Matrix frameToCropTransformations = getTransformationMatrix(image_width,image_height,size,size,0,false);
        Matrix cropToFrameTransformations = new Matrix();
        frameToCropTransformations.invert(cropToFrameTransformations);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(source, frameToCropTransformations, null);

        return croppedBitmap;


    }

    @Override
    protected void onResume() {
        super.onResume();
        //cameraView.start();
    }

    @Override
    protected void onPause() {
        //cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = new Yolov3Classifier(getAssets());
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDetectObject.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case CHOOSE_PHOTO:
                if (resultCode == RESULT_OK){
                    handleImageOnKitKat(data);
                }
                break;
            default:
                break;
        }
    }

    private void handleImageOnKitKat(Intent data){
        String imagePath = null;
        Uri uri = data.getData();
        if (DocumentsContract.isDocumentUri(MainActivity.this,uri)){
            //如果是document类型的Uri,则通过document id处理
            String docId = DocumentsContract.getDocumentId(uri);
            if ("com.android.providers.media.documents".equals(uri.getAuthority())){
                String id = docId.split(":")[1];//解析出数字格式的id
                String selection = MediaStore.Images.Media._ID + "=" + id;
                imagePath = getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,selection);
            }else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())){
                Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"),Long.valueOf(docId));
                imagePath = getImagePath(contentUri,null);
            }
        }else if ("content".equalsIgnoreCase(uri.getScheme())){
            //如果是content类型的Uri，则使用普通方式处理
            imagePath = getImagePath(uri,null);
        }else if ("file".equalsIgnoreCase(uri.getScheme())){
            //如果是file类型的Uri，直接获取图片路径即可
            imagePath = uri.getPath();
        }

        Log.d("wangmin", "get Image path: " + imagePath);
        if (!TextUtils.isEmpty(imagePath)) {
            new DetectThread(imagePath).start();
        }
    }

    private String getImagePath(Uri uri,String selection){
        String path = null;
        //通过Uri和selection来获取真实的图片路径
        Cursor cursor = getContentResolver().query(uri,null,selection,null,null);
        if (cursor != null){
            if (cursor.moveToFirst()){
                path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }
}
