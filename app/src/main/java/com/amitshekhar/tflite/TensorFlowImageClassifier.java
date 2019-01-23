package com.amitshekhar.tflite;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static android.content.ContentValues.TAG;

/**
 * Created by amitshekhar on 17/03/18.
 */

public class TensorFlowImageClassifier implements Classifier {

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private static final int NUM_CLASSES = 7;

    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;

    private float mObjThresh = 0.1f;
    private float mNmsThresh = 0.5f;

    private static final int[] ANCHORS = {
            10,14,  23,27,  37,58, 81,82,  135,169,  344,319
    };

    private static final String[] LABELS = {
            "towercrane",
            "pushdozer",
            "motocrane",
            "smog",
            "digger",
            "pumper",
            "fire"
    };

    private final int[][] MASKS = {{3,4,5},{0,1,2}};
    private static final int NUM_BOXES_PER_BLOCK = 3;

    private TensorFlowImageClassifier() {

    }

    static TensorFlowImageClassifier create(AssetManager assetManager,
                             String modelPath,
                             String labelPath,
                             int inputSize) throws IOException {

        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath));
        classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;

        return classifier;
    }

    public static float[] normalizeBitmap(Bitmap source,int size,float mean,float std){

        float[] output = new float[size * size * 3];

        int[] intValues = new int[source.getHeight() * source.getWidth()];

        source.getPixels(intValues, 0, source.getWidth(), 0, 0, source.getWidth(), source.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            output[i * 3] = (((val >> 16) & 0xFF) - mean)/std;
            output[i * 3 + 1] = (((val >> 8) & 0xFF) - mean)/std;
            output[i * 3 + 2] = ((val & 0xFF) - mean)/std;
        }

        return output;

    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        return null;
    }

    public ArrayList<YoloRecognition> yoloRecognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        //byte[][] result = new byte[1][labelList.size()];

        float[][][][][] out1 = new float[1][13][13][3][12];
        float[][][][][] out2 = new float[1][26][26][3][12];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, out1);
        outputMap.put(1, out2);

        Object[] inputArray = {byteBuffer};
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        ArrayList<YoloRecognition> detections = new ArrayList<YoloRecognition>();

        Log.d("wangmin", "yolo1 detect start");
        int gridWidth = 13;
        for (int y = 0; y < gridWidth; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    final float confidence = expit(out1[0][y][x][b][4]);
                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = out1[0][y][x][b][5+c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;

                    if (confidenceInClass > mObjThresh) {
                        final float xPos = (x + expit(out1[0][y][x][b][0])) * (MainActivity.INPUT_SIZE / gridWidth);
                        final float yPos = (y + expit(out1[0][y][x][b][1])) * (MainActivity.INPUT_SIZE / gridWidth);

                        final float w = (float) (Math.exp(out1[0][y][x][b][2]) * ANCHORS[2 * MASKS[0][b] + 0]);
                        final float h = (float) (Math.exp(out1[0][y][x][b][3]) * ANCHORS[2 * MASKS[0][b] + 1]);

                        Log.d("wangmin","box x:" + xPos + ", y:" + yPos + ", w:" + w + ", h:" + h);

                        final RectF rect =
                                new RectF(
                                        Math.max(0, xPos - w / 2),
                                        Math.max(0, yPos - h / 2),
                                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                        Log.d("wangmin", "detect " + LABELS[detectedClass] + ", confidence: " + confidenceInClass
                            + ", box: " + rect.toString());
                        detections.add(new YoloRecognition("" + offset, LABELS[detectedClass], confidenceInClass, rect, detectedClass));
                    }
                }
            }
        }
        Log.d("wangmin", "yolo1 detect end");

        Log.d("wangmin", "yolo2 detect start");
        gridWidth = 26;
        for (int y = 0; y < gridWidth; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    final float confidence = expit(out2[0][y][x][b][4]);
                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = out2[0][y][x][b][5+c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;

                    if (confidenceInClass > mObjThresh) {
                        final float xPos = (x + expit(out2[0][y][x][b][0])) * (MainActivity.INPUT_SIZE / gridWidth);
                        final float yPos = (y + expit(out2[0][y][x][b][1])) * (MainActivity.INPUT_SIZE / gridWidth);

                        final float w = (float) (Math.exp(out2[0][y][x][b][2]) * ANCHORS[2 * MASKS[1][b] + 0]);
                        final float h = (float) (Math.exp(out2[0][y][x][b][3]) * ANCHORS[2 * MASKS[1][b] + 1]);

                        final RectF rect =
                                new RectF(
                                        Math.max(0, xPos - w / 2),
                                        Math.max(0, yPos - h / 2),
                                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                        Log.d("wangmin", "detect " + LABELS[detectedClass] + ", confidence: " + confidenceInClass
                                + ", box: " + rect.toString());
                        detections.add(new YoloRecognition("" + offset, LABELS[detectedClass], confidenceInClass, rect, detectedClass));
                    }
                }
            }
        }
        Log.d("wangmin", "yolo2 detect end");

        final ArrayList<YoloRecognition> recognitions = nms(detections);

        return recognitions;
    }

    //non maximum suppression
    private ArrayList<YoloRecognition> nms(ArrayList<YoloRecognition> list) {
        ArrayList<YoloRecognition> nmsList = new ArrayList<YoloRecognition>();

        for (int k = 0; k < NUM_CLASSES; k++) {
            //1.find max confidence per class
            PriorityQueue<YoloRecognition> pq =
                    new PriorityQueue<YoloRecognition>(
                            10,
                            new Comparator<YoloRecognition>() {
                                @Override
                                public int compare(final YoloRecognition lhs, final YoloRecognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).detectedClass == k) {
                    pq.add(list.get(i));
                }
            }
            Log.d("wangmin", "class[" + k + "] pq size: " + pq.size());

            //2.do non maximum suppression
            while(pq.size() > 0) {
                //insert detection with max confidence
                YoloRecognition[] a = new YoloRecognition[pq.size()];
                YoloRecognition[] detections = pq.toArray(a);
                YoloRecognition max = detections[0];
                nmsList.add(max);

                Log.d("wangmin", "before nms pq size: " + pq.size());

                //clear pq to do next nms
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    YoloRecognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh){
                        pq.add(detection);
                    }
                }
                Log.d("wangmin", "after nms pq size: " + pq.size());
            }
        }
        return nmsList;
    }

    float box_iou(RectF a, RectF b)
    {
        return box_intersection(a, b)/box_union(a, b);
    }

    float box_intersection(RectF a, RectF b)
    {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if(w < 0 || h < 0) return 0;
        float area = w*h;
        return area;
    }

    float box_union(RectF a, RectF b)
    {
        float i = box_intersection(a, b);
        float u = (a.right - a.left)*(a.bottom - a.top) + (b.right - b.left)*(b.bottom - b.top) - i;
        return u;
    }

    float overlap(float x1, float w1, float x2, float w2)
    {
        float l1 = x1 - w1/2;
        float l2 = x2 - w2/2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1/2;
        float r2 = x2 + w2/2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }


    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResult(byte[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public class YoloRecognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /**
         * Optional location within the source image for the location of the recognized object.
         */
        private RectF location;

        private int detectedClass;

        public YoloRecognition(
                final String id, final String title, final Float confidence, final RectF location, int detectedClass) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.detectedClass = detectedClass;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }
}
