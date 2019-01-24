package com.amitshekhar.tflite;

import android.content.res.AssetManager;

import java.io.IOException;

/**
 * Created by WangMin on 2019/1/24.
 */

public class Yolov3Classifier extends Classifier {

    protected float mObjThresh = 0.1f;

    public Yolov3Classifier(AssetManager assetManager) throws IOException {
        super(assetManager, "yolov3_pb.tflite", "coco.txt", 608);
        mAnchors = new int[]{
                10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
        };

        mMasks = new int[][]{{6,7,8},{3,4,5},{0,1,2}};
        mOutWidth = new int[]{19,38,76};
        mObjThresh = 0.6f;
    }

    @Override
    protected float getObjThresh() {
        return mObjThresh;
    }
}
