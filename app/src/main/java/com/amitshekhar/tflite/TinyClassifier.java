package com.amitshekhar.tflite;

import android.content.res.AssetManager;

import java.io.IOException;

/**
 * Created by WangMin on 2019/1/24.
 */

public class TinyClassifier extends Classifier {

    protected float mObjThresh = 0.1f;

    public TinyClassifier(AssetManager assetManager) throws IOException{
        super(assetManager, "yolov3_tiny_pb.tflite", "my_tiny.txt", 416);
        mAnchors = new int[]{
                10,14,  23,27,  37,58, 81,82,  135,169,  344,319
        };

        mMasks = new int[][]{{3,4,5},{0,1,2}};
        mOutWidth = new int[]{13,26};
        mObjThresh = 0.1f;
    }

    @Override
    protected float getObjThresh() {
        return mObjThresh;
    }
}
