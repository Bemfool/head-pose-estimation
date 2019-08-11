package com.keith.ceres_solver;

import android.graphics.Point;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * CeresSolver is android port for Ceres of cpp version.
 * @author Bemfoo
 */
public class CeresSolver {
    private static final String TAG = "CeresSolver";
    /**
     * Number of landmarks. Here is 68 because dlib support 68 landmarks detection.
     */
    private static final int LANDMARK_NUM = 68;

    /**
     * Scale of 3d landmarks or model.
     */
    private static final double RATIO = 100.0;

    /**
     * Model 3d landmarks, loaded from 'landmarks.txt'.
     */
    private static Point3f[] modelLandmarks = new Point3f[LANDMARK_NUM];

    /* Load ceres support library, which is connected with android_ceres.cpp */
    static {
        System.loadLibrary("android_ceres");
    }

    /**
     * This function use to load model 3d landmarks from file and save them into cpp.
     *
     * @param in getResources().getAssets().open("landmarks.txt")
     *           We need use it as a parameter because only in java activity could we get resource.
     */
    public static void init(InputStream in) {
        try {
            Log.i(TAG, "Loading model landmarks from file.");
            InputStreamReader inputReader = new InputStreamReader(in);
            BufferedReader bufReader = new BufferedReader(inputReader);
            String line;
            int i = 0;
            while((line = bufReader.readLine()) != null) {
                String[] nums = line.split(" ");
                modelLandmarks[i] = new Point3f(Double.valueOf(nums[0]) / RATIO,
                                                Double.valueOf(nums[1]) / RATIO,
                                                Double.valueOf(nums[2]) / RATIO);
                Log.i(TAG, "[" + i + "] => "
                        + modelLandmarks[i].x + " "
                        + modelLandmarks[i].y + " "
                        + modelLandmarks[i].z);
                i++;
            }
        } catch (Exception e) {
            Log.e(TAG, "Loading model landmarks from file failed.");
            e.printStackTrace();
        }
        Log.i(TAG, "Loading model landmarks from file succeed.");
        init_();
    }

    /**
     * This function is to init jfieldID, jmethodID and so on.
     */
    static native void init_();

    /**
     * This function is to close the optimality using ceres.
     *
     * @param x a double array of length 6. Also as return.
     *   			x[0]: yaw
     *   			x[1]: pitch
     *   			x[2]: roll
     *   			x[3]: tx
     *   			x[4]: ty
     *   			x[5]: tz
     * @param landmarks 2d point got from dlib
     */
    public static native void solve(double[] x, Point[] landmarks);

    /**
     * Transform a series 3d points to rotate with R(yaw, pitch, roll) and translate
     * with T(tx, ty, tz).
     *
     * @param x a double array of length 6. As transform parameters.
     * @return 3d points after transformed.
     */
    public static native Point3f[] transform(double[] x);

    /**
     * Transform 3d landmarks into 2d landmarks.
     *
     * @param points 3d points to be transformed into 2d.
     * @return 2d points after transformed.
     */
    public static native Point[] transformTo2d(Point3f[] points);

    /**
     * This function is used to save model landmarks into cpp.
     *
     * @param idx the index of model landmark to get.
     * @return the landmark coordinate in corresponding index.
     */
    public static Point3f getLandmarks(int idx) {
        Log.i(TAG, "Cpp calls for " + idx + " landmark");
        Log.i(TAG, "Final get " + modelLandmarks[idx].x + " "
                                     + modelLandmarks[idx].y + " "
                                     + modelLandmarks[idx].z);
        return modelLandmarks[idx];
    }
}
