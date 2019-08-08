package com.keith.ceres_solver;

import android.graphics.Point;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

public class CeresSolver {
    private static final int LANDMARK_NUM = 68;
    private static Point3f[] modelLandmarks = new Point3f[LANDMARK_NUM];

    static {
        System.loadLibrary("android_ceres");
    }

    public static void init(InputStream in) {
        try {
            Log.i("INIT", "================ Init Landmarks.txt ===============");
            InputStreamReader inputReader = new InputStreamReader(in);
            BufferedReader bufReader = new BufferedReader(inputReader);
            String line;
            int i = 0;
            while((line = bufReader.readLine()) != null) {
                String[] nums = line.split(" ");
                modelLandmarks[i] = new Point3f(Double.valueOf(nums[0]),
                                                Double.valueOf(nums[1]),
                                                Double.valueOf(nums[2]));
                Log.i("LANDMARKS", "[" + i + "]==================> "
                        + modelLandmarks[i].x + " "
                        + modelLandmarks[i].y + " "
                        + modelLandmarks[i].z);
                i++;
            }
        } catch (Exception e) {
            Log.e("ERROR", "================ Init Landmarks.txt Failed ===============");
            e.printStackTrace();
        }
        init_();
    }

    static native void init_();
    public static native void solve(double[] x, Point[] landmarks);
    public static native Point3f[] transform(double[] x);
    public static native Point[] transformTo2d(Point3f[] points);
    public static Point3f getLandmarks(int idx) {
        Log.i("GET-LANDMARK", "JNI Call for " + idx + " landmark");
        Log.i("GET-LANDMARK", "Final Get " + modelLandmarks[idx].x + " "
                                                    + modelLandmarks[idx].y + " "
                                                    + modelLandmarks[idx].z);
        return modelLandmarks[idx];
    }
}
