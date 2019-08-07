package com.keith.ceres_solver;

import android.graphics.Point;

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
            InputStreamReader inputReader = new InputStreamReader(in);
            BufferedReader bufReader = new BufferedReader(inputReader);
            String line;
            int i = 0;
            while((line = bufReader.readLine()) != null) {
                String[] nums = line.split(" ");
                modelLandmarks[i] = new Point3f(Double.valueOf(nums[0]),
                                                Double.valueOf(nums[1]),
                                                Double.valueOf(nums[2]));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        init_(new Point(), new Point3f());
    }

    static native void init_(Point point2d, Point3f point3f);
    public static native void solve(double[] x, Point[] landmarks);
    public static native Point3f[] transform(double[] x);
    public static native Point[] transformTo2d(Point3f[] points);
    public Point3f getLandmarks(int idx) {
        return modelLandmarks[idx];
    }
}
