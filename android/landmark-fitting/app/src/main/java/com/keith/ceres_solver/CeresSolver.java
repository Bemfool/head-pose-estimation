package com.keith.ceres_solver;

import android.graphics.Point;

public class CeresSolver {
static {
    System.loadLibrary("android_ceres");
}
    native void initModelLandmarks(Point point);
    native void solve(double[] x, Point[] landmarks);
}
