package com.keith.ceres_solver;

/**
 * Point3f is used for clear 3d point expression.
 * And use android.graphics.Point to express 2d point.
 */
public class Point3f {
    public double x;
    public double y;
    public double z;
    public Point3f(double _x, double _y, double _z) {
        x = _x;
        y = _y;
        z = _z;
    }
    public Point3f() { }
}
