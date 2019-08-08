#include <jni.h>
#include <string>
#include <cmath>
#include <iostream>
#include <ceres/ceres.h>
#include <android/log.h>

using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

#define LOG    "ffmpegDemo-jni" // 这个是自定义的LOG的标识
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...)  __android_log_print(ANDROID_LOG_FATAL,LOG,__VA_ARGS__) // 定义LOGF类型

/* Intrinsic parameters (from calibration) */
#define FX 1744.327628674942
#define FY 1747.838275588676
#define CX 800
#define CY 600

/* Number of landmarks */
#define LANDMARK_NUM 68

/* File name of 3d standard model landmarks */
#define LANDMARK_FILE_NAME "landmarks.txt"

jclass point2dClass;
jclass point3fClass;

class Point2d {
public:
    int x{};
    int y{};
public:
    Point2d(int _x, int _y) {
        x = _x;
        y = _y;
    }
    Point2d() = default;
};
jfieldID getX2d;
jfieldID getY2d;
jmethodID init2d;

class Point3f {
public:
    double x;
    double y;
    double z;
public:
    Point3f() = default;
    Point3f(const double _x, const double _y, const double _z) {
        x = _x;
        y = _y;
        z = _z;
    }
    Point3f operator+(Point3f op) {
        Point3f result{};
        result.x = x + op.x;
        result.y = y + op.y;
        result.z = z + op.z;
        return result;
    }
};

jfieldID getX3f;
jfieldID getY3f;
jfieldID getZ3f;
jmethodID init3f;

class point_transform_affine3d {
public:
    point_transform_affine3d() = default;
    explicit point_transform_affine3d(double _m[3][3]) {
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                m[i][j] = _m[i][j];
    }
    Point3f operator() (const Point3f& p) const {
        Point3f result{};
        result.x = p.x * m[0][0] + p.y * m[0][1] + p.z *m[0][2];
        result.y = p.x * m[1][0] + p.y * m[1][1] + p.z *m[1][2];
        result.z = p.x * m[2][0] + p.y * m[2][1] + p.z *m[2][2];
        return result;
    }
private:
    double m[3][3] = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
    };
};

std::vector<Point3f> model_landmarks(LANDMARK_NUM);
std::vector<Point3f> fitting_landmarks(LANDMARK_NUM);

inline point_transform_affine3d rotate_around_x (double angle)
{
    const double ca = std::cos(angle);
    const double sa = std::sin(angle);
    double m[3][3] = {
            {1, 0,   0 },
            {0, ca, -sa},
            {0, sa,  ca}
    };
    return point_transform_affine3d(m);
}

inline point_transform_affine3d rotate_around_y (double angle)
{
    const double ca = std::cos(angle);
    const double sa = std::sin(angle);
    double m[3][3] = {
            { ca, 0, sa},
            { 0,  1, 0},
            {-sa, 0, ca}
    };
    return point_transform_affine3d(m);
}

inline point_transform_affine3d rotate_around_z (double angle)
{
    const double ca = std::cos(angle);
    const double sa = std::sin(angle);
    double m[3][3] = {
            {ca, -sa, 0},
            {sa,  ca, 0},
            {0,   0,  1}
    };
    return point_transform_affine3d(m);
}

void landmarks_3d_to_2d(std::vector<Point3f>& landmarks_3d, std::vector<Point2d>& landmarks_2d)
{
    landmarks_2d.clear();
    double xs;
    double ys;
    double zs;
    double xo;
    double yo;
    for (auto &iter : landmarks_3d) {
        xs = iter.x;
        ys = iter.y;
        zs = iter.z;
        xo = FX * xs / zs + CX;
        yo = FY * ys / zs + CY;
        landmarks_2d.emplace_back(xo, yo);
    }
}



void rotate(std::vector<Point3f>& points, const double yaw, const double pitch, const double roll)
{
    point_transform_affine3d around_z = rotate_around_z(roll * M_PI / 180);
    point_transform_affine3d around_y = rotate_around_y(yaw * M_PI / 180);
    point_transform_affine3d around_x = rotate_around_x(pitch * M_PI / 180);
    for (auto &point : points)
        point = around_z(around_y(around_x(point)));
}


void translate(std::vector<Point3f>& points, const double x, const double y, const double z)
{
    for (auto &point : points)
        point = point + Point3f(x, y, z);
}

void transform(std::vector<Point3f>& points, const double * const x)
{
    rotate(points, x[0], x[1] ,x[2]);
    translate(points, x[3], x[4], x[5]);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_keith_ceres_1solver_CeresSolver_init_1(JNIEnv *env, jclass type) {
    point2dClass = env->FindClass("android/graphics/Point");
    point3fClass = env->FindClass("com/keith/ceres_solver/Point3f");
    getX2d = env->GetFieldID(point2dClass, "x", "I");
    getY2d = env->GetFieldID(point2dClass, "y", "I");
    init2d = env->GetMethodID(point2dClass, "<init>", "(II)V");
    getX3f = env->GetFieldID(point3fClass, "x", "D");
    getY3f = env->GetFieldID(point3fClass, "y", "D");
    getZ3f = env->GetFieldID(point3fClass, "z", "D");
    init3f = env->GetMethodID(point3fClass, "<init>", "(DDD)V");

    jmethodID getLandmarks = env->GetStaticMethodID(type, "getLandmarks", "(I)Lcom/keith/ceres_solver/Point3f;");
    jobject tmp;
    for(unsigned int i=0; i<LANDMARK_NUM; i++) {
        LOGI("#################### Call %d ##################", i);
        tmp = env->CallStaticObjectMethod(type, getLandmarks, i);
        if(tmp== nullptr) {
            LOGI("#################### Stop in %d ##################", i);
        }
        model_landmarks[i] = Point3f(env->GetDoubleField(tmp, getX3f),
                                     env->GetDoubleField(tmp, getY3f),
                                     env->GetDoubleField(tmp, getZ3f));
    }
}

struct CostFunctor {
public:
    explicit CostFunctor(JNIEnv *_env, jobjectArray _shape){
        env = _env;
        shape = _shape; }
    bool operator()(const double* const x, double* residual) const {
        /* Init landmarks to be transformed */
        fitting_landmarks.clear();
        for (auto &model_landmark : model_landmarks)
            fitting_landmarks.push_back(model_landmark);
        transform(fitting_landmarks, x);
        std::vector<Point2d> model_landmarks_2d;
        landmarks_3d_to_2d(fitting_landmarks, model_landmarks_2d);

        /* Calculate the energe (Euclid distance from two points) */
        for(unsigned long i=0; i<LANDMARK_NUM; i++) {
            jobject point = env->GetObjectArrayElement(shape, static_cast<jsize>(i));
            long tmp1 = env->GetIntField(point, getX2d) - model_landmarks_2d.at(i).x;
            long tmp2 = env->GetIntField(point, getY2d) - model_landmarks_2d.at(i).y;
            residual[i] = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
        }
        return true;
    }
private:
    JNIEnv *env;
    jobjectArray shape;	/* 3d landmarks coordinates got from dlib */
};


extern "C"
JNIEXPORT void JNICALL
Java_com_keith_ceres_1solver_CeresSolver_solve(JNIEnv *env, jclass type, jdoubleArray x_,
                                       jobjectArray landmarks) {
    jdouble *x = env->GetDoubleArrayElements(x_, nullptr);
    Problem problem;
    CostFunction *cost_function =
            new NumericDiffCostFunction<CostFunctor, ceres::RIDDERS, LANDMARK_NUM, 6>(
                    new CostFunctor(env, landmarks));
    problem.AddResidualBlock(cost_function, nullptr, x);
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    env->ReleaseDoubleArrayElements(x_, x, 0);
}


extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_keith_ceres_1solver_CeresSolver_transform(JNIEnv *env, jclass type,
                                                   jdoubleArray x_) {
    jdouble *x = env->GetDoubleArrayElements(x_, nullptr);
    std::vector<Point3f> points(LANDMARK_NUM);
    for(int i=0; i<LANDMARK_NUM; i++) {
        points[i] = model_landmarks[i];
    }
    transform(points, x);
    jobjectArray results = env->NewObjectArray(LANDMARK_NUM, point3fClass, nullptr);
    for(int i=0; i<LANDMARK_NUM; i++) {
        jobject object = env->NewObject(point3fClass, init3f, points[i].x, points[i].y, points[i].z);
        env->SetObjectArrayElement(results, i, object);
    }
    env->ReleaseDoubleArrayElements(x_, x, 0);
    return results;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_keith_ceres_1solver_CeresSolver_transformTo2d(JNIEnv *env, jclass type,
                                                       jobjectArray points_) {
    std::vector<Point3f> points(LANDMARK_NUM);
    for(int i=0; i<LANDMARK_NUM; i++) {
        jobject point = env->GetObjectArrayElement(points_, i);
        points[i] = Point3f(env->GetDoubleField(point, getX3f),
                            env->GetDoubleField(point, getY3f),
                            env->GetDoubleField(point, getZ3f));
    }
    std::vector<Point2d> points2d(LANDMARK_NUM);
    landmarks_3d_to_2d(points, points2d);
    jobjectArray results = env->NewObjectArray(LANDMARK_NUM, point2dClass, nullptr);
    for(int i=0; i<LANDMARK_NUM; i++) {
        jobject object = env->NewObject(point2dClass, init2d, points[i].x, points[i].y);
        env->SetObjectArrayElement(results, i, object);
    }
    return results;
}