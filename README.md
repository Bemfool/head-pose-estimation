# 头部姿态估计

使用dlib和Ceres进行人脸特征点拟合，从而得到头部姿态的各个参数。

## 内容

landmark-fiitting-cam：使用Ceres和dlib，用电脑自带摄像头进行实时拟合。

landmark-fitting-ceres：使用Ceres和dlib，读取单张照片进行拟合。

landmark-fitting-cminpack：使用Minpack和dlib，读取单张照片进行拟合（存在问题）。

*[NOTE] landmark-fitting-cam.cpp中有详细注释。*

## 测试

![only_face](C:\Users\Keith Lin\Desktop\head-pose-estimation\assets\only_face.gif)

![work_place](C:\Users\Keith Lin\Desktop\head-pose-estimation\assets\work_place.gif)


## 编译

使用CMake进行编译，需要使用`Release`版本，以及增加选项`USE_AVX_INSTRUCTIONS`和`USE_SSE2_INSTRUCTIONS`/`USE_SSE4_INSTRUCTIONS`，否则因为dlib的检测耗时较长，使用`landmark-fitting-cam`会有卡顿。