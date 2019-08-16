# 头部姿态估计 - C++

使用 [dlib](<https://github.com/davisking/dlib>) 和 [Ceres](<https://github.com/ceres-solver/ceres-solver>) 进行人脸特征点拟合，从而得到头部姿态的各个参数（**yaw**, **pitch**, **roll**, tx, ty, tz）。详细原理解释见博客：[头部姿态估计 - OpenCV/Dlib/Ceres](https://www.cnblogs.com/bemfoo/p/11253450.html)

## 内容

| 文件名                             | 内容                                                         | 照片来源 | 求解器  | 差分方式 | 是否可用 |
| ---------------------------------- | ------------------------------------------------------------ | -------- | ------- | -------- | -------- |
| hpe-cam-ceres-analyticdiff.cpp     | 使用Ceres中的自动差分和dlib，用电脑自带摄像头进行实时拟合。  | 摄像头   | ceres   | 分析差分 | 否       |
| hpe-cam-ceres-numericdiff.cpp      | 使用Ceres中的数值差分和dlib，读取单张照片进行拟合。          | 摄像头   | ceres   | 数值差分 | 是       |
| hpe-oneshot-ceres-analyticdiff.cpp | 使用Ceres中的自动差分和dlib，读取单张照片进行拟合            | 本地图片 | ceres   | 分析差分 | 否       |
| hpe-oneshot-ceres-numericdiff.cpp  | 使用Ceres中的数值差分和dlib，读取单张照片进行拟合            | 本地图片 | ceres   | 数值差分 | 是       |
| hpe-oneshot-cminpack.cpp           | 使用 [Minpack](<https://github.com/devernay/cminpack>) 和dlib，读取单张照片进行拟合（存在问题）。 |          | minpack | 数值差分 | 否       |
| landmarks.txt                      | BFM标准人脸的68个特征点的三维坐标，获取方式和格式见：[BFM使用 - 获取平均脸模型的68个特征点坐标](https://www.cnblogs.com/bemfoo/p/11215643.html) |      |

*[NOTE] landmark-fitting-cam.cpp中有详细注释。*



## 测试

![only_face](https://github.com/Great-Keith/head-pose-estimation/raw/master/cpp/assets/only_face.gif)

![work_place](https://github.com/Great-Keith/head-pose-estimation/raw/master/cpp/assets/work_place.gif)


## 编译

修改CMakeLists.txt中的各个依赖包路径，使用CMake进行编译。

*[NOTE] 需要使用`Release`版本，以及增加选项`USE_AVX_INSTRUCTIONS`和`USE_SSE2_INSTRUCTIONS`/`USE_SSE4_INSTRUCTIONS`，否则因为dlib的检测耗时较长，使用`landmark-fitting-cam`会有卡顿。*
