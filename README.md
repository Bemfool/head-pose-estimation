# Head Pose Estimation & Tracking - C++

Estimate (from one image) and track (from video stream) head pose using C++.

Fit 2d landmarks got from Dlib and 3d landmarks got from Basel face model.



## Dependence

* Ceres Solver;
* Dlib (OpenCV)
* [BFM-tools](https://github.com/Great-Keith/BFM-tools) 



## Content

| File/Directory Name    | Usage                                                        |
| ---------------------- | ------------------------------------------------------------ |
| hpe_oneshot.cpp        | Source file of head pose estimation from one image, only depending on Dlib. |
| hpe_oneshot_openGL.cpp | Source file of head pose estimation from one image, using OpenGL framework. |
| hpe_webcam.cpp         | Source file of head pose tracking from video stream, only depending on Dlib. Use multi-step optimization strategy. |
| hpe_webcam_openGL.cpp  | Source file of head pose tracking from video stream, using OpenGL framework. Use multi-step optimization strategy. |
| time_recorder.cpp      | Source file of counting cost time of different implementations. |
| hmr_gen_data.cpp       | Source file of generating synthetic data for head motion recognition. |
| batch_hpe_videos.sh    | Bash script for batching process videos.                     |
| example_inputs         | Contains some examples inputs. Format see also https://github.com/Great-Keith/BFM-tools |
| data                   | Contains `shape_predictor_68_face_landmarks.dat`, landmark index file (*.anl) and Basel face model (\*.h5). Download by yourself. |
| include                | Contains header files of head pose estimation.               |
| lib                    | Contains third-party library (BFM-tools).                    |
| shader                 | Contains vertex shader and fragment shader used by OpenGL framework. |



## Landmark Selection

There are 5 different landmark selections, 6, 12, 21, 33, and 68. They are selected from 68 landmarks by myself. Different numbers of landmark would make different results.

![fp](.\assets\fp.jpg)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Pic 1. (a) 68 landmarks and (b) 6 landmarks selections </center> 

![fp2](.\assets\fp2.jpg)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Pic 2. (a) 6 (b) 12 (c) 21 (d) 33 (e) 68 landmarks make different affects to face reconstruction and head pose </center> 

[This learning blog](<https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/>) which uses Dlib to head pose estimation is popular, and it choose 6 landmarks just like Pic 2. (a) shows. But as the result shows, it's not good enough to estimate head pose. We could see obviously mismatching with 68 landmark (which is assumed as perfect solution).



## Result

![Animation2](.\assets\Animation2.gif)



## Make

Replace library path in CMakeLists.txt with your local path. (Sorry for my poor knowledge about CMake to set custom config :/ ) and do a serial of general cmake instructions to compile.

*[NOTE] Make sue that you are using`Release`option，and set`USE_AVX_INSTRUCTIONS`and`USE_SSE2_INSTRUCTIONS`/`USE_SSE4_INSTRUCTIONS`on，otherwise, detection supported by Dlib would take too much time.



## Others

I also use the same algorithm to implement an Android app in <https://github.com/Great-Keith/FaceTracker>, If interested, plz check it.



