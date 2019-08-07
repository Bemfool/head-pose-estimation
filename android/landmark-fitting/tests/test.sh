#/bin/bash
echo "$(adb shell ls /sdcard/ | egrep -i shape_predictor_68_face_landmarks)"

if (adb shell ls /sdcard/ | egrep -i shape_predictor_68_face_landmarks) 2> /dev/null; then
    echo shape_predictor_68_face_landmarks exists
else
    echo shape_predictor_68_face_landmarks does not exist
    adb push ../app/src/main/res/raw/shape_predictor_68_face_landmarks.dat /sdcard/
fi

adb push test.bmp /sdcard/
cd ..
./gradlew assembleDebugAndroidTest
./gradlew connectedDebugAndroidTest
