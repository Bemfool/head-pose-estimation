/*
*  Copyright (C) 2015-present TzuTaLin
*/

package com.tzutalin.dlibtest;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Toast;

import com.dexafree.materialList.card.Card;
import com.dexafree.materialList.card.provider.BigImageCardProvider;
import com.dexafree.materialList.view.MaterialListView;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.PedestrianDet;
import com.tzutalin.dlib.VisionDetRet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import hugo.weaving.DebugLog;
import timber.log.Timber;

@RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN)
public class MainActivity extends AppCompatActivity {
    private static final int RESULT_LOAD_IMG = 1;
    private static final int REQUEST_CODE_PERMISSION = 2;

    private static final String TAG = "MainActivity";

    // Storage Permissions
    private static String[] PERMISSIONS_REQ = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA
    };

    // UI
    private ProgressDialog mDialog;
    private MaterialListView mListView;
    private FloatingActionButton mFabActionBt;
    private FloatingActionButton mFabCamActionBt;
    private Toolbar mToolbar;

    private String mTestImgPath;
    private FaceDet mFaceDet;
    private PedestrianDet mPersonDet;
    private List<Card> mCard = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mListView = (MaterialListView) findViewById(R.id.material_listview);
        setSupportActionBar(mToolbar);
        // Just use hugo to print log
        isExternalStorageWritable();
        isExternalStorageReadable();

        // Android 6.0-采用静态权限, 安装的时候询问所有权限, 若拒绝则无法安装;
        // Android 6.0+采用动态权限, 安装默认全部权限为禁止.

        // For API 23+ you need to request the read/write permissions even if they are already in your manifest.
        int currentapiVersion = android.os.Build.VERSION.SDK_INT;

        if (currentapiVersion >= Build.VERSION_CODES.M) {
            verifyPermissions(this);
        }

        setupUI();
    }

    protected void setupUI() {
        mListView = (MaterialListView) findViewById(R.id.material_listview);
        // 右下角打开照片的按钮
        mFabActionBt = (FloatingActionButton) findViewById(R.id.fab);
        // 左下角打开相机的按钮
        mFabCamActionBt = (FloatingActionButton) findViewById(R.id.fab_cam);
        // 最上方的一栏
        mToolbar = (Toolbar) findViewById(R.id.toolbar);

        mFabActionBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // launch Gallery
                // Toast.makeText: 正下方弹出的浮圆角灰色文本框
                // Toast.LENGTH_SHORT: 2s; Toast.LENGTH_LONG: 3.5s
                Toast.makeText(MainActivity.this, "Pick one image", Toast.LENGTH_SHORT).show();
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, RESULT_LOAD_IMG);
                // 照片选中了之后, 会调用onActivityResult()方法
            }
        });

        mFabCamActionBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Activity跳转
                startActivity(new Intent(MainActivity.this, CameraActivity.class));
            }
        });

        mToolbar.setTitle(getString(R.string.app_name));
        Toast.makeText(MainActivity.this, getString(R.string.description_info), Toast.LENGTH_LONG).show();
    }

    /**
     * Checks if the app has permission to write to device storage or open camera
     * If the app does not has permission then the user will be prompted to grant permissions
     *
     * @param activity
     */
    @DebugLog
    private static boolean verifyPermissions(Activity activity) {
        // ActivityCompat.checkSelfPermission: 判断是否被授予了指定的权限.
        // Check if we have write permission
        int write_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int read_persmission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int camera_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.CAMERA);

        if (write_permission != PackageManager.PERMISSION_GRANTED ||
                read_persmission != PackageManager.PERMISSION_GRANTED ||
                camera_permission != PackageManager.PERMISSION_GRANTED) {
            // ActivityCompat.requestPermissions: 弹出窗口授予权限(Android 6.0+)
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_REQ,
                    REQUEST_CODE_PERMISSION
            );
            return false;
        } else {
            return true;
        }
    }

    /* Checks if external storage is available for read and write */
    @DebugLog
    private boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }

    /* Checks if external storage is available to at least read */
    @DebugLog
    private boolean isExternalStorageReadable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state) ||
                Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
            return true;
        }
        return false;
    }

    @DebugLog
    protected void demoStaticImage() {
        if (mTestImgPath != null) {
            Timber.tag(TAG).d("demoStaticImage() launch a task to det");
            runDemosAsync(mTestImgPath);
        } else {
            Timber.tag(TAG).d("demoStaticImage() mTestImgPath is null, go to gallery");
            Toast.makeText(MainActivity.this, "Pick an image to run algorithms", Toast.LENGTH_SHORT).show();
            // Create intent to Open Image applications like Gallery, Google Photos
            Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent, RESULT_LOAD_IMG);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CODE_PERMISSION) {
            Toast.makeText(MainActivity.this, "Demo using static images", Toast.LENGTH_SHORT).show();
            demoStaticImage();
        }
    }

    // 当选中了照片之后, 触发该方法(如果没有选中照片直接退出也会触发)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        try {
            // 选中了一张照片
            // When an Image is picked
            if (requestCode == RESULT_LOAD_IMG && resultCode == RESULT_OK && null != data) {
                // Get the Image from data
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                // Get the cursor
                Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                // Cursor初始位置指向第一条记录的前一个位置;
                // Cursor.moveToFirst(): 指向第一个位置
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                mTestImgPath = cursor.getString(columnIndex);
                cursor.close();
                if (mTestImgPath != null) {
                    // 开始处理图片
                    runDemosAsync(mTestImgPath);
                    Toast.makeText(this, "Img Path:" + mTestImgPath, Toast.LENGTH_SHORT).show();
                }
            } else {
                // 如果没有选择照片直接退出
                Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
        }
    }

    // ==========================================================
    // Tasks inner class
    // ==========================================================

    // 处理照片
//    @NonNull
    private void runDemosAsync(@NonNull final String imgPath) {
        // 处理照片得到人的框选并画到照片上
        demoPersonDet(imgPath);
        // 处理照片得到人脸的框选和landmark并画到照片上
        demoFaceDet(imgPath);
    }

    // 处理人体信息
    @SuppressLint("StaticFieldLeak")
    private void demoPersonDet(final String imgPath) {
        // VisionDetRet: 包含框选
        new AsyncTask<Void, Void, List<VisionDetRet>>() {
            // 预处理: 无
            @Override
            protected void onPreExecute() {
                super.onPreExecute();
            }

            // 后处理: 将得到的框选和landmarks画上去并显示
            @Override
            protected void onPostExecute(List<VisionDetRet> personList) {
                super.onPostExecute(personList);
                if (personList.size() > 0) {
                    // Card: MaterialList的单位
                    Card card = new Card.Builder(MainActivity.this)
                            .withProvider(BigImageCardProvider.class)
                            .setDrawable(drawRect(imgPath, personList, Color.BLUE)) // 画出框选
                            .setTitle("Person det")
                            .endConfig()
                            .build();
                    mCard.add(card);
                } else {
                    Toast.makeText(getApplicationContext(), "No person", Toast.LENGTH_LONG).show();
                }
                updateCardListView();
            }

            // 处理中
            @Override
            protected List<VisionDetRet> doInBackground(Void... voids) {
                // 初次使用mPersonDet, 进行初始化(有点像单例模式)
                // Init
                if (mPersonDet == null) {
                    mPersonDet = new PedestrianDet();
                }

                // Timer: 一个懒人logger
                Timber.tag(TAG).d("Image path: " + imgPath);

                // 检测人体
                List<VisionDetRet> personList = mPersonDet.detect(imgPath);
                return personList;
            }
        }.execute();
    }

    @SuppressLint("StaticFieldLeak")
    private void demoFaceDet(final String imgPath) {
        new AsyncTask<Void, Void, List<VisionDetRet>>() {
            // 预处理: 打印信息"检测人脸中.."
            @Override
            protected void onPreExecute() {
                super.onPreExecute();
                showDiaglog("Detecting faces");
            }

            @Override
            protected void onPostExecute(List<VisionDetRet> faceList) {
                super.onPostExecute(faceList);
                if (faceList.size() > 0) {
                    Card card = new Card.Builder(MainActivity.this)
                            .withProvider(BigImageCardProvider.class)
                            .setDrawable(drawRect(imgPath, faceList, Color.GREEN))
                            .setTitle("Face det")
                            .endConfig()
                            .build();
                    mCard.add(card);
                } else {
                    Toast.makeText(getApplicationContext(), "No face", Toast.LENGTH_LONG).show();
                }
                updateCardListView();
                dismissDialog();
            }

            @Override
            protected List<VisionDetRet> doInBackground(Void... voids) {
                // Init
                if (mFaceDet == null) {
                    mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
                }

                // 预训练好的landmark神经网络模型
                final String targetPath = Constants.getFaceShapeModelPath();
                // 将模型放到本地
                if (!new File(targetPath).exists()) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Copy landmark model to " + targetPath, Toast.LENGTH_SHORT).show();
                        }
                    });
                    FileUtils.copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_68_face_landmarks, targetPath);
                }

                List<VisionDetRet> faceList = mFaceDet.detect(imgPath);
                return faceList;
            }
        }.execute();
    }

    private void updateCardListView() {
        // mListView关联了MaterialList
        mListView.clearAll();
        for (Card each : mCard) {
            mListView.add(each);
        }
    }

    private void showDiaglog(String title) {
        dismissDialog();
        // 显示一个处理框, 标题title, 内容message
        mDialog = ProgressDialog.show(MainActivity.this, title, "process..", true);
    }

    private void dismissDialog() {
        if (mDialog != null) {
            mDialog.dismiss();
            mDialog = null;
        }
    }

    // 在path所指向的图片上画出results中所有的框选和landmarks
    @DebugLog
    private BitmapDrawable drawRect(String path, List<VisionDetRet> results, int color) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        Bitmap bm = BitmapFactory.decodeFile(path, options);
        android.graphics.Bitmap.Config bitmapConfig = bm.getConfig();
        // set default bitmap config if none
        if (bitmapConfig == null) {
            bitmapConfig = android.graphics.Bitmap.Config.ARGB_8888;
        }
        // resource bitmaps are imutable,
        // so we need to convert it to mutable one
        bm = bm.copy(bitmapConfig, true);
        int width = bm.getWidth();
        int height = bm.getHeight();
        // By ratio scale
        float aspectRatio = bm.getWidth() / (float) bm.getHeight();

        final int MAX_SIZE = 512;
        int newWidth = MAX_SIZE;
        int newHeight = MAX_SIZE;
        float resizeRatio = 1;
        newHeight = Math.round(newWidth / aspectRatio);
        if (bm.getWidth() > MAX_SIZE && bm.getHeight() > MAX_SIZE) {
            Timber.tag(TAG).d("Resize Bitmap");
            bm = getResizedBitmap(bm, newWidth, newHeight);
            resizeRatio = (float) bm.getWidth() / (float) width;
            Timber.tag(TAG).d("resizeRatio " + resizeRatio);
        }

        // Create canvas to draw
        Canvas canvas = new Canvas(bm);
        Paint paint = new Paint();
        paint.setColor(color);
        paint.setStrokeWidth(2);
        paint.setStyle(Paint.Style.STROKE);
        // Loop result list
        for (VisionDetRet ret : results) {
            Rect bounds = new Rect();
            bounds.left = (int) (ret.getLeft() * resizeRatio);
            bounds.top = (int) (ret.getTop() * resizeRatio);
            bounds.right = (int) (ret.getRight() * resizeRatio);
            bounds.bottom = (int) (ret.getBottom() * resizeRatio);
            canvas.drawRect(bounds, paint);
            // Get landmark
            ArrayList<Point> landmarks = ret.getFaceLandmarks();
            for (Point point : landmarks) {
                int pointX = (int) (point.x * resizeRatio);
                int pointY = (int) (point.y * resizeRatio);
                canvas.drawCircle(pointX, pointY, 2, paint);
            }
        }

        return new BitmapDrawable(getResources(), bm);
    }

    @DebugLog
    private Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bm, newWidth, newHeight, true);
        return resizedBitmap;
    }
}
