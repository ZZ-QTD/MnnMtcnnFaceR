package opencv4unity.camera1gltest1;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;


public class MyNDKOpencv {
    private Bitmap resultImg;
    public int leftUpPt_x = 0, leftUpPt_y = 0;

    MyNDKOpencv() {
    }

    public Bitmap scanfEffect(byte[] data, int srcWidth, int srcHeight, int model) {
        BitmapFactory.Options newOpts = new BitmapFactory.Options();
        newOpts.inJustDecodeBounds = true;
        Log.i("openvc","模型下载");
        YuvImage yuvimage = new YuvImage(
                data,
                ImageFormat.NV21, srcWidth, srcHeight, null);
        Log.i("openvc","图片获取");
        //下面的构造方法创建一个32字节（默认大小）的缓冲区。
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Log.i("openvc","1231");
        yuvimage.compressToJpeg(new Rect(0, 0, srcWidth, srcHeight), 100, baos);// 80--JPG图片的质量[0-100],100最高
        Log.i("openvc","13212312");
        byte[] rawImage = baos.toByteArray();
        //将rawImage转换成bitmap
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length, options);
        int tmp_width, tmp_height;
        tmp_width = bitmap.getWidth();
        tmp_height = bitmap.getHeight();
        int[] pixels = new int[tmp_width * tmp_height];
        bitmap.getPixels(pixels, 0, tmp_width, 0, 0, tmp_width, tmp_height);
 //########################################################################################################################################################


        //=================================================================================================================================================
//        Stng fileDir = sdcard.getAbsolutePath() + "/1/";
        String filepath = Environment.getExternalStorageDirectory() + "/MnnModel";
        File sdcard =new File(Environment.getExternalStorageDirectory()+"/MnnModel","det1.mnn") ;
        Log.i("openvc",filepath);
        Log.i("openvc",sdcard.exists()+"--path--"+sdcard.getAbsolutePath());
        //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//        InputStream abpath = getClass().getResourceAsStream("/assets/文件名");
//        String path = new String(InputStreamToByte(abpath ));
//        String state = Environment.getExternalStorageState();  //获取内部存储状态
//        String CameraPic = "/CameraPic";
//        try {
//            File file = new File(filepath + CameraPic + ".jpg");
//            FileOutputStream out = new FileOutputStream(file);
//            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
//            out.flush();
//            out.close();
//            //保存图片后发送广播通知更新数据库
////            Uri uri = Uri.fromFile(file);
////            sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, uri));
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        // String fileDir =  "";
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // Log.i("DEBUG", fileDir);
        ResultFromJni2 resultFromJni2 = getScannerEffect(pixels, filepath, tmp_width, tmp_height, 2);

        int[] resultInt = resultFromJni2.resultInt;
        leftUpPt_x = resultFromJni2.x;
        leftUpPt_y = resultFromJni2.y;
        resultImg = Bitmap.createBitmap(tmp_width, tmp_height, Bitmap.Config.ARGB_8888);
        resultImg.setPixels(resultInt, 0, tmp_width, 0, 0, tmp_width, tmp_height);
        bitmap.recycle();
        bitmap = null;
        return resultImg;
    }

    static {
        System.loadLibrary("native-lib");
    }
    private static native ResultFromJni2 getScannerEffect(int[] pixels, String fileDir, int w, int h, int model);
}
