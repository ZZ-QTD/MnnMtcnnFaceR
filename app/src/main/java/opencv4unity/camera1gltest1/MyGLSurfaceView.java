package opencv4unity.camera1gltest1;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceHolder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

///extends 是继承某个类, 继承之后可以使用父类的方法, 也可以重写父类的方法; implements 是实现多个接口, 接口的方法一般为空的, 必须重写才能使用
public class MyGLSurfaceView extends GLSurfaceView implements SurfaceHolder.Callback, Camera.PreviewCallback
{
    //OpenGLES相关 , 与c语言一样，必须要先定义声明
    private int srcFrameWidth  = 640;
    private int srcFrameHeight = 480;
    private int srcDensity     = 3;

    // 纹理id
    private boolean mbpaly             = false;
    private FloatBuffer squareVertices = null;
    private FloatBuffer coordVertices  = null;
    private Bitmap grayImg             = null;

    private static float squareVertices_[] = {
            -1.0f,  -1.0f, 0.0f,
             1.0f,  -1.0f, 0.0f,
            -1.0f,   1.0f, 0.0f,
             1.0f,   1.0f, 0.0f,
    };

    private static float coordVertices_[] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f,  1.0f,
            1.0f,  1.0f,
    };
    //Camera
    private int mCameraIndex = Camera.CameraInfo.CAMERA_FACING_BACK;  ///使用前摄像头，（如果只有一个摄像头）
    private Camera camera;
//    public  Bitmap resultImg;
    private SurfaceHolder surfaceHolder;
    private boolean printCharacters = true;
    private static final String TAG = "MyGLSurfaceView";
    private SurfaceTexture surfaceTexture;
    private Activity activity;
    public int model = 0;

    public MyGLSurfaceView(Context context, GetUiHandlerInterface mainUiHandler)
    {
        super(context);
        // Log.i("DEBUG", "hehehe"); majin
        getUiHandlerInterface = mainUiHandler;
        DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
        srcFrameWidth  = displayMetrics.widthPixels;
        srcFrameHeight = displayMetrics.heightPixels;
        srcDensity     = (int)displayMetrics.density;

        //设置Renderer到GLSurfaceView
        setRenderer(new MyGL20Renderer());

        // 只有在绘制数据改变时才绘制view
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        // 顶点坐标
        squareVertices = floatBufferUtil(squareVertices_);

        //纹理坐标
        coordVertices = floatBufferUtil(coordVertices_);

        surfaceTexture = new SurfaceTexture(2);
        surfaceHolder  = getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    // 定义一个工具方法，将float[]数组转换为OpenGL ES所需的FloatBuffer
    private FloatBuffer floatBufferUtil(float[] arr)
    {
        FloatBuffer mBuffer;

        // 初始化ByteBuffer，长度为arr数组的长度*4，因为一个int占4个字节
        ByteBuffer qbb = ByteBuffer.allocateDirect(arr.length * 4);

        // 数组排列用nativeOrder
        qbb.order(ByteOrder.nativeOrder());
        mBuffer = qbb.asFloatBuffer();
        mBuffer.put(arr);
        mBuffer.position(0);
        return mBuffer;
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) 
    {
        super.surfaceCreated(holder);
        startCamera(mCameraIndex);
        printCharacters = false;
    }

    private void startCamera(int mCameraIndex)
    {
        // 初始化并打开摄像头
        if (camera == null)
        {
            try
            {
                /// mCameraIndex = Camera.CameraInfo.CAMERA_FACING_BACK
                camera = Camera.open(mCameraIndex); ///打开摄像头，mCameraIndex
            }
            catch (Exception e)
            {
                return;
            }
            Camera.Parameters params = camera.getParameters();  ////参考SDK中的API，获取相机的参数：
            if (printCharacters)
            {
                List<Camera.Size> previewSizes = params.getSupportedPreviewSizes();  //获取预览的各种分辨率
                for(int i = 0; i < previewSizes.size(); i++)
                {
                    if ( 1280 / previewSizes.get(i).width == 720 / previewSizes.get(i).height)
                    {
                        Log.i(TAG, "SupportedGlassesPreviewSizes: "+
                              String.valueOf(previewSizes.get(i).width)+", "+
                              String.valueOf(previewSizes.get(i).height));
                    }
                }
            }
            // .contains 判断字符串中是否包含指定的字符或字符串
            if (params.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE))
            {
                // 自动对焦
                params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
            }

            List<int[]> previewFpsRange = params.getSupportedPreviewFpsRange();
            int[] setFps = {10000, 15000};
            for (int i = 0; i < previewFpsRange.size(); i++)
            {
                if (previewFpsRange.get(i)[1] == setFps[1])
                {
                    setFps[0]=previewFpsRange.get(i)[0];
                }
            }
            //注意setPreviewFpsRange的输入参数是帧率*1000，如30帧/s则参数为30*1000
            params.setPreviewFpsRange(setFps[0], setFps[1]);//15,30

            try
            {
                //眼镜的尺寸为1280x720
                srcFrameWidth  = 640;
                srcFrameHeight = 480;
                params.setPreviewFormat(ImageFormat.NV21);
                params.setPreviewSize(srcFrameWidth, srcFrameHeight);
                camera.setParameters(params);
            }
            catch (Exception e)
            { }

            try
            {
                camera.setDisplayOrientation(90);
                camera.setPreviewTexture(surfaceTexture);
                camera.startPreview();
                camera.setPreviewCallback(this);

            }
            catch(Exception ex)
            {
                ex.printStackTrace();
                camera.release();
                camera=null;
            }
        }
    }

    private void stopCamera()
    {
        if (camera != null){
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera=null;
            printCharacters=true;
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        super.surfaceChanged(holder, format, w, h);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        super.surfaceDestroyed(holder);
        stopCamera();
    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) 
    {
        synchronized (this)
        {
            // 开启后台进程，也可通过AsyncTask
            // 通过Callback，将前一帧图像通过，byte[] data 形式传递出去
            int width = camera.getParameters().getPreviewSize().width;
            int height = camera.getParameters().getPreviewSize().height;
            onSaveFrames(bytes,bytes.length,width,height);
        }
    }

    public class MyGL20Renderer implements Renderer
    {
        private int texture;

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config)
        {
            mbpaly = false;
            // 关闭抗抖动
            gl.glDisable(GL10.GL_DITHER);
            //设置背景的颜色
            gl.glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
            //启动纹理
            gl.glEnable(GL10.GL_TEXTURE_2D);

            loadTexture(gl);
            mbpaly = true;
        }
        private void loadTexture(GL10 gl)
        {
            try
            {
                // 加载位图
                int[] textures = new int[1];
                // 指定生成N个纹理（第一个参数指定生成一个纹理）
                // textures数组将负责存储所有纹理的代号
                gl.glGenTextures(1, textures, 0);
                // 获取textures纹理数组中的第一个纹理
                texture = textures[0];
                // 通知OpenGL将texture纹理绑定到GL10.GL_TEXTURE_2D目标中
                gl.glBindTexture(GL10.GL_TEXTURE_2D, texture);
                // 设置纹理被缩小（距离视点很远时被缩小）时的滤波方式
                gl.glTexParameterf(GL10.GL_TEXTURE_2D,
                        GL10.GL_TEXTURE_MIN_FILTER, GL10.GL_NEAREST);
                // 设置纹理被放大（距离视点很近时被方法）时的滤波方式
                gl.glTexParameterf(GL10.GL_TEXTURE_2D,
                        GL10.GL_TEXTURE_MAG_FILTER, GL10.GL_LINEAR);
                // 设置在横向、纵向上都是平铺纹理
                gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_WRAP_S,
                        GL10.GL_REPEAT);
                gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_WRAP_T,
                        GL10.GL_REPEAT);
                // 加载位图生成纹理
                if (grayImg!=null)
                    GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, grayImg, 0);
            }
            finally
            {
                // 生成纹理之后，回收位图
                if (grayImg != null)
                    grayImg.recycle();
            }
        }

        public void onDrawFrame(GL10 gl)
        {
            // 清除屏幕缓存和深度缓存
            gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);
            // 启用顶点坐标数据
            gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
            // 启用贴图坐标数组数据
            gl.glEnableClientState(GL10.GL_TEXTURE_COORD_ARRAY);
            gl.glLoadIdentity();
        // 旋转视频画面图片
            gl.glRotatef(00.0f, 0, 0,1);
//            gl.glRotatef(270.0f, 0, 1,0);
            gl.glRotatef(180.0f, 1, 0,0);
            //绑定贴图
            if (grayImg != null){
                gl.glBindTexture(GL10.GL_TEXTURE_2D, texture);
                GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, grayImg, 0);
                // 设置贴图的坐标数据
                gl.glTexCoordPointer(2, GL10.GL_FLOAT, 0, coordVertices);  // ②
                // 执行纹理贴图
                gl.glBindTexture(GL10.GL_TEXTURE_2D, texture);
            }

            // 设置顶点的位置数据
            gl.glVertexPointer(3, GL10.GL_FLOAT, 0, squareVertices);
            // 按cubeFacetsBuffer指定的面绘制三角形
//            gl.glDrawElements(GL10.GL_TRIANGLES, cubeFacetsBuffer.remaining(),
//                    GL10.GL_UNSIGNED_BYTE, cubeFacetsBuffer);
            gl.glDrawArrays(GL10.GL_TRIANGLE_STRIP,0,4);
            // 绘制结束
            gl.glFinish();
            // 禁用顶点、纹理坐标数组
            gl.glDisableClientState(GL10.GL_VERTEX_ARRAY);
            gl.glDisableClientState(GL10.GL_TEXTURE_COORD_ARRAY);

            if (grayImg != null)
                grayImg.recycle();
        }
        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height)
        {
            // 初始化单位矩阵
            gl.glLoadIdentity();
            // 计算透视视窗的宽度、高度比
            float ratio = (float) width / height;
            // 调用此方法设置透视视窗的空间大小。
            gl.glFrustumf(-ratio, ratio, -1, 1, 1, 10);
        }
    }

    private GetUiHandlerInterface getUiHandlerInterface;
    public interface GetUiHandlerInterface
    {
        void getUiHandler(Message leftUpPt);
    }

    public void onSaveFrames(byte[] data, int length, int srcWidth, int srcHeight)//读取图像数据
    {
        if ( length != 0 && mbpaly )
        {
            final MyNDKOpencv myNDKOpencv=new MyNDKOpencv();
            //OpenCV Processing
            grayImg = myNDKOpencv.scanfEffect(data,srcFrameWidth,srcFrameHeight,model);
            requestRender();

            Message leftUpPtMsg = Message.obtain();
            leftUpPtMsg.what    = 0;
            leftUpPtMsg.arg1    = myNDKOpencv.leftUpPt_x;
            leftUpPtMsg.arg2    = myNDKOpencv.leftUpPt_y;
            getUiHandlerInterface.getUiHandler(leftUpPtMsg);

        }
    }
}
