package opencv4unity.camera1gltest1;

import android.Manifest;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Handler;
import android.os.Message;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity 
{
    private RelativeLayout  previewLayout;
    private MyGLSurfaceView myGLSurfaceView;
    private TextView        textView;


    ///@Override是伪代码,表示重写。(当然不写@Override也可以)，不过写上有如下好处:
    //1、可以当注释用,方便阅读；
    //2、编译器可以给你验证@Override下面的方法名是否是你父类中所有的，如果没有则报错。
    // 例如，你如果没写@Override，而你下面的方法名又写错了，这时你的编译器是可以编译通过的，因为编译器以为这个方法是你的子类中自己增加的方法。
    @Override
    protected void onCreate(Bundle savedInstanceState) 
    {
        ///Android中Bundle类的作用
        //Bundle类用作携带数据，它类似于Map，用于存放key-value名值对形式的值
        super.onCreate(savedInstanceState);

//        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
//                             WindowManager.LayoutParams.FLAG_FULLSCREEN);
//
//        requestWindowFeature(Window.FEATURE_NO_TITLE);

///改变屏幕方向
//        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
//        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        setContentView(R.layout.activity_main);

        requestPermission();

        previewLayout = (RelativeLayout)findViewById(R.id.previewLayout);

        myGLSurfaceView = new MyGLSurfaceView(this, new MyGLSurfaceView.GetUiHandlerInterface() 
        {
            @Override
            public void getUiHandler(Message leftUpPt) {
                mHandler.sendMessage(leftUpPt);
            }
        } );

        previewLayout.addView(myGLSurfaceView);

        textView = (TextView)findViewById(R.id.sample_text);

    }
///处理异步信息,自动调用
    private Handler mHandler = new Handler()
    {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            switch (msg.what){
                case 0:
                    textView.setText("LeftUp Point is "+msg.arg1+" , "+msg.arg2);
                    break;
                default:
                    break;
            }
        }
    };

    void requestPermission()
    {
        final int REQUEST_CODE = 1;
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                            Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_CODE);
        }
    }

    public void clickText(View view) 
    {
        if (myGLSurfaceView.model == 2)
            myGLSurfaceView.model = 0;
        else myGLSurfaceView.model++;
        Toast.makeText(this,"Change Effect",Toast.LENGTH_SHORT).show();
    }
}
