package opencv4unity.camera1gltest1;

import android.app.Application;

public class MyApplication extends Application {
    public static MyApplication instance;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
    }

    public MyApplication getApplicationContext() {
        return instance;
    }
}
