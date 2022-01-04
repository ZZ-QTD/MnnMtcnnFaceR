#include <regex>
#include <fstream>
#include "imgProcess.h"
#include "mtcnn.h"
#include "FaceRecognize.h"
#include <time.h>
#include <ctime>
#include <unistd.h>


///对比数据相似度
double calculSimilar1(std::vector<float>& v1, std::vector<float>& v2, int distance_metric)
{
    if(v1.size() != v2.size()||!v1.size())
        return 0;
    double ret = 0.0, mod1 = 0.0, mod2 = 0.0, dist = 0.0, diff = 0.0;

    if (distance_metric == 0) {         // Euclidian distance
        for (std::vector<double>::size_type i = 0; i != v1.size(); ++i) {
            diff = v1[i] - v2[i];
            dist += (diff * diff);
        }
        dist = sqrt(dist);
    }
    else {                              // Distance based on cosine similarity
        for (std::vector<double>::size_type i = 0; i != v1.size(); ++i) {
            ret += v1[i] * v2[i];
            mod1 += v1[i] * v1[i];
            mod2 += v2[i] * v2[i];
        }
        dist = ret / (sqrt(mod1) * sqrt(mod2));
    }
    return dist;
}

///这个函数通过txt文件，获得人物以及特征
std::vector<std::string> get_data(){
    std::vector<std::string> all_buf;
    std::ifstream ifs;
    ifs.open("/storage/emulated/0/MnnModel/feature.txt");
    if(!ifs.is_open())
    {
        __android_log_print(ANDROID_LOG_INFO, "imgProcess", "文件打开失败");
        return all_buf;
    }
    std::string buf;
    while(std::getline(ifs, buf)){
        all_buf.push_back(buf);
    }
    ifs.close();
    return all_buf;
}
///分割人物名称
std::string segmentation_name(std::string str){
    std::size_t pos = str.find(" ");
    std::string s1 = str.substr(0,pos);
    return s1;
}
///分割人物特征向量
std::vector<float> segmentation_feature(std::string str){
    std::vector<float> feature;
    std::size_t pos = str.find(" ");
    std::string all_feature = str.substr(pos+1);

    for(int i = 0;i<128;i++){
        std::size_t pos1 = all_feature.find(" ");
        std::string get_feature_str = all_feature.substr(0, pos1);
        float get_feature = atof(get_feature_str.c_str());
        feature.push_back(get_feature);
        all_feature = all_feature.substr(pos1+1);
    }
    return feature;
}

///将路径进行分割，获取图片的人名
std::string SplitFilename(const std::string& str){
    std::size_t found = str.find_last_of("/\\");
//    __android_log_print(ANDROID_LOG_INFO, "图片路径", "path:\t%s", str.substr(0, found).c_str());
    std::string filename_ = str.substr(found+1);
    std::size_t found_ = filename_.find_first_of("_");
    std::string filename = filename_.substr(0, found_);
//    __android_log_print(ANDROID_LOG_INFO, "图片名称", "name:\t%s", filename.c_str());
    return filename;
}

///使用opencv中的glob函数将文件夹下的文件进行遍历，主要是将文件名保存下来
std::vector<std::string> read_img_path(std::string pattern){
    std::vector<std::string> fn;
    cv::glob(pattern, fn, false);

    std::size_t count = fn.size();

    std::vector<std::string> image_path;

    for(int i=0; i<count; i++){
        image_path.push_back(SplitFilename(fn[i]));
    }
    return image_path;
}

//读取图片路径
std::vector<std::string> read_img(std::string pattern){
    std::vector<std::string> fn;
    cv::glob(pattern, fn, false);
    return fn;
}

int detFace(cv::Mat input_rgb, const char* path, std::vector<Bbox> &finalBbox)
{
    static MTCNN mtcnn;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        mtcnn.load(model_path);
        is_model_prepared = true;
        __android_log_print(ANDROID_LOG_INFO, "ProjectName", "Output statement %s", "fxxking keypoints");
    }
    mtcnn.detect(input_rgb, finalBbox, 0.25f);  //
    return 0;
}
//==-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=--=-=-=-=-=-=-=-=-=-=--=--=-
void DectectAndGetFeature(const char* path){
    //这里是入库函数
    //获取所有的图片
    std::vector<std::string> imgs = read_img("/storage/emulated/0/etonedu/img");
    //获取所有的图片名称
    std::vector<std::string> name = read_img_path("/storage/emulated/0/etonedu/img");
    std::ofstream ofs;
    std::string save_path = "/storage/emulated/0/MnnModel/ImageAligan";

    for (int i = 0; i < imgs.size(); ++i) {
        ofs.open("/storage/emulated/0/MnnModel/feature.txt", std::ios::out|std::ios::app);
        __android_log_print(ANDROID_LOG_INFO, "imgProcess", "图片个数i:\t%d", i);
        std::string PicName = name[i];
        std::string PicPath = imgs[i];
        __android_log_print(ANDROID_LOG_INFO, "imgProcess", "这个是:\t%s的图片", PicName.c_str());
        cv::Mat img = cv::imread(imgs[i]);

        //begin:将获取到的图片,进行转置，转置之后需要进行通道变换，接着就进行人脸检测
        cv::Mat input_rgb, input_rgb_T, ImageAligned;
        //矩阵转置
        //	input_rgb = input_rgb.t();
        cv::transpose(img, input_rgb_T);
//    std::string rgb_t_path_1 = save_path + "/Original_Image_T.jpg";
//    cv::imwrite(rgb_t_path_1, input_rgb_T);

        //这个是人脸检测的图片变换，需要将图片从BGR变成RGB
        cv::cvtColor(input_rgb_T, input_rgb, cv::COLOR_BGRA2RGB);  //通道变换
//    std::string s_path_1 = save_path + "/Original_Image_BGRA2RGB.jpg";
//    cv::imwrite(s_path_1, input_rgb);

        std::vector<Bbox> finalBbox;      //定义一个vector类型的 finalBbox
        clock_t faceDstartTime, faceDendTime, faceRstartTime, faceRendTime;

        faceDstartTime = clock();

        detFace(input_rgb, path, finalBbox);

        faceDendTime = clock();
        __android_log_print(ANDROID_LOG_INFO, "人脸检测的时间", "time:\t%f",
                            (double) (faceDendTime - faceDstartTime) / CLOCKS_PER_SEC);

        ///获得所有的人脸数据，包括人名，特征
//        std::vector<std::string> get_allData;
//        get_allData = get_data();

        std::string name;

        KEYPOINT_S keypoint;
        std::vector<KEYPOINT_S> keypoints;

        for (int i = 0; i < finalBbox.size(); i++) {
            keypoint.px1 = finalBbox[i].ppoint[0];
            keypoint.py1 = finalBbox[i].ppoint[5];
            keypoint.px2 = finalBbox[i].ppoint[1];
            keypoint.py2 = finalBbox[i].ppoint[6];
            keypoint.px3 = finalBbox[i].ppoint[2];
            keypoint.py3 = finalBbox[i].ppoint[7];
            keypoint.px4 = finalBbox[i].ppoint[3];
            keypoint.py4 = finalBbox[i].ppoint[8];
            keypoint.px5 = finalBbox[i].ppoint[4];
            keypoint.py5 = finalBbox[i].ppoint[9];
            keypoints.push_back(keypoint);
        }
        static FaceRecognize faceRecognize;
///////============================== m人脸模型加载 load_model-===========================
        static bool is_face_model_prepared = false;
        if (false == is_face_model_prepared) {
            const char *model_path = path;
            faceRecognize.load_model(model_path);
            is_face_model_prepared = true;
        }
/////=====================================================================================
        for (int i = 0; i < keypoints.size(); i++) {
            ///=====================源图片经过人脸对齐矫正=========================
            //img原图，imagealignes是占位符号，所扣下来的图片赋予它，keypoint是人脸关键五个点。
            faceRecognize.preprocess(img, ImageAligned, &keypoints[i]);
            __android_log_print(ANDROID_LOG_INFO, "ProjectName",
                                "ImageAligned height: %d, width: %d, channel: %d",
                                ImageAligned.rows, ImageAligned.cols, ImageAligned.channels());
            std::string j = std::to_string(i);
            std::string Aligned_path = save_path +"/ImageAligned1"+PicName+".jpg";
            cv::imwrite(Aligned_path, ImageAligned);
            //这里要注意的是需要
//            cv::cvtColor(ImageAligned, ImageAligned, cv::COLOR_BGR2RGB);
//            std::string Aligned_BGRA2BGR_path = save_path + "/ImageAligned_BGRA2BGR.jpg";
//            cv::imwrite(Aligned_BGRA2BGR_path, ImageAligned);
//            std::vector<float> embedding;
//            embedding.clear();
//            faceRecognize.getEmbedding(ImageAligned, embedding);
//            __android_log_print(ANDROID_LOG_INFO, "embedding", "embedding121212 %f",embedding[0]);
            std::vector<float> emb;
            std::vector<float> embeding;
            embeding.clear();
            faceRstartTime = clock();
            embeding = faceRecognize.RecogNet(ImageAligned, emb, save_path);
            faceRendTime = clock();
            __android_log_print(ANDROID_LOG_INFO, "imgProcess人脸特征输出", "feature out\t%f", embeding[0]);
            ofs<<PicName;
            for (int j = 0; j < 128; j++)
            {
                __android_log_print(ANDROID_LOG_INFO, "imgProcess人脸特征输出", "feature out\t%f", embeding[j]);
                ofs<<" "<<embeding[j];
            }
            ofs<<"\n";
            ofs.close();
        }
    }
    __android_log_print(ANDROID_LOG_INFO, "人脸入库完成", "录入成功");
    sleep(10000);

}
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=--==--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=---=-=--=-=-=-=
int deal(cv::Mat& frame, const char* path)
{

    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "path:\t%s", path);
    std::string save_path = path ;
    std::string s_path = save_path + "/Original_Image.jpg";
    cv::imwrite(s_path, frame);
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "original size:\t%d, \t%d, \t%d",frame.channels(), frame.rows, frame.cols);
//=======================================这个是分割线==================================================
//    DectectAndGetFeature(path);
//========================================这个是分割线====================================================
    //begin:将获取到的图片,进行转置，转置之后需要进行通道变换，接着就进行人脸检测
    cv::Mat input_rgb, input_rgb_T, ImageAligned;
    //矩阵转置
    //	input_rgb = input_rgb.t();
    cv::transpose(frame,input_rgb_T);
//    std::string rgb_t_path_1 = save_path + "/Original_Image_T.jpg";
//    cv::imwrite(rgb_t_path_1, input_rgb_T);

    cv::cvtColor(input_rgb_T, input_rgb, cv::COLOR_BGRA2RGB);  //通道变换
//    std::string s_path_1 = save_path + "/Original_Image_BGRA2RGB.jpg";
//    cv::imwrite(s_path_1, input_rgb);

	std::vector<Bbox> finalBbox;      //定义一个vector类型的 finalBbox
    clock_t faceDstartTime,faceDendTime,faceRstartTime,faceRendTime;

    faceDstartTime = clock();

	detFace(input_rgb, path, finalBbox);

    faceDendTime = clock();
    __android_log_print(ANDROID_LOG_INFO, "人脸检测的时间", "time:\t%f", (double)(faceDendTime - faceDstartTime)/ CLOCKS_PER_SEC);

    ///获得所有的人脸数据，包括人名，特征
	std::vector<std::string> get_allData;
	get_allData = get_data();

    std::string name;

    KEYPOINT_S keypoint;
    std::vector<KEYPOINT_S> keypoints;

    for (int i = 0; i < finalBbox.size(); i++)
    {
        keypoint.px1 = finalBbox[i].ppoint[0];
        keypoint.py1 = finalBbox[i].ppoint[5];
        keypoint.px2 = finalBbox[i].ppoint[1];
        keypoint.py2 = finalBbox[i].ppoint[6];
        keypoint.px3 = finalBbox[i].ppoint[2];
        keypoint.py3 = finalBbox[i].ppoint[7];
        keypoint.px4 = finalBbox[i].ppoint[3];
        keypoint.py4 = finalBbox[i].ppoint[8];
        keypoint.px5 = finalBbox[i].ppoint[4];
        keypoint.py5 = finalBbox[i].ppoint[9];
        keypoints.push_back(keypoint);
    }
    static FaceRecognize faceRecognize;
///////============================== m人脸模型加载 load_model-===========================
    static bool is_face_model_prepared = false;
    if (!is_face_model_prepared)
    {
        const char* model_path = path;
        faceRecognize.load_model(model_path);
        is_face_model_prepared = true;
    }
/////=====================================================================================
	for (auto & keypoint : keypoints)
    {
        ///=====================源图片经过人脸对齐矫正=========================
        faceRecognize.preprocess(frame, ImageAligned, &keypoint);
        __android_log_print(ANDROID_LOG_INFO, "ProjectName", "ImageAligned height: %d, width: %d, channel: %d", ImageAligned.rows, ImageAligned.cols, ImageAligned.channels());

        std::string Aligned_path = save_path + "/ImageAligned.jpg";
        cv::imwrite(Aligned_path, ImageAligned);

        cv::cvtColor(ImageAligned, ImageAligned, cv::COLOR_BGRA2BGR);
//        std::string Aligned_BGRA2BGR_path = save_path + "/ImageAligned_BGR2RGB.jpg";
//
//        cv::imwrite(Aligned_BGRA2BGR_path, ImageAligned);
//        faceRecognize.getEmbedding(ImageAligned, embedding);
//        __android_log_print(ANDROID_LOG_INFO, "embedding", "embedding %f",embedding[0]);
        std::vector<float> emb;
        std::vector<float> embeding;
        embeding.clear();
        faceRstartTime = clock();
        embeding=faceRecognize.RecogNet(ImageAligned, emb, save_path);
        faceRendTime = clock();

        __android_log_print(ANDROID_LOG_INFO, "特征提取的时间", "time:\t%f", (double)(faceRendTime - faceRstartTime)/ CLOCKS_PER_SEC);

        for(int j = 0;j<get_allData.size();j++){
            std::vector<float> feature = segmentation_feature(get_allData[j]);
            double distance;
            distance = calculSimilar1(feature,embeding,1);
            __android_log_print(ANDROID_LOG_INFO, "distance的值为","distance的值为 %f", distance);
            if(distance>0.65){
                name = segmentation_name(get_allData[j]);
                __android_log_print(ANDROID_LOG_INFO, "distance距离","name名字 %s", name.c_str());
                break;
            } else{
//                __android_log_print(ANDROID_LOG_INFO, "distance距离","name %s", "还好还好");
                name = "other";
            }
//            __android_log_print(ANDROID_LOG_INFO, "这是一个标记符","name %s", name.c_str());
        }
////
    }
//	////========================================================-=-=-=-=-=-=-=-=-=-=-=
	for (int i = 0; i < finalBbox.size(); i++)
	{
		cv::Scalar color = cv::Scalar(255, 0,   0, 255);  //定义color的颜色

        for (int s = 0; s < 5; s++)
        {
            cv::Point2f pt(finalBbox[i].ppoint[s], finalBbox[i].ppoint[s + 5]);  //定义五个关键点的坐标
//            __android_log_print(ANDROID_LOG_INFO, "finalBbox[i]", "finalBbox[x1]%f,finalBbox[y1]%f",finalBbox[i].ppoint[s], finalBbox[i].ppoint[s + 5]);
//            cv::Point2f pt(finalBbox[i].ppoint[s + 5], finalBbox[i].ppoint[s]);  //定义五个关键点的坐标
            cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0, 255), cv::FILLED);  //画五个关键点。
        }
        __android_log_print(ANDROID_LOG_INFO, "名字输出","name %s", name.c_str());
        cv::Rect rs(finalBbox[i].rect().x,      finalBbox[i].rect().y,
                    finalBbox[i].rect().width, finalBbox[i].rect().height);   //定义人脸框坐标点
		cv::rectangle(frame, rs, color, 1);    //画框
		cv::putText(frame,name,cv::Point(finalBbox[i].rect().x, finalBbox[i].rect().y),cv::FONT_HERSHEY_SIMPLEX,2, color, 4, 8);
	}
    return 0;
}