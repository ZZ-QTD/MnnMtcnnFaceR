//
// Created by qiu on 2021/6/30.
//

#include <algorithm>
#include <map>
#include "FaceRecognize.h"
#include <fstream>


FaceRecognize::FaceRecognize()
{

}
FaceRecognize::~FaceRecognize()
{

}

int FaceRecognize::getEmbedding(cv::Mat &img, std::vector<float> &embedding_list) {
    if(!mMNNInterpreter || !mMNNSession)
    {
        return -1;
    }
    int image_h = img.rows;
    int image_w = img.cols;
    int in_h = 112;
    int in_w = 112;

    cv::Mat image;
    if(image_h == 112 && image_w == 112)
    {
        image = img;
    }
    else{cv::resize(img, image, cv::Size(in_h, in_w));}

    mMNNInterpreter -> resizeTensor(mInputTensor,{1,3,in_h,in_w});
    mMNNInterpreter -> resizeSession(mMNNSession);

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB));

    mMNNInterpreter -> runSession(mMNNSession);

    std::string ebd_layer = "fc1_scale";

    MNN::Tensor* tensor_embedding = mMNNInterpreter -> getSessionOutput(mMNNSession, ebd_layer.c_str());
    MNN::Tensor tensor_embedding_host(tensor_embedding, tensor_embedding -> getDimensionType());
    tensor_embedding -> copyToHostTensor(&tensor_embedding_host);

    embedding_list.clear();
    for(int i=0; i<128; i++)
    {
        embedding_list.push_back(tensor_embedding_host.host<float>()[i]);
    }

    return 0;

}

void FaceRecognize::load_model(const std::string& model_path, int num_thread)
{
    std::vector<float> mean_vals{ 127.5, 127.5, 127.5 };
    std::vector<float> norm_vals{ 0.0078125, 0.0078125, 0.0078125 };

    std::vector<std::string> FaceRmnn = {model_path + "/caffe_mobileface.mnn"};
    FaceRnet.face_load_param(FaceRmnn, num_thread);
//    FaceRnet.face_set_params_recog(1, 2);
    FaceRnet.face_set_params(1, 2,mean_vals,norm_vals);
}

std::vector<float> FaceRecognize::RecogNet(cv::Mat &img_ ,std::vector<float> emb,std::string path) {

    feature_out.resize(128);
//    cv::Mat in;
    std::vector<float> embeding;
//
    cv::resize(img_, img_, cv::Size(112, 112));

    Face_Inference_engine_tensor out;
    std::string tmp_str = "fc1_scale";
    out.add_name(tmp_str);
//    /定义文件，写入特征数据
//    std::ofstream ofs;
//    ofs.open(path+"/feature.txt", std::ios::out|std::ios::app);

    FaceRnet.face_infer_img(img_, emb, out);


    float* feature = (float*)out.out_feat[0].data;



    for (int j = 0; j < 128; j++)
    {
        feature_out[j] = feature[j];
    }
//    normalize(feature_out);
//    for (int i = 0; i < 128; i++)
//    {
////        __android_log_print(ANDROID_LOG_INFO, "人脸特征输出", "feature out\t%f", feature_out[i]);
////        ofs<<feature_out[i];
//    }
//    ofs<<"------"<<std::endl;
//    ofs.close();

    embeding = feature_out;
    return embeding;
}

void FaceRecognize::normalize(std::vector<float> &feature) {
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++) {
        sum += (float)*it * (float)*it;
    }
    sum = sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++) {
        *it /= sum;
    }
}

void Calcul_ElewiseMinus(const cv::Mat &SrcMatA, const cv::Mat &SrcMatB, cv::Mat &DstMat)
{
    unsigned int i = 0, j = 0;

    DstMat.create(SrcMatA.rows, SrcMatA.cols, SrcMatA.type());
    assert(SrcMatB.rows == SrcMatA.cols);
    if(SrcMatB.rows == SrcMatA.cols) {
        for(i = 0 ; i <  SrcMatA.rows; i ++) {
            for(j = 0 ; j < SrcMatA.cols; j++) {
                DstMat.at<float>(i, j) = SrcMatA.at<float>(i, j) - SrcMatB.at<float>(j, 0);
            }
        }
    }

}
void Calcul_MeanAxis0(const cv::Mat &SrcMat, cv::Mat &DstMat)
{
    float fColSum = 0 ;
    unsigned int i = 0, j = 0;
    unsigned int u32Height = SrcMat.rows;   //5
    unsigned int u32Width = SrcMat.cols;    //2
    __android_log_print(ANDROID_LOG_INFO, "u32Height", "u32Height%d",u32Height);

    // x1 y1
    // x2 y2
    DstMat.create(u32Width, 1, CV_32F);
    for(i = 0; i <  u32Width; i ++) {
        fColSum = 0 ;
        for(j = 0; j < u32Height; j++) {
            fColSum += SrcMat.at<float>(j, i);
        }
        DstMat.at<float>(i, 0) = fColSum / u32Height;
    }

}
static int Calcul_MatrixRank(cv::Mat &SrcMat)
{
    cv::Mat S, U, V;
    unsigned int i = 0, j = 0;

    cv::SVD::compute(SrcMat, S, U, V);
    cv::Mat_<unsigned char> nonZeroSingularValues = S;

    unsigned int u32Rank = countNonZero(nonZeroSingularValues > 0.0001);   //计算非零像素点的个数
    //printf("u32Rank: %d\n", u32Rank);

    return u32Rank;
}
void Calcul_VarAxis0(const cv::Mat &SrcMat, cv::Mat &SrcMatVarAxis0)
{

    cv::Mat SrcMatMean, SrcMatDeMean;

    Calcul_MeanAxis0(SrcMat, SrcMatMean);
    Calcul_ElewiseMinus(SrcMat, SrcMatMean, SrcMatDeMean);
    multiply(SrcMatDeMean, SrcMatDeMean, SrcMatDeMean);
    Calcul_MeanAxis0(SrcMatDeMean, SrcMatVarAxis0);


}

void FaceRecognize::getAffineMatrix(cv::Mat &Src, cv::Mat &Dst, cv::Mat &TranMatrix,
                                    bool EstimateScale) {
    int s32Ret = 1;
    unsigned int i = 0, j = 0;
    unsigned int u32Rank = 0, u32S = 0;
    float fVal = 0.0f;
    float fScale = 0.0f;
    cv::Mat SrcMean, DstMean;
    cv::Mat SrcDeMean, DstDeMean;
    cv::Mat A, d;
    cv::Mat U, S, V;
    cv::Mat diagD, twp;
    cv::Mat SrcDeMeanVar;
    cv::Mat temp0, temp1, temp2, temp3;

    unsigned int  u32Height = Src.rows;
    unsigned int  u32Width = Src.cols;
    Calcul_MeanAxis0(Src, SrcMean);
    Calcul_MeanAxis0(Dst, DstMean);
    Calcul_ElewiseMinus(Src, SrcMean, SrcDeMean);
    Calcul_ElewiseMinus(Dst, DstMean, DstDeMean);
    A = (DstDeMean.t() * SrcDeMean) / static_cast<float >(u32Height);
    d.create(u32Width, 1, CV_32F);
    d.setTo(1.0f);
    if (determinant(A) < 0) {
        d.at<float>(u32Width - 1, 0) = -1;
    }
    TranMatrix = cv::Mat::eye(u32Width + 1, u32Width + 1, CV_32F);
    cv::SVD::compute(A, S, U, V);
    u32Rank = Calcul_MatrixRank(A);
    if (u32Rank == 0) {
        //printf("Run if\n");
        assert(u32Rank == 0);
    }
    else if (u32Rank == u32Width - 1) {
        //printf("Run elseif\n");
        if (determinant(U) * determinant(V) > 0) {
            TranMatrix.rowRange(0, u32Width).colRange(0, u32Width) = U * V;
        }
        else {

            u32S = d.at<float>(u32Width - 1, 0);
            d.at<float>(u32Width - 1, 0) = -1;
            diagD = cv::Mat::diag(d);    //2 x 2
            TranMatrix.rowRange(0, u32Width).colRange(0, u32Width) = U * diagD * V;
            d.at<float>(u32Width - 1, 0) = u32S;
        }
    }
    else{
        //printf("Run else\n");
        diagD = cv::Mat::diag(d);
        TranMatrix.rowRange(0, u32Width).colRange(0, u32Width) = U * diagD * V;
    }
    if (EstimateScale == 1) {
        Calcul_VarAxis0(SrcDeMean, SrcDeMeanVar);   //计算矩阵的方差
        if (s32Ret != 0) {
            __android_log_print(ANDROID_LOG_INFO, "ProjectName", "Output statement %s", "Calcul_VarAxis0 failed!!!");
        }
        fVal = cv::sum(SrcDeMeanVar).val[0];

        temp0 = S.t() * d;

        fScale =  1.0 / fVal * temp0.at<float>(0, 0);

    }
    else {
        fScale = 1.0;
    }

    temp1 = TranMatrix.rowRange(0, u32Width).colRange(0, u32Width);

    temp2 = temp1 * SrcMean;   //2 x 1

    temp3 = fScale * temp2;      //2 x 1

    TranMatrix.rowRange(0, u32Width).colRange(u32Width, u32Width+1) =  -(temp3 - DstMean);
    TranMatrix.rowRange(0, u32Width).colRange(0, u32Width) *= fScale;

}

int FaceRecognize::preprocess(cv::Mat &SrcMat, cv::Mat &ImageAligned, hiKEYPOINT_S *pstKeyPoints) {

    cv::Mat TranMatrix, M;
    int s32Ret = 0;

    float afStanPoints[5][2] = {                       ///仿射变换目标点,
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}};

    cv::Mat StaPoints(5, 2, CV_32FC1, afStanPoints);
    ///C 库函数 void *memcpy(void *str1, const void *str2, size_t n) 从存储区 str2 复制 n 个字节到存储区 str1。
    memcpy(StaPoints.data, afStanPoints, 2 * 5 * sizeof(float));
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "Output statement %s", "1 keypoints");

    float afKeyPoints[5][2] = {
            {0.0f, 0.0f},
            {0.0f, 0.0f},
            {0.0f, 0.0f},
            {0.0f, 0.0f},
            {0.0f, 0.0f}};

    afKeyPoints[0][0] = pstKeyPoints->px1;
    afKeyPoints[0][1] = pstKeyPoints->py1;
    afKeyPoints[1][0] = pstKeyPoints->px2;
    afKeyPoints[1][1] = pstKeyPoints->py2;
    afKeyPoints[2][0] = pstKeyPoints->px3;
    afKeyPoints[2][1] = pstKeyPoints->py3;
    afKeyPoints[3][0] = pstKeyPoints->px4;
    afKeyPoints[3][1] = pstKeyPoints->py4;
    afKeyPoints[4][0] = pstKeyPoints->px5;
    afKeyPoints[4][1] = pstKeyPoints->py5;
    cv::Mat KeyPoints(5, 2, CV_32FC1, afKeyPoints);
    memcpy(KeyPoints.data, afKeyPoints, 2 * 5 * sizeof(float));
   for (int i = 0; i<5; i++){
       const float* data = KeyPoints.ptr<float>(i);
       for (int j = 0; j<2; j++){

           float data_ = *data++;
           __android_log_print(ANDROID_LOG_INFO, "KeyPoints", "KeyPoints %f", data_);
       }
   }

    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "Output statement %s", "2 keypoints");

    ///KeyPoints表示为输入的五个关键点
    ///afKeyPoint表示为标准矩阵
    ///TranMatrix表示要计算的矩阵
    getAffineMatrix(KeyPoints, StaPoints, TranMatrix, 1);

    M = TranMatrix.rowRange(0, 2);
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "Output statement %s", "3 keypoints");
    //. src: 输入图像
    //. dst: 输出图像，尺寸由dsize指定，图像类型与原图像一致
    //. M: 变换矩阵
    //. dsize: 指定图像输出尺寸
    warpAffine(SrcMat, ImageAligned, M, {112,112});
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "Output statement %s", "4 keypoints");
    // do resize if src size != dst size
    return 0;
}

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2, int distance_metric)
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