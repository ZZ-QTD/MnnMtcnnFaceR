//
// Created by qiu on 2021/6/30.
//

#ifndef ANDROID_FACERECOGNIZE_H
#define ANDROID_FACERECOGNIZE_H

#include <string>
#include <vector>
#include "facenet.h"
#include "Bbox.h"
#include "opencv2/opencv.hpp"
#include <android/log.h>

#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/ImageProcess.hpp>

class FaceRecognize{

protected:
    std::shared_ptr<MNN::Interpreter> mMNNInterpreter;
    MNN::Session* mMNNSession = nullptr;
    MNN::Tensor* mInputTensor = nullptr;


public:
    FaceRecognize();
    ~FaceRecognize();

    int getEmbedding(cv::Mat& img, std::vector<float>& embedding_list);

    void load_model(const std::string& model_path, int num_thread = 1);
    void recognize(cv::Mat& img, std::vector<Bbox>& finalBbox);

    void getAffineMatrix(cv::Mat &Src_img, cv::Mat &Dst_img, cv::Mat &Matrix, bool EstimateScale);

    double calculSimilar(std::vector<float>& v1, std::vector<float>& v2, int distance_metric);

    void normalize(std::vector<float>& feature);

    int preprocess(cv::Mat &Src_img, cv::Mat &ImageAligned, hiKEYPOINT_S *pstKeyPoints );

    std::vector<float> RecogNet(cv::Mat& img_ , std::vector<float> emb,std::string path);



private:
    Face_Inference_engine FaceRnet;
    std::vector<float> feature_out;
};


#endif //ANDROID_FACERECOGNIZE_H
