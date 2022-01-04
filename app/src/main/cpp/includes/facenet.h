//
// Created by qiu on 2021/7/6.
//

#ifndef ANDROID_FACENET_H
#define ANDROID_FACENET_H

#include <vector>
#include <string>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

#include "opencv2/opencv.hpp"

class Face_Inference_engine_tensor
{
public:
    Face_Inference_engine_tensor()
    { }

    ~Face_Inference_engine_tensor()
    { }

    void add_name(std::string &layer)
    {
        layer_name.push_back(layer);
    }

    float* score(int idx)
    {
        return (float*)out_feat[idx].data;
    }

public:
    std::vector<std::string> layer_name;
    std::vector<cv::Mat> out_feat;
};


class Face_Inference_engine
{
public:
    Face_Inference_engine();
    ~Face_Inference_engine();

    int face_load_param(std::vector<std::string> &file, int num_thread = 1);
    int face_set_params(int inType, int outType, std::vector<float> &mean, std::vector<float> &scale);
    int face_set_params_recog(int srcType, int dstType);
    int face_infer_img(cv::Mat& imgs,  std::vector<float>& embedding_list ,Face_Inference_engine_tensor& out);
    int face_infer_imgs(std::vector<cv::Mat>& imgs, std::vector<Face_Inference_engine_tensor>& out);
private:
    MNN::Interpreter* facenetPtr;
    MNN::Session* facesessionPtr;
    MNN::CV::ImageProcess::Config faceconfig;
};

#endif //ANDROID_FACENET_H
