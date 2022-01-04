//
// Created by qiu on 2021/7/6.
//

#include "facenet.h"
#include <android/log.h>


Face_Inference_engine::Face_Inference_engine()
{ }

Face_Inference_engine::~Face_Inference_engine()
{
    if ( facenetPtr != NULL )
    {
        if ( facesessionPtr != NULL)
        {
            facenetPtr->releaseSession(facesessionPtr);
            facesessionPtr = NULL;
        }

        delete facenetPtr;
        facenetPtr = NULL;
    }
}

int Face_Inference_engine::face_load_param(std::vector<std::string>& file, int num_thread)     //加载模型文件
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
            facenetPtr = MNN::Interpreter::createFromFile(file[0].c_str());   ///createFromFile函数将模型文件读入，并用变量netPtr接受
            if (nullptr == facenetPtr)
            {
                __android_log_print(ANDROID_LOG_INFO, "face_net", "load_model %s", "face_model_null");
                return -1;
            }

            MNN::ScheduleConfig sch_config;     ///
            sch_config.type = (MNNForwardType)MNN_FORWARD_OPENCL;
//            sch_config.type = (MNNForwardType)MNN_FORWARD_OPENGL;
//            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            facesessionPtr = facenetPtr->createSession(sch_config);  //
            if (nullptr == facesessionPtr)
            {
                __android_log_print(ANDROID_LOG_INFO, "face_net", "facesession: %s", "NULL");
                return -1;
            }

        }
        else
        {
            return -1;
        }

        __android_log_print(ANDROID_LOG_INFO, "face_net", "load_model %s", "SUCCESS");
    }

    return 0;
}


int Face_Inference_engine::face_set_params_recog(int srcType, int dstType)
{
    faceconfig.destFormat   = (MNN::CV::ImageFormat)dstType;   ///设置最终图片的格式 RGB
    faceconfig.sourceFormat = (MNN::CV::ImageFormat)srcType;   ///设置最初图片的格式 RGB

    // filterType��wrap
    //config.filterType = (MNN::CV::Filter)(1);    //
    //config.wrap = (MNN::CV::Wrap)(2);

    return 0;
}


int Face_Inference_engine::face_set_params(int srcType, int dstType,
                                 std::vector<float>& mean, std::vector<float>& scale)
{
    faceconfig.destFormat   = (MNN::CV::ImageFormat)dstType;   ///设置最终图片的格式 RGB
    faceconfig.sourceFormat = (MNN::CV::ImageFormat)srcType;   ///设置最初图片的格式 RGB

    // mean��normal
    ::memcpy(faceconfig.mean,   &mean[0],   3 * sizeof(float));      ///由src所指内存区域复制count个字节到dest所指内存区域。memcpy(void *dest, void *src, unsigned int count);
    ::memcpy(faceconfig.normal, &scale[0],  3 * sizeof(float));   ///也就是给mean这个变量开辟空间。

    // filterType��wrap
    faceconfig.filterType = (MNN::CV::Filter)(1);    //
    faceconfig.wrap = (MNN::CV::Wrap)(2);

    return 0;
}

// infer推论
int Face_Inference_engine::face_infer_img(cv::Mat& img,  std::vector<float>& embedding_list,Face_Inference_engine_tensor& out)
{

    MNN::Tensor* tensorPtr = facenetPtr->getSessionInput(facesessionPtr, nullptr);
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "face_infer_img height: %d, width: %d, channel: %d!!!!", img.rows, img.cols, img.channels());

    /// auto resize for full conv network.
    bool auto_resize = false;
    cv::Mat input_rgb;
    cv::cvtColor(img, input_rgb, cv::COLOR_BGR2RGB);
//    cv::resize(input_rgb, input_rgb, cv::Size(48, 48));
//    cv::imwrite("/storage/emulated/0/model/abd.jpg", input_rgb);
//    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "face_infer_img height: %d, width: %d, channel: %d!!!!", input_rgb.rows, input_rgb.cols, input_rgb.channels());

    if ( !auto_resize )
    {
        std::vector<int>dims = { 1, img.channels(), img.rows, img.cols };

        facenetPtr->resizeTensor(tensorPtr, dims);

        facenetPtr->resizeSession(facesessionPtr);
    }

    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(faceconfig));
    process->convert((const unsigned char*)input_rgb.data, input_rgb.cols, input_rgb.rows, input_rgb.step[0], tensorPtr);
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "tensorPtr shape: %d\t%d\t%d\t%d!!!!", tensorPtr->shape()[0], tensorPtr->shape()[1], tensorPtr->shape()[2], tensorPtr->shape()[3]);
    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "face_infer_img height: %s","fucking y mm");
    facenetPtr->runSession(facesessionPtr);////=====-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=

    ////////============================================

    for (int i = 0; i < out.layer_name.size(); i++)
    {
        const char* layer_name = NULL;
        if( strcmp(out.layer_name[i].c_str(), "") != 0)
        {
            layer_name = out.layer_name[i].c_str();

        }
        MNN::Tensor* tensorOutPtr = facenetPtr->getSessionOutput(facesessionPtr, layer_name);

        std::vector<int> shape = tensorOutPtr->shape();

        cv::Mat feat(shape.size(), &shape[0], CV_32F);


        auto tensor = reinterpret_cast<MNN::Tensor*>(tensorOutPtr);
        float *destPtr = (float*)feat.data;
        if (nullptr == destPtr)
        {

            std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), false));  ///std::unique_ptr,为了摧毁tensor

            return hostTensor->elementSize();

        }

        std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), true));

        tensor->copyToHostTensor(hostTensor.get());

        tensor = hostTensor.get();

        auto size = tensor->elementSize();
        ::memcpy(destPtr, tensor->host<float>(), size * sizeof(float));

        out.out_feat.push_back(feat.clone());

    }

    return 0;
}

