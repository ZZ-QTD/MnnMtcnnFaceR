#include "net.h"
#include <android/log.h>


Inference_engine::Inference_engine()
{ }

Inference_engine::~Inference_engine()
{ 
    if ( netPtr != NULL )
	{
		if ( sessionPtr != NULL)
		{
			netPtr->releaseSession(sessionPtr);
			sessionPtr = NULL;
		}

		delete netPtr;
		netPtr = NULL;
	}
}

int Inference_engine::load_param(std::vector<std::string>& file, int num_thread)     //加载模型文件
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
	        netPtr = MNN::Interpreter::createFromFile(file[0].c_str());   ///createFromFile函数将模型文件读入，并用变量netPtr接受
            if (nullptr == netPtr) return -1;

            MNN::ScheduleConfig sch_config;     ///
//            sch_config.type = (MNNForwardType)MNN_FORWARD_OPENCL;
            //sch_config.type = (MNNForwardType)MNN_FORWARD_OPENGL;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            sessionPtr = netPtr->createSession(sch_config);
            if (nullptr == sessionPtr) return -1;
        }
        else
        {
            return -1;
        }
    }

    return 0;
}


int Inference_engine::set_params_recog(int srcType, int dstType)
{
    config.destFormat   = (MNN::CV::ImageFormat)dstType;   ///设置最终图片的格式 RGB
    config.sourceFormat = (MNN::CV::ImageFormat)srcType;   ///设置最初图片的格式 RGB

    // filterType��wrap
    //config.filterType = (MNN::CV::Filter)(1);    //
    //config.wrap = (MNN::CV::Wrap)(2);

    return 0;
}


int Inference_engine::set_params(int srcType, int dstType, 
                                 std::vector<float>& mean, std::vector<float>& scale)
{
    config.destFormat   = (MNN::CV::ImageFormat)dstType;   ///设置最终图片的格式 RGB
    config.sourceFormat = (MNN::CV::ImageFormat)srcType;   ///设置最初图片的格式 RGB

    // mean��normal
    ::memcpy(config.mean,   &mean[0],   3 * sizeof(float));      ///由src所指内存区域复制count个字节到dest所指内存区域。memcpy(void *dest, void *src, unsigned int count);
    ::memcpy(config.normal, &scale[0],  3 * sizeof(float));   ///也就是给mean这个变量开辟空间。

    // filterType��wrap
    config.filterType = (MNN::CV::Filter)(1);    //
    config.wrap = (MNN::CV::Wrap)(2);

    return 0;
}

// infer推论
int Inference_engine::infer_img(cv::Mat& img, Inference_engine_tensor& out)
{

    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);
//    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "infer_img height: %d, width: %d, channel: %d!!!!", img.rows, img.cols, img.channels());

    /// auto resize for full conv network.
    bool auto_resize = false;
    if ( !auto_resize )
    {

        std::vector<int>dims = { 1, img.channels(), img.rows, img.cols };

        netPtr->resizeTensor(tensorPtr, dims);

        netPtr->resizeSession(sessionPtr);
    }

    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));

    process->convert((const unsigned char*)img.data, img.cols, img.rows, img.step[0], tensorPtr);
//    __android_log_print(ANDROID_LOG_INFO, "ProjectName", "tensorPtr shape: %d\t%d\t%d\t%d!!!!", tensorPtr->shape()[0], tensorPtr->shape()[1], tensorPtr->shape()[2], tensorPtr->shape()[3]);

    netPtr->runSession(sessionPtr);

    for (int i = 0; i < out.layer_name.size(); i++)
    {
        const char* layer_name = NULL;
        if( strcmp(out.layer_name[i].c_str(), "") != 0)
        {
            layer_name = out.layer_name[i].c_str();
//            __android_log_print(ANDROID_LOG_INFO, "infer_img", "layer_name\t%s", layer_name);
        }
        MNN::Tensor* tensorOutPtr = netPtr->getSessionOutput(sessionPtr, layer_name);

        std::vector<int> shape = tensorOutPtr->shape();
//        __android_log_print(ANDROID_LOG_INFO, "infer_img", "shape[0]\t%d", shape[0]);
        cv::Mat feat(shape.size(), &shape[0], CV_32F);


        auto tensor = reinterpret_cast<MNN::Tensor*>(tensorOutPtr);
        float *destPtr = (float*)feat.data;
        if (nullptr == destPtr)
        {
//            __android_log_print(ANDROID_LOG_INFO, "infer_img", "infer_img\t%s", "destPtr==NULL");
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

int Inference_engine::infer_imgs(std::vector<cv::Mat>& imgs, std::vector<Inference_engine_tensor>& out)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        infer_img(imgs[i], out[i]);
    }

    return 0;
}
