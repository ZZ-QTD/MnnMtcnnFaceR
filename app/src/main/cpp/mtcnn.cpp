#include <algorithm>
#include <map>
#include "mtcnn.h"
#include <android/log.h>
#include <jni.h>

bool cmpScore(Bbox lsh, Bbox rsh)
{
    bool res = lsh.score < rsh.score;
    return res;
}

bool cmpArea(Bbox lsh, Bbox rsh)
{
    bool res = lsh.area() >= rsh.area();
    return res;
}

MTCNN::MTCNN()
{ }

MTCNN::~MTCNN()
{ }

void MTCNN::load(const std::string &model_path, int num_thread)
{
    std::vector<float> mean_vals{ 127.5, 127.5, 127.5 };
    std::vector<float> norm_vals{ 0.0078125, 0.0078125, 0.0078125 };

    std::vector<std::string> tmpp = { model_path + "/det1.mnn" };
    Pnet.load_param(tmpp, num_thread);
    Pnet.set_params(1, 1, mean_vals, norm_vals);  //

    std::vector<std::string> tmpr = { model_path + "/det2.mnn" };
    Rnet.load_param(tmpr, num_thread);
    Rnet.set_params(1, 1, mean_vals, norm_vals);

    std::vector<std::string> tmpo = { model_path + "/det3.mnn" };
    Onet.load_param(tmpo, num_thread);
    Onet.set_params(1, 1, mean_vals, norm_vals);
}

void MTCNN::generateBbox(cv::Mat score, cv::Mat location, std::vector<Bbox>& boundingBox_, float scale)
{
    const int stride = 2;
    const int cellsize = 12;

    int sc_rows, sc_cols;// 13,9
    if ( 4 == score.dims)
    {
        ///因为这个是一个四通道，所以他们的结构是 N C H W ，分别对应着 0 1 2 3 ，在索引2的位置是 H， 3的位置是W。
        sc_rows = score.size[2];    // = 1
        sc_cols = score.size[3];    // = 1
    }
///p是一个地址，首先这个score是一个两通道的Mat结构(根据Pnet得出)，前部分指向score.data第一个数据的地址，因为这里有两个通道
///而且第一层是得到背景的地址，第二层才是人脸的概率，所以加上第二部分的sc_rows * sc_cols，便可以得到第二层的首要地址。
    float* p  = (float *)score.data + sc_rows * sc_cols;
///这个是scale缩放因子，因为图像金字塔的存在，所以需要将特征图除以缩放因子，
    float inv_scale = 1.0f / scale;
    for(int row = 0; row < sc_rows; row++)
    {
        for(int col = 0; col < sc_cols; col++)
        {
            Bbox bbox;
            if( *p > threshold[0] )  ///threshold[0]为0.6，置信度大于0.6即可筛选出来。
            {
                /// *p解引用，获得当前地址的数据，
                bbox.score = *p;
                ///输入图片为CHW，得到的特征图为WH，所以拿到特征索引之后，要进行翻转，eg：（row,col）->转为（col,row）
                ///接着映射到图片位置既为[(col*stride, row*stride),(col*stride + kenel, row*stride + kenel)]
                ///这里的kenel即为卷积核，这里非常有意思就是因为输入是一个Pnet，卷积核大小为12*12，而在实践过程中，将3*3代替12*12
                ///那么Pnet的网络结构就只是为了代替这个12*12，恩。。。可以这么理解。
                ///这里的+1，可以忽略
                bbox.x1 = round((stride * col + 1) * inv_scale);
                bbox.y1 = round((stride * row + 1) * inv_scale);
                bbox.x2 = round((stride * col + 1 + cellsize) * inv_scale);
                bbox.y2 = round((stride * row + 1 + cellsize) * inv_scale);

//                int w = bbox.x2 - bbox.x1;
//                int h = bbox.y2 - bbox.y1;

//                bbox.y1 = bbox.y1 + h;
//                bbox.y2 = bbox.y2 - h;


                const int index = row * sc_cols + col;
                for(int channel = 0;channel < 4; channel++)
                {
                    ///这里边是location的输入,可以理解为将偏移量全部拿到,这个写的代码的含义先不用理解，瞎几把操作
                    float* tmp = (float *)(location.data) + channel * sc_rows * sc_cols;
                    bbox.regreCoord[channel] = tmp[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
        }
    }

    return;
}

void MTCNN::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname)
{
    if (boundingBox_.empty()) return;

    ///参数1：第一个参数是数组的首地址，一般写上数组名就可以，因为数组名是一个指针常量。
    ///参数2：第二个参数相对较好理解，即首地址加上数组的长度n（代表尾地址的下一地址）。
    ///参数3：默认可以不填，如果不填sort会默认按数组升序排序。也就是1,2,3,4排序。也可以自定义一个排序函数，改排序方式为降序什么的，也就是4,3,2,1这样。
    ///bool compare(int a,int b)
    ///{return a<b; //升序排列，如果改为return a>b，则为降序}
    ///本文是一个升序排列
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);

    float iou = 0; float maxX = 0; float maxY = 0; float minX = 0; float minY = 0;

    std::vector<int> vPick; int nPick = 0;

    std::multimap<float, int> vScores;
    ///计算盒子的数量
    const int num_boxes = boundingBox_.size();
    ///给ｖPick赋予空间容量；
    vPick.resize(num_boxes);

    for (int i = 0; i < num_boxes; ++i)
    {   ///将bouningbox排列好的分数赋予vScores
        ///结构体为{(    boundingBox_[i].score: 0.704024, idx: 1
        ///    boundingBox_[i].score: 0.862983, idx: 2
        ///    boundingBox_[i].score: 0.930548, idx: 3
        ///    boundingBox_[i].score: 0.962403, idx: 4)}
        vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
//        __android_log_print(ANDROID_LOG_INFO, "nms", "boundingBox_[i].score: %f, idx: %d", boundingBox_[i].score, i);
    }
///
    if (1 == vScores.size())
    {
        vPick[0] = vScores.rbegin()->second;
        nPick += 1;
    }

    while(vScores.size() > 1)
    {
        ///因为这个boundingBox是按照升序进行排列，所以拿到的这个last是最大的值
        int last = vScores.rbegin()->second;
//        __android_log_print(ANDROID_LOG_INFO, "vScores.rbegin()->second", "last: %d", last);
        ///vpick这个容器的作用是将置信度较高的bouningbox的框保存,pick是挑选的意思,将最大的一个置信度值赋予pick
        vPick[nPick] = last;
        nPick += 1;

        if (nPick >= vPick.size())break;
        ///iterator,迭代器
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();)
        {
            int it_idx = it->second;
            ///这里对ＩＯＵ进行计算,
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);

            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;

            iou = maxX * maxY;
            if (!modelname.compare("Union"))
            {
                iou = iou / (boundingBox_.at(it_idx).area() + boundingBox_.at(last).area() - iou);
            }
            else if(!modelname.compare("Min"))
            {
                iou = iou / ((boundingBox_.at(it_idx).area() < boundingBox_.at(last).area())
                    ? boundingBox_.at(it_idx).area() : boundingBox_.at(last).area());
            }

            if(iou > overlap_threshold)
            {
                ///从容器中移除指定的元素
                it = vScores.erase(it);
                if ( vScores.size() <= 1)
                {
                    break;
                }
            }
            else
            {
                it++;
            }
        }
    }
    ///
    vPick.resize(nPick);
    std::vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++)
    {
        ///vPick[i]zheg
        tmp_[i] = boundingBox_[vPick[i]];
//        __android_log_print(ANDROID_LOG_INFO, "boundingBox_[vPick[i]]", "vPick[i]: %d", vPick[i]);
    }
    boundingBox_ = tmp_;
}

///最后还多了一个refineAndSquareBox的后处理过程，这个函数是把所有留下的框变成正方形并且将这些框的边界限定在原图长宽范围内。
void MTCNN::refine(std::vector<Bbox>& vecBbox, const int& height, const int& width, bool square)
{
    if (vecBbox.empty())return;
    
    float bbw = 0, bbh = 0, max_side = 0;
    float h = 0, w = 0;
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    for (auto it = vecBbox.begin(); it != vecBbox.end(); it++)
    {
        ///这里计算建议框的边长，
        bbw = it->x2 - it->x1 + 1;
        bbh = it->y2 - it->y1 + 1;
        ///之前计算的是偏移量（x-x_）/w=offset,计算x的时候,x=w*offset + x_
        x1 = it->x1 + bbw * it->regreCoord[1];
        y1 = it->y1 + bbh * it->regreCoord[0];
        x2 = it->x2 + bbw * it->regreCoord[3];
        y2 = it->y2 + bbh * it->regreCoord[2];
//        __android_log_print(ANDROID_LOG_INFO, "regreCoord[1]", "regreCoord[1]: %f", it->regreCoord[1]);
//        __android_log_print(ANDROID_LOG_INFO, "regreCoord[0]", "regreCoord[0]: %f", it->regreCoord[0]);
//        __android_log_print(ANDROID_LOG_INFO, "regreCoord[3]", "regreCoord[3]: %f", it->regreCoord[3]);
//        __android_log_print(ANDROID_LOG_INFO, "regreCoord[2]", "regreCoord[2]: %f", it->regreCoord[2]);
        if(square)
        {
              w = x2 - x1 + 1;
              h = y2 - y1 + 1;
              int maxSide = ( h > w ) ? h:w;
              x1 = x1 + w * 0.5 - maxSide * 0.5;
              y1 = y1 + h * 0.5 - maxSide * 0.5;
              x2 = round(x1 + maxSide - 1);
              y2 = round(y1 + maxSide - 1);
              x1 = round(x1);
              y1 = round(y1);



         }

        it->x1 = x1 < 0 ? 0 : x1;
        it->y1 = y1 < 0 ? 0 : y1;
        it->x2 = x2 >= width  ? width  - 1 : x2;
        it->y2 = y2 >= height ? height - 1 : y2;
    }
}

void MTCNN::PNet()
{
    int img_h = img.rows;
    int img_w = img.cols;
    __android_log_print(ANDROID_LOG_INFO, "MTCNN.CPP", "Pnet_img_channel: %d", img.channels());

    /***这个列表中放的就是一系列的scale值，表示依次将原图缩小成原图的scale倍，从而组成图像金字塔***/

    if (scales_.empty())
    {
        const int min_det_size = 12;      ///
        const int minsize      = 40;
        const float pre_facetor = 0.5f;
        float minl = img_w < img_h ? img_w: img_h;   ///选择最小的宽或者长 640*480
        float m = (float)min_det_size / minsize;     /// 12/40=0.3
        minl *= m;  ///480*0.5
        float factor = pre_facetor;
        __android_log_print(ANDROID_LOG_INFO, "MTCNN.CPP", "test1111111111");
        while(minl > min_det_size)    /////图像金字塔，将最小边跟12做比较，如果长于12，则继续缩小，并添加到scales之中，直到小于12
        {
            scales_.push_back(m);
            minl *= factor;   ///480*0.5 = 240*0.5  = 120*0.5   =  60x0.5    =   30x0.5     = 15
            m = m * factor;   ///0.3x0.5=  0.15x0.5 = 0.075x0.5 =  0.0375x0.5= 0.01875x0.5  = 0.009375
        }
    }

    __android_log_print(ANDROID_LOG_INFO, "MTCNN.CPP", "test2222222222222");
    firstBbox_.clear();
    for (size_t i = 0; i < scales_.size(); i++)
    {
        __android_log_print(ANDROID_LOG_INFO, "MTCNN.CPP", "test33333333333333333");
        int hs = (int)ceil(img_h * scales_[i]);  ///将浮点数转化为整数,向上取整
        int ws = (int)ceil(img_w * scales_[i]);

        cv::Mat in;
        cv::resize(img, in, cv::Size(ws, hs));       //一次次生成图像金字塔中的一层图
        cv::imwrite("/storage/emulated/0/model/789.jpg", in);
        cv::imwrite("/storage/emulated/0/model/101112.jpg", img);

        __android_log_print(ANDROID_LOG_INFO, "MTCNN.CPP", "test44444444444444");
        Inference_engine_tensor out;
        std::string tmp_str = "prob1";
        out.add_name(tmp_str);

        std::string tmp_str1 = "conv4-2";
        out.add_name(tmp_str1);

        Pnet.infer_img(in, out);   ///out 包含模型文件中的节点名称

        std::vector<Bbox> boundingBox_;
        ///

        __android_log_print(ANDROID_LOG_INFO, "out_feat", "out_feat[0]\t%d, out_feat[0][0]\t%d, out_feat[0][0]\t%d", out.out_feat[0].dims, out.out_feat[0].size[2], out.out_feat[0].size[3]);
        generateBbox(out.out_feat[0], out.out_feat[1], boundingBox_, scales_[i]);
        nms(boundingBox_, 0.7);
        ///insert(const_iterator __position, _ForwardIterator __first, _ForwardIterator __last);
        firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
        boundingBox_.clear();
    }
    return;
}
void MTCNN::RNet()
{
    secondBbox_.clear();
    int count = 0;
    for(std::vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++)
    {
        ///获得框之后，对原图片进行框区域截图
        cv::Rect rs = it->rect();
        cv::Mat in = img(rs).clone();

        cv::resize(in, in, cv::Size(24, 24));

        Inference_engine_tensor out;
        std::string tmp_str = "prob1";
        out.add_name(tmp_str);

        std::string tmp_str1 = "conv5-2";
        out.add_name(tmp_str1);
        Rnet.infer_img(in, out);

        float* thresh = (float*)out.out_feat[0].data;
        float* bbox   = (float*)out.out_feat[1].data;
        Bbox tmp_box = *it;
        if ( thresh[1] > threshold[1] )
        {
            for (int channel = 0; channel < 4; channel++)
            {
                tmp_box.regreCoord[channel] = bbox[channel];
            }
            tmp_box.score = thresh[1]; 
            secondBbox_.push_back(tmp_box);
        }
    }
}

void MTCNN::ONet()
{
    thirdBbox_.clear();
    for(std::vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++)
    {
        cv::Rect rs = it->rect();
        cv::Mat in = img(rs).clone();
        cv::resize(in, in, cv::Size(48, 48));

        Inference_engine_tensor out;
        std::string tmp_str = "prob1";
        out.add_name(tmp_str);

        std::string tmp_str1 = "conv6-2";
        out.add_name(tmp_str1);

        std::string tmp_str2 = "conv6-3";
        out.add_name(tmp_str2);

        Onet.infer_img(in, out);

        float* thresh = (float*)out.out_feat[0].data;
        float* bbox   = (float*)out.out_feat[1].data;
        float* keyPoint = (float*)out.out_feat[2].data;

        if (thresh[1] > threshold[2])
        {
            for (int channel = 0; channel < 4; channel++)
            {
                it->regreCoord[channel] = bbox[channel];
            }
            it->score = thresh[1];
            for (int num = 0; num < 5; num++)
            {
                (it->ppoint)[num]     = it->x1 + it->width()  * keyPoint[num + 5] - 1;
                (it->ppoint)[num + 5] = it->y1 + it->height() * keyPoint[num] - 1;
            }
            thirdBbox_.push_back(*it);
        }
    }
}

void MTCNN::detectInternal(cv::Mat& img_, std::vector<Bbox>& finalBbox_)
{
    const float nms_threshold[3] = {0.7f, 0.7f, 0.7f};

    img = img_;   //这个img相当于全局变量？

    PNet();
    if ( !firstBbox_.empty())
    {
        __android_log_print(ANDROID_LOG_INFO, "Mtcnn", "firstBbox_.size\t%d", firstBbox_.size());
        nms(firstBbox_, nms_threshold[0]);
        refine(firstBbox_, img_.rows, img_.cols, true);

        RNet();
        if( !secondBbox_.empty())
        {
            nms(secondBbox_, nms_threshold[1]);
            refine(secondBbox_, img_.rows, img_.cols, true);

            ONet();
            if ( !thirdBbox_.empty())
            {
                refine(thirdBbox_, img_.rows, img_.cols, false);

                std::string ts = "Min";
                nms(thirdBbox_, nms_threshold[2], ts);
            }
        }
    }

    finalBbox_ = thirdBbox_;
    __android_log_print(ANDROID_LOG_INFO, "Mtcnn", "finalBbox_:\t%d", finalBbox_.size());
}

void MTCNN::detect(const cv::Mat& img_, 
                   std::vector<Bbox>& finalBbox, 
                   float scale,
                   cv::Rect detRegion) //cv::Rect detRegion = cv::Rect()
{   
    if (img_.empty()) return;

    cv::Mat testImg;  //

    if (detRegion.x > 0  && detRegion.y > 0
        && detRegion.br().x < img_.cols && detRegion.br().y < img_.rows)      //?????这是判断个fxxking what
    {
        testImg = img_(detRegion).t();
//        testImg = img_(detRegion);
    }
    else
    {
//        testImg = img_.t();
        testImg = img_;
//        cv::imwrite(, testImg);
    }
    std::string path = "/storage/emulated/0/MnnModel";
    std::string testImg_path = path + "/testImg.jpg";
    cv::imwrite(testImg_path, testImg);

    if ( scale < 1.0f && scale > 0.0f )
    {
        cv::resize(testImg, testImg, cv::Size(), scale, scale); //testimg.width testimg.heigth = testimg*scale,将图片缩放scale倍
    }
    
    detectInternal(testImg, finalBbox);

    for (int i = 0; i < finalBbox.size(); i++)
    {
        if ( scale < 1.0f && scale > 0.0f )
        {
            finalBbox[i].scale_inverse(scale);
        }

        finalBbox[i] = finalBbox[i].trans();

        for (int ii = 0; ii < 5; ii++)
        {
            finalBbox[i].ppoint[ii]     += detRegion.x;
            finalBbox[i].ppoint[ii + 5] += detRegion.y;
        }

        finalBbox[i].x1 += detRegion.x;
        finalBbox[i].y1 += detRegion.y;
        finalBbox[i].x2 += detRegion.x;
        finalBbox[i].y2 += detRegion.y;
    }

    return;
}

