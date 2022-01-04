#pragma once
#include "opencv2/opencv.hpp"

class Bbox
{
public:
    Bbox();
    ~Bbox();
    Bbox(const Bbox& rhs);
    Bbox& operator=(const Bbox& rhs);

    cv::Rect  rect();   //Rect 是OPENCV里边的基本数据类型，其中
    cv::Point center();
    int       width();
    int       height();
    float     area();

    Bbox trans();
    void scale_inverse(float scale);

public:
    float score;
    int x1, y1, x2, y2;
    float ppoint[10];
    float regreCoord[4];
};

typedef struct hiKEYPOINT_S
{
    float px1;
    float py1;
    float px2;
    float py2;
    float px3;
    float py3;
    float px4;
    float py4;
    float px5;
    float py5;
}KEYPOINT_S;