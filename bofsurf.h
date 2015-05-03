#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

class BofSurf {
public:
    void ImgResizeProcess(const cv::Mat& img, cv::Mat& resized_img);
    void BuildingBoFVocabulary(const std::string& img_path);
    void BoFDescriptor(const std::string& img_path);
};