#include "bofsurf.h"

void BofSurf::BuildingBoFVocabulary(const std::string& img_path) {
    const int minHessian = 400; //Hessian Threshold
    cv::Mat img, resized_img, dst;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    cv::Mat featuresUnclustered;
    cv::SurfDescriptorExtractor detector(minHessian);
    std::vector<cv::String> image_dataset;
    cv::glob(img_path + "/*.jpg", image_dataset);
    for (int i = 0; i < image_dataset.size(); i++) {
        img = cv::imread(image_dataset[i]);
        ImgResizeProcess(img, resized_img);
        cv::cvtColor(resized_img, dst, CV_BGR2GRAY);
        detector.detect(dst, keypoints);
        detector.compute(dst, keypoints, descriptor);
        featuresUnclustered.push_back(descriptor);
        std::cout << i << " features" << std::endl;
    }
    
    int dictionarySize=50;
    //define Term Criteria
    cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
    //retries number
    int retries=1;
    //necessary flags
    int flags=cv::KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    cv::BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    //cluster the feature vectors
    cv::Mat dictionary=bowTrainer.cluster(featuresUnclustered);
    //store the vocabulary
    cv::FileStorage fs("dictionary.yml", cv::FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();
}

void BofSurf::ImgResizeProcess(const cv::Mat& img, cv::Mat& resized_img) {
    float k = 1.0;
    if (img.rows < 128 && img.cols < 128) {
        k = ceil(float(128 / img.rows));
    } else if (img.rows > 128 && img.cols > 128) {
        k = 1 / ceil(float(img.rows / 128));
    }
    cv::resize(img, resized_img, cv::Size(int(k * img.rows), int(k * img.cols)));
}

void BofSurf::BoFDescriptor(const std::string& img_path) {
    cv::Mat dictionary;
    const int minHessian = 400; //Hessian Threshold
    cv::FileStorage fs("dictionary.yml", cv::FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();
    cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher);
    cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(minHessian));
    cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor(minHessian));
    cv::BOWImgDescriptorExtractor bowDE(extractor,matcher);
    cv::Mat udict;
    bowDE.setVocabulary(dictionary);
    cv::FileStorage fs1("descriptor.yml", cv::FileStorage::WRITE);
    std::vector<cv::String> image_dataset;
    cv::Mat img, resized_img, dst;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat bowDescriptor;
    cv::glob(img_path + "/*.jpg", image_dataset);
    for (int i = 0; i < image_dataset.size(); i++) {
        img = cv::imread(image_dataset[i]);
        ImgResizeProcess(img, resized_img);
        cv::cvtColor(resized_img, dst, CV_BGR2GRAY);
        cv::imwrite("1.jpg", dst);
        detector->detect(dst, keypoints);
        bowDE.compute(dst, keypoints, bowDescriptor);
        std::cout << i << " descriptors" << std::endl;
        fs1 << "jpg-" + cv::format("%d", i) << bowDescriptor;
    }			
    fs1.release();
}