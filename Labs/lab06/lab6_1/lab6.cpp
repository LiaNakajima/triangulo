#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

void detectShiTomasi(const cv::Mat& src, const std::string& outputPath) {
    cv::Mat gray, output = src.clone();
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Detect good features to track
    std::vector<cv::Point2f> corners;
    int maxCorners = 50;
    double qualityLevel = 0.01;
    double minDistance = 10;
    cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);

    // Draw corners on the image
    for (const auto& corner : corners) {
        cv::circle(output, corner, 5, cv::Scalar(0, 255, 0), -1);
    }

    // Save the output image
    cv::imwrite(outputPath, output);
}

void detectFeatures(const cv::Mat& src, const std::string& outputPath) {
    cv::Mat gray, output = src.clone();
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Create feature detector
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray, keypoints);

    // Draw keypoints on the image
    cv::drawKeypoints(src, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Save the output image
    cv::imwrite(outputPath, output);
}

int main(int argc, char** argv) {
    // Check for input arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_or_video_path>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    cv::VideoCapture cap;
    bool isVideo = false;

    // Check if input is an image or video
    if (inputPath.find(".mp4") != std::string::npos || inputPath.find(".avi") != std::string::npos) {
        cap.open(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file!" << std::endl;
            return -1;
        }
        isVideo = true;
    } else {
        cv::Mat image = cv::imread(inputPath);
        if (image.empty()) {
            std::cerr << "Error: Cannot open image file!" << std::endl;
            return -1;
        }

        // Apply detection on single image
        detectShiTomasi(image, "output_shitomasi.jpg");
        detectFeatures(image, "output_features.jpg");
        std::cout << "Processed image saved as output_shitomasi.jpg and output_features.jpg" << std::endl;
        return 0;
    }

    // Process video frame by frame
    cv::Mat frame;
    int frameCount = 0;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        std::string shiTomasiOutput = "frame_" + std::to_string(frameCount) + "_shitomasi.jpg";
        std::string featuresOutput = "frame_" + std::to_string(frameCount) + "_features.jpg";

        detectShiTomasi(frame, shiTomasiOutput);
        detectFeatures(frame, featuresOutput);
        frameCount++;
    }

    std::cout << "Processed video frames saved as frame_<n>_shitomasi.jpg and frame_<n>_features.jpg" << std::endl;
    return 0;
}
