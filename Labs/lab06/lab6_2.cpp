#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

void detectShiTomasi(const cv::Mat& src, cv::Mat& output) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Detect good features to track
    std::vector<cv::Point2f> corners;
    int maxCorners = 50;
    double qualityLevel = 0.01;
    double minDistance = 10;
    cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);

    output = src.clone();
    // Draw corners on the image
    for (const auto& corner : corners) {
        cv::circle(output, corner, 5, cv::Scalar(0, 255, 0), -1);
    }
}

void detectFeatures(const cv::Mat& src, cv::Mat& output) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Create feature detector
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray, keypoints);

    output = src.clone();
    // Draw keypoints on the image
    cv::drawKeypoints(src, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

int main() {
    cv::VideoCapture cap(0); // Open the default camera (webcam)
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the webcam!" << std::endl;
        return -1;
    }

    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Shi-Tomasi Features", cv::WINDOW_NORMAL);
    cv::namedWindow("ORB Features", cv::WINDOW_NORMAL);

    int frameCount = 0;

    while (true) {
        cv::Mat frame, shiTomasiOutput, orbOutput;
        cap >> frame; // Capture a new frame from the webcam

        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame!" << std::endl;
            break;
        }

        // Detect features using Shi-Tomasi
        detectShiTomasi(frame, shiTomasiOutput);

        // Detect features using ORB
        detectFeatures(frame, orbOutput);

        // Show the frames
        cv::imshow("Original", frame);
        cv::imshow("Shi-Tomasi Features", shiTomasiOutput);
        cv::imshow("ORB Features", orbOutput);

        // Save the frames periodically (e.g., every 50 frames)
        if (frameCount % 50 == 0) {
            std::string shiTomasiFile = "webcam_shitomasi_" + std::to_string(frameCount) + ".jpg";
            std::string orbFile = "webcam_orb_" + std::to_string(frameCount) + ".jpg";
            cv::imwrite(shiTomasiFile, shiTomasiOutput);
            cv::imwrite(orbFile, orbOutput);
        }

        frameCount++;

        // Break the loop if the user presses 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
