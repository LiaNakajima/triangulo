#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, src_gray;
int maxCorners = 23;
int maxTrackbar = 100;
RNG rng(12345);

const char* source_window = "Live Feed";
const char* features_window = "Features Detected";

void goodFeaturesToTrack_Demo(Mat& frame, Mat& outputFrame);

int main(int argc, char** argv)
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "Could not open the webcam!\n" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    Size frame_size(frame_width, frame_height);

    VideoWriter videoWriter("features_detected.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame_size);

    if (!videoWriter.isOpened())
    {
        cout << "Could not open the video file for saving!\n" << endl;
        return -1;
    }

    namedWindow(source_window);
    namedWindow(features_window);
    createTrackbar("Max corners:", features_window, &maxCorners, maxTrackbar);

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            cout << "No frame captured. Exiting...\n" << endl;
            break;
        }

        Mat outputFrame;
        goodFeaturesToTrack_Demo(frame, outputFrame);

        imshow(source_window, frame);
        imshow(features_window, outputFrame);

        videoWriter.write(outputFrame);

        if (waitKey(1) == 'q')
        {
            cout << "Exiting...\n" << endl;
            break;
        }
    }

    cap.release();
    videoWriter.release();
    destroyAllWindows();
    return 0;
}

void goodFeaturesToTrack_Demo(Mat& frame, Mat& outputFrame)
{
    // Convert to grayscale
    cvtColor(frame, src_gray, COLOR_BGR2GRAY);

    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    outputFrame = frame.clone();

    // Detect corners
    goodFeaturesToTrack(src_gray,
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        Mat(),
                        blockSize,
                        gradientSize,
                        useHarrisDetector,
                        k);

    cout << "** Number of corners detected: " << corners.size() << endl;

    int radius = 4;
    for (size_t i = 0; i < corners.size(); i++)
    {
        circle(outputFrame, corners[i], radius, Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
    }
}
