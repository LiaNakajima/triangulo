#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
 
using namespace cv;

const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
const String window_detection_blur_name = "Object Detection Blur";
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
int kernel_size = 11;

int main(int argc, char* argv[])
{
    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
 
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    namedWindow(window_detection_blur_name);
 
    // Trackbars to set thresholds for HSV values
    createTrackbar( "Min Threshold:", window_detection_name, &lowThreshold, max_lowThreshold );

    Mat frame, frame_blur, frame_edges;
    while (true) {
        cap >> frame;
        if(frame.empty())
        {
            break;
        }
        
        //Aplicando filtro gaussiano
        GaussianBlur(frame, frame_blur, Size(kernel_size, kernel_size), 0);
 
        //Aplicando o detector de CANNY
        Canny(frame_blur, frame_edges, lowThreshold, kernel_size);

        // Show the frames
        imshow(window_capture_name, frame);
        imshow(window_detection_name, frame_edges);
        imshow(window_detection_blur_name, frame_blur);
        
        char key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }
    return 0;
}
