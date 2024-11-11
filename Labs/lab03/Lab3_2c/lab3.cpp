#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;

const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
const String window_detection_blur_name = "Object Detection Blur";
int lowThreshold = 0;
int kernel_size = 11;
const int max_lowThreshold = 100;
const int ratio = 3;
const int FPS = 24;
const int TIME = 5 * FPS;


int main(int argc, char* argv[])
{
    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
    Size frame_size(static_cast<int>(cap.get(3)), static_cast<int>(cap.get(4)));
    VideoWriter output("video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), FPS, frame_size);
 
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    namedWindow(window_detection_blur_name);
 
    // Trackbars to set thresholds for HSV values
    createTrackbar( "Min Threshold:", window_detection_name, &lowThreshold, max_lowThreshold );

    Mat frame, frame_blur, frame_edges;
    int count = 0;
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
        imshow(window_detection_blur_name, frame_blur);
        imshow(window_detection_name, frame_edges);
        
        char key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
        
        if (key == 's'){
            count++;
            imwrite("imagem_original_" + to_string(count) + ".jpg", frame);
            imwrite("imagem_filtrada_" + to_string(count) + ".jpg", frame_blur);
            imwrite("imagem_canny_" + to_string(count) + ".jpg", frame_edges);
            cout << "Imagens salvas" << endl;
        } 

        // if (key == 'k' ){
        //     output.write(frame);
        //     cout << "Gravação iniciada" << endl;

        //     if (++count == TIME)
        //         break;
        // }

    }
    return 0;
}
