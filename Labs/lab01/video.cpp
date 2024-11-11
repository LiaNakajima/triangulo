#include <opencv2/opencv.hpp>
 
#include <iostream>
 
using namespace cv;
using namespace std;

const int FPS = 24;
const int TIME = 5 * FPS;

int main(int argc, char** argv)
{
    VideoCapture vid(0);

    Size frame_size(static_cast<int>(vid.get(3)), static_cast<int>(vid.get(4)));
    VideoWriter output("video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), FPS, frame_size);

    bool take = false;
    int count = 0;
    while (vid.isOpened())
    {
        Mat frame;
        if (!vid.read(frame))
        {
            cout << "Video camera is disconnected" << endl;
            break;
        }

        if (take)
        {
            output.write(frame);
            if (++count == TIME)
                break;
        }
        else if (waitKey(50) == ' ')
            take = true;
        else
            imshow("Preview", frame);
        
    }
    vid.release();
    output.release();

    return 0;
}