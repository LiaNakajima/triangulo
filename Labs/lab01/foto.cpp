#include <opencv2/opencv.hpp>
 
#include <iostream>
 
using namespace cv;
using namespace std;

const int MAX = 5;
const int INTERVAL = 500;

int main(int argc, char** argv)
{
    VideoCapture vid(0);

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

        imshow("Preview", frame);

        if (waitKey(INTERVAL) == ' ')
            take = true;
        if (take)
        {
            count++;
            imwrite("foto" + to_string(count) + ".jpg", frame);
            if (count == MAX)
                break;
        }
        
    }
    vid.release();

    return 0;
}