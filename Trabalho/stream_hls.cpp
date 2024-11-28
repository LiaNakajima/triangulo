#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    std::string url = "https://3d30a042367404fa.mediapackage.sa-east-1.amazonaws.com/out/v1/ea0b2a29932c4ba8b3459b7403eac5d3/CMAF_HLS/index.m3u8";


    cv::VideoCapture cap(url);

    if (!cap.isOpened()) {
        std::cerr << "Erro: Não foi possível abrir o stream HLS." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {

        if (!cap.read(frame)) {
            std::cerr << "Erro: Não foi possível ler o frame do stream." << std::endl;
            break;
        }


        cv::imshow("Stream HLS", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }


    cap.release();
    cv::destroyAllWindows();

    return 0;
}