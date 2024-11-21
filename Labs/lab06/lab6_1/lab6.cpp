#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;

void detectAndSaveFeatures(const std::string& inputPath, const std::string& outputPath) {
    // Carregar imagem
    Mat image = imread(inputPath, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Erro ao carregar a imagem: " << inputPath << std::endl;
        return;
    }

    // Conversão para escala de cinza
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Detector de Good Features to Track
    std::vector<Point2f> corners;
    goodFeaturesToTrack(grayImage, corners, 100, 0.01, 10);

    // Marcar os pontos detectados
    for (const auto& corner : corners) {
        circle(image, corner, 5, Scalar(0, 255, 0), -1);
    }

    // Detector SURF
    Ptr<SURF> surf = SURF::create(400);
    std::vector<KeyPoint> keypoints;
    Mat descriptors;

    surf->detectAndCompute(grayImage, noArray(), keypoints, descriptors);

    // Desenhar os keypoints na imagem
    Mat imageWithKeypoints;
    drawKeypoints(image, keypoints, imageWithKeypoints, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Salvar imagem com as features detectadas
    imwrite(outputPath, imageWithKeypoints);
    std::cout << "Imagem salva com features detectadas: " << outputPath << std::endl;
}

int main() {
    // Caminhos das imagens
    std::vector<std::string> inputPaths = {"imagem1.jpg", "imagem2.jpg"};
    std::vector<std::string> outputPaths = {"imagem1_features.jpg", "imagem2_features.jpg"};

    for (size_t i = 0; i < inputPaths.size(); ++i) {
        detectAndSaveFeatures(inputPaths[i], outputPaths[i]);
    }

    // Processar frames de vídeo
    VideoCapture video("video.mp4");
    if (!video.isOpened()) {
        std::cerr << "Erro ao abrir o vídeo!" << std::endl;
        return -1;
    }

    int frameIndex = 0;
    while (true) {
        Mat frame;
        video >> frame;

        if (frame.empty())
            break;

        std::string frameOutputPath = "frame_" + std::to_string(frameIndex++) + "_features.jpg";
        detectAndSaveFeatures(frame, frameOutputPath);
    }

    return 0;
}
