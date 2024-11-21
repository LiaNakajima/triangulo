#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;

void detectAndDrawFeatures(const Mat& frame, Mat& frameWithFeatures) {
    // Conversão para escala de cinza
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    // Detector de Good Features to Track
    std::vector<Point2f> corners;
    goodFeaturesToTrack(grayFrame, corners, 100, 0.01, 10);

    // Copiar o frame original para desenhar as features
    frameWithFeatures = frame.clone();
    for (const auto& corner : corners) {
        circle(frameWithFeatures, corner, 5, Scalar(0, 255, 0), -1);
    }

    // Detector SURF
    Ptr<SURF> surf = SURF::create(400);
    std::vector<KeyPoint> keypoints;
    Mat descriptors;

    surf->detectAndCompute(grayFrame, noArray(), keypoints, descriptors);

    // Desenhar os keypoints
    drawKeypoints(frameWithFeatures, keypoints, frameWithFeatures, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

int main() {
    // Abrir webcam
    VideoCapture webcam(0); // 0 para a webcam padrão
    if (!webcam.isOpened()) {
        std::cerr << "Erro ao abrir a webcam!" << std::endl;
        return -1;
    }

    // Obter as propriedades do vídeo da webcam
    int frameWidth = static_cast<int>(webcam.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(webcam.get(CAP_PROP_FRAME_HEIGHT));
    double fps = webcam.get(CAP_PROP_FPS);
    if (fps == 0) fps = 30; // Valor padrão caso a câmera não informe o FPS

    // Criar VideoWriter para salvar vídeos
    VideoWriter originalVideo("video_original.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameWidth, frameHeight));
    VideoWriter featuresVideo("video_features.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameWidth, frameHeight));

    if (!originalVideo.isOpened() || !featuresVideo.isOpened()) {
        std::cerr << "Erro ao abrir os arquivos de vídeo para gravação!" << std::endl;
        return -1;
    }

    std::cout << "Pressione 'q' para encerrar o programa." << std::endl;

    Mat frame, frameWithFeatures;
    while (true) {
        // Capturar frame da webcam
        webcam >> frame;

        if (frame.empty()) {
            std::cerr << "Erro ao capturar frame da webcam!" << std::endl;
            break;
        }

        // Processar frame para detecção de features
        detectAndDrawFeatures(frame, frameWithFeatures);

        // Exibir imagens ao vivo
        imshow("Webcam - Imagem Original", frame);
        imshow("Webcam - Imagem com Features", frameWithFeatures);

        // Salvar os frames no vídeo
        originalVideo.write(frame);
        featuresVideo.write(frameWithFeatures);

        // Finalizar se a tecla 'q' for pressionada
        if (waitKey(30) == 'q') {
            break;
        }
    }

    // Liberar recursos
    webcam.release();
    originalVideo.release();
    featuresVideo.release();
    destroyAllWindows();

    std::cout << "Vídeos salvos como 'video_original.avi' e 'video_features.avi'." << std::endl;
    return 0;
}
