#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;
 
int maxCorners = 23;
int maxTrackbar = 100;
 
RNG rng(12345);
const char* source_window = "Image";

void detectAndDrawFeatures(const Mat& frame, Mat& frameWithFeatures)
{
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
 
    Mat copy = frame.clone();
 
    goodFeaturesToTrack( grayFrame,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         gradientSize,
                         useHarrisDetector,
                         k );
 
 
    cout << "** Number of corners detected: " << corners.size() << endl;
    int radius = 4;
    for( size_t i = 0; i < corners.size(); i++ )
    {
        circle( copy, corners[i], radius, Scalar(rng.uniform(0,255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED );
    }


 
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