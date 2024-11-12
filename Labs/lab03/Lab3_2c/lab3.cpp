#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <ctime>
#include <sstream>

using namespace cv;
using namespace std;

const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
const String window_detection_blur_name = "Object Detection Blur";
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
int kernel_size = 11;

// Função para gerar nomes únicos para arquivos
string generateFilename(const string &baseName, const string &extension) {
    time_t now = time(0);
    tm *ltm = localtime(&now);
    stringstream ss;
    ss << baseName << "_" << ltm->tm_year + 1900 << "_" 
       << ltm->tm_mon + 1 << "_" << ltm->tm_mday << "_" 
       << ltm->tm_hour << "_" << ltm->tm_min << "_" 
       << ltm->tm_sec << extension;
    return ss.str();
}

int main(int argc, char* argv[])
{
    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera." << endl;
        return -1;
    }

    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    namedWindow(window_detection_blur_name);

    // Trackbar para definir limiar
    createTrackbar("Min Threshold:", window_detection_name, &lowThreshold, max_lowThreshold);

    Mat frame, frame_blur, frame_edges;
    VideoWriter video_writer;
    bool isRecording = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        
        // Aplicando filtro gaussiano
        GaussianBlur(frame, frame_blur, Size(kernel_size, kernel_size), 0);

        // Aplicando o detector de Canny
        Canny(frame_blur, frame_edges, lowThreshold, kernel_size);

        // Exibindo as janelas
        imshow(window_capture_name, frame);
        imshow(window_detection_name, frame_edges);
        imshow(window_detection_blur_name, frame_blur);

        char key = (char) waitKey(30);
        
        if (key == 'q' || key == 27) {  // Sair
            break;
        } else if (key == 's') {  // Salvar imagem
            imwrite(generateFilename("captura", ".png"), frame);
            cout << "Imagem salva." << endl;
        } else if (key == 'k') {  // Iniciar gravação de vídeo
            if (!isRecording) {
                string filename = generateFilename("video", ".avi");
                int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
                double fps = 30.0;
                Size frame_size(frame.cols, frame.rows);
                video_writer.open(filename, codec, fps, frame_size, true);
                
                if (!video_writer.isOpened()) {
                    cerr << "Erro ao abrir o arquivo de vídeo para gravação." << endl;
                    continue;
                }
                
                isRecording = true;
                cout << "Gravação de vídeo iniciada." << endl;
            }
        } else if (key == 'h') {  // Parar gravação de vídeo
            if (isRecording) {
                video_writer.release();
                isRecording = false;
                cout << "Gravação de vídeo encerrada." << endl;
            }
        }

        // Gravar o quadro atual se a gravação estiver ativa
        if (isRecording) {
            video_writer.write(frame);
        }
    }

    // Liberar o objeto VideoWriter se ainda estiver gravando
    if (isRecording) {
        video_writer.release();
    }

    return 0;
}
