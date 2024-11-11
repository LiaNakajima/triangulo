#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0); // Acessa a webcam
    if(!cap.isOpened()) {
        cout << "Erro ao abrir a webcam!" << endl;
        return -1;
    }

    Mat frame, filtered_frame;
    char key;

    while(true) {
        cap >> frame; // Captura frame da webcam
        if(frame.empty()) break;

        // Aplica o filtro desejado (por exemplo, blur com kernel 5x5)
        GaussianBlur(frame, filtered_frame, Size(5, 5), 0);

        imshow("Webcam - Filtro Gaussiano 5x5", filtered_frame);

        key = waitKey(30); // Aguarda tecla por 30 ms
        if(key == 's') {
            imwrite("imagem_filtrada.jpg", filtered_frame); // Salva a imagem
        } else if(key == 27) {
            break; // Sai do loop se pressionar Esc
        }
    }

    return 0;
}
