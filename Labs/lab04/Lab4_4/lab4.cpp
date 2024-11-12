#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Inicializa a captura da webcam (câmera padrão - índice 0)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Erro ao acessar a webcam!" << endl;
        return -1;
    }

    Mat frame, equalizedImage;
    vector<Mat> channels(3);

    cout << "Pressione qualquer tecla para salvar a imagem equalizada..." << endl;

    while (true) {
        // Captura o frame da webcam
        cap >> frame;

        // Verifica se o frame foi capturado com sucesso
        if (frame.empty()) {
            cout << "Erro ao capturar frame!" << endl;
            break;
        }

        // Separa a imagem nos três canais de cor (B, G, R)
        split(frame, channels);

        // Aplica a equalização do histograma em cada canal de cor separadamente
        equalizeHist(channels[0], channels[0]); // Canal Azul
        equalizeHist(channels[1], channels[1]); // Canal Verde
        equalizeHist(channels[2], channels[2]); // Canal Vermelho

        // Junta os canais equalizados para formar a imagem colorida final
        merge(channels, equalizedImage);

        // Exibe a imagem original e a imagem equalizada
        imshow("Imagem Original", frame);
        imshow("Imagem Equalizada", equalizedImage);

        // Verifica se uma tecla foi pressionada
        if (waitKey(30) >= 0) {
            // Salva a imagem equalizada
            imwrite("imagem_equalizada_colorida.png", equalizedImage);

            cout << "Imagem equalizada colorida salva com sucesso!" << endl;
            break;
        }
    }

    // Libera a captura da webcam e fecha as janelas
    cap.release();
    destroyAllWindows();

    return 0;
}
