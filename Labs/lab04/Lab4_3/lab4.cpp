#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Função para plotar e salvar o histograma de uma imagem em tons de cinza
void plotHistogram(const Mat& image, const string& filename) {
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;

    // Calcula o histograma
    calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    // Normaliza o histograma para caber na imagem de exibição
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Desenha o histograma
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
             Scalar(0), 2, 8, 0);
    }

    // Salva a imagem do histograma
    imwrite(filename, histImage);
}

int main() {
    // Inicializa a captura da webcam (câmera padrão - índice 0)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Erro ao acessar a webcam!" << endl;
        return -1;
    }

    Mat frame, gray, equalized, binary;

    cout << "Pressione qualquer tecla para salvar as imagens..." << endl;

    while (true) {
        // Captura o frame da webcam
        cap >> frame;

        // Verifica se o frame foi capturado com sucesso
        if (frame.empty()) {
            cout << "Erro ao capturar frame!" << endl;
            break;
        }

        // Converte o frame para tons de cinza
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Realiza a equalização do histograma
        equalizeHist(gray, equalized);

        // Aplica a limiarização (binarização) após a equalização do histograma
        double threshold_value = 128;  // Valor de limiar, ajustável conforme necessário
        double max_BINARY_value = 255;
        threshold(equalized, binary, threshold_value, max_BINARY_value, THRESH_BINARY);

        // Exibe as imagens ao vivo
        imshow("Imagem em Tons de Cinza", gray);
        imshow("Imagem Equalizada", equalized);
        imshow("Imagem Binária (Limiarizada)", binary);

        // Verifica se uma tecla foi pressionada
        if (waitKey(30) >= 0) {
            // Salva as imagens
            imwrite("imagem_cinza_webcam.png", gray);
            imwrite("imagem_equalizada_webcam.png", equalized);
            imwrite("imagem_binaria_webcam.png", binary);

            // Salva os histogramas
            plotHistogram(gray, "histograma_original_webcam.png");
            plotHistogram(equalized, "histograma_equalizado_webcam.png");

            cout << "Imagens e histogramas salvos com sucesso!" << endl;
            break;
        }
    }

    // Libera a captura da webcam e fecha as janelas
    cap.release();
    destroyAllWindows();

    return 0;
}
