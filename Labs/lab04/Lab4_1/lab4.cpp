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
    // Carrega a imagem em cores
    Mat img = imread("foto.jpg");

    // Verifica se a imagem foi carregada com sucesso
    if (img.empty()) {
        cout << "Erro ao carregar a imagem!" << endl;
        return -1;
    }

    // Converte a imagem para tons de cinza
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Calcula e salva o histograma da imagem em tons de cinza
    plotHistogram(gray, "histograma_original.png");

    // Realiza a equalização do histograma
    Mat equalized;
    equalizeHist(gray, equalized);

    // Calcula e salva o histograma da imagem equalizada
    plotHistogram(equalized, "histograma_equalizado.png");

    // Exibe as imagens
    imshow("Imagem Original em Cinza", gray);
    imshow("Imagem Equalizada", equalized);

    // Aguarda uma tecla para salvar as imagens e os histogramas
    cout << "Pressione qualquer tecla para salvar as imagens..." << endl;
    waitKey(0);

    // Salva as imagens
    imwrite("imagem_cinza.png", gray);
    imwrite("imagem_equalizada.png", equalized);

    cout << "Imagens e histogramas salvos com sucesso!" << endl;

    return 0;
}
