#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

void detectAndDisplay(cv::Mat& frame, cv::CascadeClassifier& cascade) {
    std::vector<cv::Rect> objects;
    cv::Mat frameGray;

    // Converter para escala de cinza para melhor desempenho
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frameGray, frameGray);

    // Detectar objetos (rostos, dependendo do Haarcascade usado)
    cascade.detectMultiScale(frameGray, objects, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    // Desenhar retângulos ao redor dos objetos detectados
    for (const auto& obj : objects) {
        cv::rectangle(frame, obj, cv::Scalar(255, 0, 0), 2); // Azul para detecção
    }

    // Exibir a imagem com as detecções em tempo real
    cv::imshow("Detecção ao Vivo", frame);
}

int main() {
    // Carregar o modelo Haarcascade
    std::string cascadePath = "haarcascade_frontalface_default.xml"; // Substitua pelo modelo desejado
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        std::cerr << "Erro ao carregar o modelo Haarcascade!" << std::endl;
        return -1;
    }

    // Abrir a webcam
    cv::VideoCapture capture(0); // '0' se refere à webcam padrão
    if (!capture.isOpened()) {
        std::cerr << "Erro ao acessar a webcam!" << std::endl;
        return -1;
    }

    std::cout << "Pressione 's' para salvar uma imagem ou 'q' para sair." << std::endl;

    cv::Mat frame;
    while (capture.read(frame)) {
        // Detectar e exibir objetos na janela ao vivo
        detectAndDisplay(frame, cascade);

        // Esperar por uma tecla
        char key = cv::waitKey(30);
        if (key == 's') {
            // Salvar a imagem com as detecções
            static int savedCount = 0;
            std::string filename = "detected_" + std::to_string(savedCount++) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Imagem salva: " << filename << std::endl;
        } else if (key == 'q') {
            // Encerrar o programa
            break;
        }
    }

    return 0;
}
