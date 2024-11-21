#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

void detectAndDisplay(cv::Mat& frame, cv::CascadeClassifier& cascade, const std::string& windowName) {
    std::vector<cv::Rect> objects;
    cv::Mat frameGray;

    // Converter para escala de cinza para melhor desempenho
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frameGray, frameGray);

    // Detectar objetos
    cascade.detectMultiScale(frameGray, objects, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    // Desenhar retângulos ao redor dos objetos detectados
    for (const auto& obj : objects) {
        cv::rectangle(frame, obj, cv::Scalar(0, 255, 0), 2); // Verde para rostos
    }

    // Exibir a imagem com as detecções
    cv::imshow(windowName, frame);
}

int main() {
    // Carregar o modelo Haarcascade
    std::string cascadePath = "haarcascade_frontalface_default.xml"; // Substitua pelo modelo desejado
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        std::cerr << "Erro ao carregar o modelo Haarcascade!" << std::endl;
        return -1;
    }

    // Leitura de imagens
    cv::Mat teamImage = cv::imread("team.jpg");  // Substitua pelo caminho da imagem da equipe
    cv::Mat avatarImage = cv::imread("avatars.jpg"); // Substitua pelo caminho da imagem com avatares
    if (teamImage.empty() || avatarImage.empty()) {
        std::cerr << "Erro ao carregar as imagens!" << std::endl;
        return -1;
    }

    // Detectar e exibir rostos/objetos nas imagens
    detectAndDisplay(teamImage, cascade, "Imagem da Equipe");
    detectAndDisplay(avatarImage, cascade, "Imagem dos Avatares");

    // Leitura de vídeo
    cv::VideoCapture videoCapture(0); // Use 0 para webcam ou substitua pelo caminho do vídeo
    if (!videoCapture.isOpened()) {
        std::cerr << "Erro ao acessar o vídeo!" << std::endl;
        return -1;
    }

    std::cout << "Pressione 's' para salvar uma imagem ou 'q' para sair." << std::endl;

    // Processar o vídeo quadro a quadro
    cv::Mat frame;
    while (videoCapture.read(frame)) {
        detectAndDisplay(frame, cascade, "Vídeo");

        char key = cv::waitKey(30);
        if (key == 's') {
            // Salvar a imagem com as detecções
            static int savedCount = 0;
            std::string filename = "detected_" + std::to_string(savedCount++) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Imagem salva: " << filename << std::endl;
        } else if (key == 'q') {
            break;
        }
    }

    return 0;
}
