#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::string modelConfiguration = "yolov4-tiny.cfg";  // Usando yolov4-tiny para melhorar a performance
    std::string modelWeights = "yolov4-tiny.weights";    // Arquivo de pesos correspondente
    std::string classFile = "coco.names";  // Arquivo com as classes, pode ser "coco.names" ou outro

    // Carregar as classes de objetos
    std::vector<std::string> classNames;
    std::ifstream classNamesFile(classFile);
    std::string line;
    while (std::getline(classNamesFile, line)) {
        classNames.push_back(line);
    }

    // Carregar o modelo YOLO
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);  // Usar OpenCV para processamento
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Usar a CPU (ou DNN_TARGET_CUDA para GPU)

    // Lista de URLs HLS
    std::vector<std::string> hls_urls = {
        "https://65c30e1a8f106ed6.mediapackage.sa-east-1.amazonaws.com/out/v1/7f8bc744f37a46d79b9bce69566097c8/CMAF_HLS/index.m3u8",
        "https://65c30e1a8f106ed6.mediapackage.sa-east-1.amazonaws.com/out/v1/ca56764c05ca4b6fb19cc20149af9b79/CMAF_HLS/index.m3u8",
        "https://65c30e1a8f106ed6.mediapackage.sa-east-1.amazonaws.com/out/v1/c0c1d7b2ffad4d529e6870f43b23668d/CMAF_HLS/index.m3u8",
        "https://65c30e1a8f106ed6.mediapackage.sa-east-1.amazonaws.com/out/v1/f4774d456f2d4a0494cc37e7b6d155f0/CMAF_HLS/index.m3u8",
        "https://3d30a042367404fa.mediapackage.sa-east-1.amazonaws.com/out/v1/ea0b2a29932c4ba8b3459b7403eac5d3/CMAF_HLS/index.m3u8",
        "https://3d30a042367404fa.mediapackage.sa-east-1.amazonaws.com/out/v1/5cab102e0357405886be434dac4bd373/CMAF_HLS/index.m3u8",
        "https://3d30a042367404fa.mediapackage.sa-east-1.amazonaws.com/out/v1/1daf9b25f82441f7b581caa21b76c6e0/CMAF_HLS/index.m3u8",
        "https://3d30a042367404fa.mediapackage.sa-east-1.amazonaws.com/out/v1/6c9dc97655114f5ebb776047bf011172/CMAF_HLS/index.m3u8"
    };

    // Lista com os locais das câmeras
    std::vector<std::string> cameraLocations = {
        "M 110 - Taubate, SP",
        "KM 230 - Guarulhos, SP",
        "KM 156 - Sao Jose dos Campos, SP",
        "KM 276 - Barra Mansa, RJ",
        "KM 202 - Aruja, SP",
        "KM 179 - Guararema, SP",
        "KM 124 - Cacapava, SP",
        "KM 307 - Resende, RJ"
    };

    int current_url_index = 0;  // Índice do link HLS atual

    // Abrir o stream de vídeo com o primeiro link
    cv::VideoCapture cap(hls_urls[current_url_index]);
    if (!cap.isOpened()) {
        std::cerr << "Erro: Não foi possível abrir o stream HLS." << std::endl;
        return -1;
    }

    // Filtro para melhorar a visibilidade (ajustar brilho e contraste)
    float alpha = 1.5;  // Aumenta o contraste
    float beta = 0;  // Ajusta o brilho
    float confidenceThreshold = 0.3;  // Limite de confiança
    float nmsThreshold = 0.4;  // Threshold de NMS para reduzir falsas detecções

    // Focar apenas em algumas classes específicas, por exemplo, carros (ID 2), motos (ID 3), caminhões (ID 7)
    std::vector<int> vehicleClassIds = {2, 3, 5, 7};  // IDs de carros, motos, caminhões, ônibus

    // Processar frames do stream
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Erro ao ler o frame do fluxo HLS." << std::endl;
            break;
        }

        // Aplicar aumento de contraste e brilho
        frame.convertTo(frame, -1, alpha, beta);

        // Reduzir a resolução da imagem para melhorar o desempenho
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(320, 320), cv::Scalar(), true, false);

        // Passar o blob pela rede
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // Variáveis para processar as detecções
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Processar as detecções
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j) {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confidenceThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    cv::Rect box(centerX - width / 2, centerY - height / 2, width, height);

                    // Verificar se a classe é de interesse (carros, motos, etc.)
                    int classId = classIdPoint.x;
                    if (std::find(vehicleClassIds.begin(), vehicleClassIds.end(), classId) != vehicleClassIds.end()) {
                        boxes.push_back(box);
                        confidences.push_back((float)confidence);
                        classIds.push_back(classId);
                    }
                }
                data += outs[i].cols;
            }
        }

        // Aplicar NMS (Non-Maximum Suppression)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

        // Desenhar as caixas
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, classNames[classIds[idx]], box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        }

        // Adicionar o texto do local da câmera
        std::string cameraLocation = cameraLocations[current_url_index];
        cv::putText(frame, cameraLocation, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        // Exibir o frame com as detecções
        cv::imshow("YOLO Vehicle Detection", frame);

        // Aguardar a tecla pressionada
        char key = cv::waitKey(1);

        // Se pressionar 'q', sai do loop
        if (key == 'q') {
            break;
        }

        // Se pressionar 'n', alterna para o próximo link HLS
        if (key == 'n') {
            current_url_index = (current_url_index + 1) % hls_urls.size();
            cap.release();  // Libera o fluxo atual

            // Tenta abrir o próximo fluxo
            if (!cap.open(hls_urls[current_url_index])) {
                std::cerr << "Erro ao abrir o fluxo HLS: " << hls_urls[current_url_index] << std::endl;
                break;
            }

            std::cout << "Mudando para o fluxo: " << hls_urls[current_url_index] << std::endl;
        }

        // Se pressionar 'p', alterna para o fluxo anterior
        if (key == 'p') {
            current_url_index = (current_url_index - 1 + hls_urls.size()) % hls_urls.size();
            cap.release();  // Libera o fluxo atual

            // Tenta abrir o fluxo anterior
            if (!cap.open(hls_urls[current_url_index])) {
                std::cerr << "Erro ao abrir o fluxo HLS: " << hls_urls[current_url_index] << std::endl;
                break;
            }

            std::cout << "Mudando para o fluxo: " << hls_urls[current_url_index] << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
