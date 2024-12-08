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

    // Parâmetros de filtro
    int filterType = 0;  // Tipo de filtro: 0 = original, 1 = mediano, 2 = blur, 3 = gaussiano, 4 = bilateral
    int kernelSize = 3;  // Tamanho inicial do kernel
    std::string filterName = "Original";  // Nome do filtro

    // Definir limites para o tamanho do kernel
    const int minKernelSize = 3;
    const int maxKernelSize = 15;

    // Processar frames do stream
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Erro ao ler o frame do fluxo HLS." << std::endl;
            break;
        }

        // Aplicar aumento de contraste e brilho
        frame.convertTo(frame, -1, alpha, beta);

        // Convert to grayscale (necessário para o Canny)
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Detecção de bordas usando Canny
        cv::Mat edges;
        cv::Canny(grayFrame, edges, 100, 200);  // Ajuste os limiares conforme necessário

        // Transformar as bordas em uma imagem colorida para sobrepor na imagem original
        cv::Mat edgesColored;
        cv::cvtColor(edges, edgesColored, cv::COLOR_GRAY2BGR);

        // Combinar as bordas detectadas com a imagem original
        cv::Mat combinedFrame;
        cv::addWeighted(frame, 0.8, edgesColored, 0.2, 0, combinedFrame);

        // Aplicar o filtro de acordo com o tipo selecionado
        cv::Mat filteredFrame = combinedFrame.clone();
        switch (filterType) {
            case 1:
                cv::medianBlur(combinedFrame, filteredFrame, kernelSize);
                filterName = "Mediano";
                break;
            case 2:
                cv::blur(combinedFrame, filteredFrame, cv::Size(kernelSize, kernelSize));
                filterName = "Blur";
                break;
            case 3:
                cv::GaussianBlur(combinedFrame, filteredFrame, cv::Size(kernelSize, kernelSize), 0);
                filterName = "Gaussiano";
                break;
            case 4:
                cv::bilateralFilter(combinedFrame, filteredFrame, kernelSize, 75, 75);
                filterName = "Bilateral";
                break;
            default:
                filterName = "Original";
        }

        // Detecção com YOLO
        std::vector<cv::Mat> outs;
        cv::Mat blob = cv::dnn::blobFromImage(filteredFrame, 1 / 255.0, cv::Size(608, 608), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        net.forward(outs, net.getUnconnectedOutLayersNames());

        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        // Processar saídas do modelo YOLO
        for (const auto& out : outs) {
            for (int i = 0; i < out.rows; i++) {
                float* data = (float*)out.data + i * out.cols;
                cv::Mat scores = cv::Mat(1, out.cols - 5, CV_32F, data + 5);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int classId = classIdPoint.x;

                // Filtro por classe de veículos
                if (std::find(vehicleClassIds.begin(), vehicleClassIds.end(), classId) != vehicleClassIds.end() && confidence > confidenceThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    boxes.push_back(cv::Rect(left, top, width, height));
                    classIds.push_back(classId);
                    confidences.push_back((float)confidence);
                }
            }
        }

        // Reduzir sobreposição de caixas com Non-Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

        // Desenhar as caixas e rótulos
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            int classId = classIds[idx];
            float confidence = confidences[idx];
            cv::rectangle(filteredFrame, box, cv::Scalar(0, 255, 0), 2);
            std::string label = classNames[classId] + ": " + std::to_string(confidence);
            cv::putText(filteredFrame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Exibir o resultado
        cv::imshow("Deteccao de Veiculos", filteredFrame);

        // Controlar o loop com teclas
        char c = (char)cv::waitKey(1);
        if (c == 27) {  // Pressione ESC para sair
            break;
        } else if (c == 'n') {  // Trocar para o próximo stream de HLS
            current_url_index = (current_url_index + 1) % hls_urls.size();
            cap.open(hls_urls[current_url_index]);
            std::cout << "Mudando para o stream: " << cameraLocations[current_url_index] << std::endl;
        }
    }

    return 0;
}
