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

    // Variáveis para gravação de vídeo
    bool isRecording = false;
    cv::VideoWriter videoWriter;
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // Processar frames do stream
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Erro ao ler o frame do fluxo HLS." << std::endl;
            break;
        }

        // Aplicar aumento de contraste e brilho
        frame.convertTo(frame, -1, alpha, beta);

        // Aplicar o filtro de acordo com o tipo selecionado
        cv::Mat filteredFrame = frame.clone();
        switch (filterType) {
            case 1:
                cv::medianBlur(frame, filteredFrame, kernelSize);
                filterName = "Mediano";
                break;
            case 2:
                cv::blur(frame, filteredFrame, cv::Size(kernelSize, kernelSize));
                filterName = "Blur";
                break;
            case 3:
                cv::GaussianBlur(frame, filteredFrame, cv::Size(kernelSize, kernelSize), 0);
                filterName = "Gaussiano";
                break;
            case 4:
                cv::bilateralFilter(frame, filteredFrame, kernelSize, 75, 75);
                filterName = "Bilateral";
                break;
            default:
                filterName = "Original";
        }

        // Detecção com YOLO
        std::vector<cv::Mat> outs;
        cv::Mat blob = cv::dnn::blobFromImage(filteredFrame, 1 / 255.0, cv::Size(832, 832), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        net.forward(outs, net.getUnconnectedOutLayersNames());

        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        // Processar saídas do modelo YOLO
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
            cv::rectangle(filteredFrame, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(filteredFrame, classNames[classIds[idx]], box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        }

        // Adicionar o texto do local da câmera
        std::string cameraLocation = cameraLocations[current_url_index];
        cv::putText(filteredFrame, cameraLocation, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        // Exibir os parâmetros ajustáveis na tela em várias linhas
        int lineHeight = 30;
        int yOffset = 60;
        cv::putText(filteredFrame, "Parametros Ajustaveis:", cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        yOffset += lineHeight;

        cv::putText(filteredFrame, "Filtro: " + filterName, cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        yOffset += lineHeight;
        cv::putText(filteredFrame, "Kernel: " + std::to_string(kernelSize), cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        yOffset += lineHeight;
        cv::putText(filteredFrame, "Brilho: " + std::to_string(beta), cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        yOffset += lineHeight;
        cv::putText(filteredFrame, "Contraste: " + std::to_string(alpha), cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        yOffset += lineHeight;
        cv::putText(filteredFrame, "Confianca: " + std::to_string(confidenceThreshold), cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        yOffset += lineHeight;
        cv::putText(filteredFrame, "NMS: " + std::to_string(nmsThreshold), cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        // Exibir o frame com as detecções
        cv::imshow("YOLO Vehicle Detection", filteredFrame);

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

        // Alterar o filtro com as teclas de atalho
        if (key == '1') { filterType = 1; }
        if (key == '2') { filterType = 2; }
        if (key == '3') { filterType = 3; }
        if (key == '4') { filterType = 4; }
        if (key == '0') { filterType = 0; }  // Resetar para filtro original

        // Ajustar o tamanho do kernel com as teclas 'K' e 'J'
        if (key == 'K' && kernelSize < maxKernelSize) {
            kernelSize += 2;  // Aumenta o kernel
        }
        if (key == 'J' && kernelSize > minKernelSize) {
            kernelSize -= 2;  // Diminui o kernel
        }

        // Ajustar o brilho com 'b' (diminuir) e 'B' (aumentar)
        if (key == 'b') {
            beta -= 5;  // Diminuir brilho
        }
        if (key == 'B') {
            beta += 5;  // Aumentar brilho
        }

        // Ajustar o contraste com 'c' (diminuir) e 'C' (aumentar)
        if (key == 'c') {
            alpha -= 0.1;  // Diminuir contraste
        }
        if (key == 'C') {
            alpha += 0.1;  // Aumentar contraste
        }

        // Ajustar a confiança com as teclas 'v' e 'V'
        if (key == 'v') {
            confidenceThreshold -= 0.05;  // Diminuir confiança
        }
        if (key == 'V') {
            confidenceThreshold += 0.05;  // Aumentar confiança
        }

        // Ajustar o threshold do NMS com as teclas 'm' e 'M'
        if (key == 'm') {
            nmsThreshold -= 0.05;  // Diminuir NMS
        }
        if (key == 'M') {
            nmsThreshold += 0.05;  // Aumentar NMS
        }

        // Salvar a imagem com a tecla 's'
        if (key == 's') {
            cv::imwrite("captured_image.jpg", filteredFrame);  // Salvar a imagem
            std::cout << "Imagem salva como captured_image.jpg" << std::endl;
            cv::imwrite("captured_image_original.jpg", frame);  // Salvar a imagem
            std::cout << "Imagem salva como captured_image_original.jpg" << std::endl;
        }

        // Gravar vídeo com 'r'
        if (key == 'r') {
            if (!isRecording) {
                // Começar a gravação
                videoWriter.open("recorded_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frameWidth, frameHeight));
                isRecording = true;
                std::cout << "Iniciando gravação de vídeo..." << std::endl;
            } else {
                // Parar a gravação
                videoWriter.release();
                isRecording = false;
                std::cout << "Gravação de vídeo parada." << std::endl;
            }
        }

        // Gravar o frame se a gravação estiver ativa
        if (isRecording) {
            videoWriter.write(filteredFrame);
        }
    }

    cap.release();  // Liberar o fluxo de vídeo
    cv::destroyAllWindows();  // Fechar todas as janelas
    return 0;
}
