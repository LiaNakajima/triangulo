#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

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

    // Caminho do vídeo específico
    std::string videoFilePath = "/home/leonardo/Documentos/leosantos/Traffic_Car_Counter_cpp_openCV_YOLOv3/video/2.mp4"; // Substitua pelo caminho do seu vídeo

    // Abrir o vídeo específico
    cv::VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) {
        std::cerr << "Erro: Não foi possível abrir o vídeo." << std::endl;
        return -1;
    }

    // Filtro para melhorar a visibilidade (ajustar brilho e contraste)
    float alpha = 1.5;  // Aumenta o contraste
    float beta = 0;  // Ajusta o brilho
    float confidenceThreshold = 0.3;  // Limite de confiança
    float nmsThreshold = 0.4;  // Threshold de NMS para reduzir falsas detecções

    // Focar apenas em algumas classes específicas, por exemplo, carros (ID 2), motos (ID 3), caminhões (ID 7)
    std::vector<int> vehicleClassIds = {2, 3, 5, 7};  // IDs de carros, motos, caminhões, ônibus

    // Variáveis para rastreamento
    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect> trackedBoxes;
    std::map<int, cv::Ptr<cv::Tracker>> vehicleTrackers;  // Mapa de ID para tracker
    int vehicleID = 0;  // ID para novos veículos

    // Processar frames do vídeo
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Erro ao ler o frame do vídeo." << std::endl;
            break;
        }

        // Aplicar aumento de contraste e brilho
        frame.convertTo(frame, -1, alpha, beta);

        // Detecção com YOLO
        std::vector<cv::Mat> outs;
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(608, 608), cv::Scalar(0, 0, 0), true, false);
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

        // Adicionar novos rastreadores para os veículos detectados
        for (int i : indices) {
            cv::Rect box = boxes[i];
            bool newVehicle = true;

            // Verificar se o veículo já está sendo rastreado
            for (auto& trackerPair : vehicleTrackers) {
                cv::Rect trackedBox;
                if (trackerPair.second->update(frame, trackedBox)) {
                    newVehicle = false;
                    break;
                }
            }

            if (newVehicle) {
                // Criar um novo rastreador para o veículo
                cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
                tracker->init(frame, box);
                vehicleTrackers[vehicleID++] = tracker;  // Atribuir ID único ao veículo
            }
        }

        // Atualizar todos os rastreadores
        for (auto& trackerPair : vehicleTrackers) {
            cv::Rect trackedBox;
            bool success = trackerPair.second->update(frame, trackedBox);
            if (success) {
                // Desenhar a caixa de rastreamento
                cv::rectangle(frame, trackedBox, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "ID: " + std::to_string(trackerPair.first), trackedBox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
            }
        }

        // Exibir o número de veículos rastreados
        cv::putText(frame, "Veiculos Detectados: " + std::to_string(vehicleTrackers.size()), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        // Exibir o frame com as detecções e rastreamento
        cv::imshow("YOLO Vehicle Detection and Tracking", frame);

        // Pressionar ESC para sair
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
