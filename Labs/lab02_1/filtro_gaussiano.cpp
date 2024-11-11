#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/lexical_cast.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    int kernel;
    if (argc == 3) {
        kernel = boost::lexical_cast<int>(argv[2]);
    }
    else {
       cout << "Quantidade incorreta de argumentos!" << endl;
       return -1;
    }
    string kernel_str = to_string(kernel) + "x" + to_string(kernel);

    Mat image = imread(argv[1]);
    if(image.empty()) {
        cout << "Erro ao abrir a imagem!" << endl;
        return -1;
    }

    Mat result;
    // Aplicando o filtro Gaussiano (kernel 3x3, sigmaX = 0)
    GaussianBlur(image, result, Size(kernel, kernel), 0);

    // Exibindo e salvando o resultado
    namedWindow("Filtro Gaussiano - " + kernel_str, WINDOW_AUTOSIZE);
    imshow("Filtro Gaussiano - " + kernel_str, result);
    imwrite("gaussian_" + kernel_str + ".jpg", result); // Salvando o resultado

    waitKey(0);
    return 0;
}
