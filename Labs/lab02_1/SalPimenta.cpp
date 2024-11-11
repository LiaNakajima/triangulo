#include <iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void Add_salt_pepper_Noise(Mat &srcArr, float pa, float pb )

{    RNG rng; // rand number generator
    int amount1=srcArr.rows*srcArr.cols*pa;
    int amount2=srcArr.rows*srcArr.cols*pb;
    for(int counter=0; counter<amount1; ++counter)
    {
     srcArr.at<uchar>(rng.uniform( 0,srcArr.rows), rng.uniform(0, srcArr.cols)) =0;

    }
     for (int counter=0; counter<amount2; ++counter)
     {
     srcArr.at<uchar>(rng.uniform(0,srcArr.rows), rng.uniform(0,srcArr.cols)) = 255;
     }
}


int main(int argc, char *argv[])
{
    Mat srcArr;

  if (argc<=1)
     {   srcArr = imread("original.png"); }

  else if (argc>=2)
     {   srcArr = imread(argv[1]);}

  cvtColor(srcArr,srcArr, COLOR_RGB2GRAY,1);
  
  imshow("The original Image", srcArr);

  Mat srcArr1 = srcArr.clone();
  Mat srcArr2 = srcArr.clone();
  Mat dstArr;

  // Parâmetros pa e pb podem ser alterados para maior ou menor quantidade de ruido
  float pa,pb;
  pa = 0.05;
  pb = 0.05;
 
  Add_salt_pepper_Noise(srcArr1, pa,pb);
  imshow("Add salt and pepper noise to image ", srcArr1);
  imwrite("salt_pepper_noise_image.jpg",srcArr1);


  waitKey(0);
  return 0;
}

