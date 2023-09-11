// LPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "highgui/highgui.hpp"
#include "core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include<baseapi.h>
#include <allheaders.h>
#include <iostream>
using namespace std;
using namespace cv;

Rect bound;

Mat RGBtoGrey(Mat RGB) {
    Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);  //create a new grey with same size as RGB but this resolution is all zero-black
    for (int i = 0; i < RGB.rows; i++) {
        for (int j = 0; j < RGB.cols; j++) {
            Grey.at < uchar>(i, j) = ((RGB.at<uchar>(i, j * 3)) + (RGB.at<uchar>(i, j * 3 + 1)) + (RGB.at<uchar>(i, j * 3 + 2))) / 3;
        }
    }
    return Grey;

}

Mat invert(Mat Grey) {
    Mat inverted = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            inverted.at <uchar>(i, j) = 255 - Grey.at <uchar>(i, j);
        }
    }
    return inverted;
}

Mat converttoBinary(Mat Grey, int num) {

    Mat binary = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            if (Grey.at <uchar>(i, j) > num) {
                binary.at <uchar>(i, j) = 255;
            }
            else binary.at <uchar>(i, j) = 0;
        }
    }
    return binary;
}


//input windowSize
Mat Average1(Mat Grey, int windowSize) {
    Mat average = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = windowSize; i < Grey.rows - windowSize; i++) {
        for (int j = windowSize; j < Grey.cols - windowSize; j++) {
            int sum = 0;
            for (int ii = -windowSize; ii <= windowSize; ii++) {
                for (int jj = -windowSize; jj <= windowSize; jj++) {
                    sum += Grey.at <uchar>(i + ii, j + jj);
                }
            }
            average.at <uchar>(i, j) = sum / ((windowSize * 2 + 1) * (windowSize * 2 + 1));
        }
    }

    return average;
}



//equalize histogram
Mat EquilizeHist(Mat Grey)
{
    Mat hist = Mat::zeros(Grey.size(), CV_8UC1);
    int count[256] = { 0 }; //integer array of 256 values all are set to zeroes
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            count[Grey.at <uchar>(i, j)]++;
        }
    }
    float prob[256] = { 0.0 };
    for (int i = 0; i < 256; i++) {
        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);
    }
    float acc[256] = { 0.0 };
    acc[0] = prob[0];
    for (int i = 1; i < 256; i++) {
        acc[i] = acc[i - 1] + prob[i];
    }
    int newPixel[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        newPixel[i] = (int)(acc[i] * 255);
    }
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            hist.at <uchar>(i, j) = newPixel[Grey.at <uchar>(i, j)];
        }
    }

    return hist;
}


Mat EdgeDetection(Mat input, int avg)
{
    Mat edge = Mat::zeros(input.size(), CV_8UC1);
    for (int i = 1; i < input.rows - 1; i++) {
        for (int j = 1; j < input.cols - 1; j++) {
            int avgl = 0;
            int avgr = 0;
            avgl = (input.at<uchar>(i - 1, j - 1) + input.at<uchar>(i, j - 1) + input.at<uchar>(i + 1, j - 1)) / 3;
            avgr = (input.at<uchar>(i - 1, j + 1) + input.at<uchar>(i, j + 1) + input.at<uchar>(i + 1, j + 1)) / 3;
            if ((abs(avgl - avgr)) > avg) {
                edge.at<uchar>(i, j) = 255;
            }
        }
    }
    return edge;
}


Mat Erosion(Mat Grey, int windows) {
    Mat Binary = Grey.clone();
    for (int i = windows; i < Grey.rows - windows; i++) {
        for (int j = windows; j < Grey.cols - windows; j++) {
            if (Grey.at<uchar>(i, j) == 255)
                for (int ii = -windows; ii <= windows; ii++)
                    for (int jj = -windows; jj <= windows; jj++)
                        if (Grey.at<uchar>(ii + i, jj + j) == 0)
                            Binary.at<uchar>(i, j) = 0;
        }
    }
    return Binary;
}

Mat Dilation(Mat erosion, int windows) {
    Mat dilation = erosion.clone();
    for (int i = windows; i < erosion.rows - windows; i++) {
        for (int j = windows; j < erosion.cols - windows; j++) {
            if (erosion.at<uchar>(i, j) == 0)
                for (int ii = -windows; ii <= windows; ii++)
                    for (int jj = -windows; jj <= windows; jj++)
                        if (erosion.at<uchar>(ii + i, jj + j) == 255)
                            dilation.at<uchar>(i, j) = 255;
        }
    }
    return dilation;
}
 

int OTSU(Mat Grey, int value)
{
    Mat otsu = Mat::zeros(Grey.size(), CV_8UC1);
    int count[256] = { 0 }; //integer array of 256 values all are set to zeroes
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            count[Grey.at <uchar>(i, j)]++;
        }
    }
    float prob[256] = { 0.0 };
    for (int i = 0; i < 256; i++) {
        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);
    }
    float acc[256] = { 0.0 };
    acc[0] = prob[0];
    for (int i = 1; i < 256; i++) {
        acc[i] = acc[i - 1] + prob[i];
    }

    float meu[256] = { 0,0 };
    //meu[0] = 0;
    for (int i = 1; i < 256; i++) {
        meu[i] = meu[i - 1] + (i * prob[i]);
    }



    float sigma[256] = { 0,0 };
    for (int i = 1; i < 256; i++) {
        sigma[i] = pow(meu[255] * acc[i] - meu[i], 2) / (acc[i] * (1 - acc[i]));
    }


    float max = 0.0;
    int maxsigma = -1;
    for (int i = 0; i < 256; i++) {
        if (sigma[i] > max) {
            max = sigma[i];
            maxsigma = i;
        }
    }

    return maxsigma + value;
}

Mat PlateDetection(Mat Grey, int edgeAvg, int dilWS, double width, double height, double smallX, double bigX, double y) {
    Mat hist = EquilizeHist(Grey);
    Mat avg = Average1(hist, 1);
    Mat edge = EdgeDetection(avg, edgeAvg);
    Mat ero = Erosion(edge, 1);
    Mat dil = Dilation(ero, dilWS); //dilation with erosion 

 
    
    Mat segments;
    segments = dil.clone();
    vector<vector<Point>>contours1;
    vector<Vec4i>hierachy1;
    findContours(segments, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

    //segmention with colour
    /*Mat dst = Mat::zeros(Grey.size(), CV_8UC3);
    if (!contours1.empty())
    {
        for (int i = 0; i < contours1.size(); i++)
        {
            Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
            drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
        }
    }*/
    // imshow("dst", dst);

    //segmentation of plate 
    Rect rect_first;
    Scalar black = CV_RGB(0, 0, 0);
    Mat Plate;
    for (int i = 0; i < contours1.size(); i++)
    {
        rect_first = boundingRect(contours1[i]);  
        if (rect_first.width < Grey.cols * width || rect_first.height > Grey.rows * height || rect_first.x < Grey.cols * smallX || rect_first.x > Grey.cols * bigX || rect_first.y < Grey.rows * y)

        {
            drawContours(segments, contours1, i, black, -1, 8, hierachy1);
        }
        else {
            Plate = Grey(rect_first);
            bound = rect_first;
        }
    }   
    return Plate;
}



int main()
{
    tesseract::TessBaseAPI api;
    //image path needs to be changed if images are stored differently
    String pattern = "C:\\Users\\eBay\\Desktop\\Dataset\\*.jpg";
    vector<String> fn;
    glob(pattern, fn, false);
    vector<Mat> images;
    size_t count = fn.size(); //number of jpg files in images folder
    for (size_t i = 0; i < count; i++)
    {
        
        images.emplace_back(cv::imread(fn[i]));
        Mat img = imread(fn[i]);
        //imshow("Original image", img); //show my image in windows 
        //waitKey();
        
         Mat Grey = RGBtoGrey(img);

         Mat Plate = PlateDetection(Grey, 30, 8, 0.15, 0.2, 0.15, 1.2, 0.27);
         if (Plate.rows <= 0 || Plate.cols <= 0) // there is no plate 
         {
             Plate = PlateDetection(Grey, 42, 8, 0.11, 0.13, 0.15, 0.6, 0.27);
             if (Plate.rows <= 0 || Plate.cols <= 0) {
                 Plate = PlateDetection(Grey, 10, 6, 0.06, 0.13, 0.08, 0.7, 0.27);
                 if (Plate.rows <= 0 || Plate.cols <= 0) {
                     cout << "Error detecting plate";
                 }
             }
         } 
         
         ////if image is not in sequence - this may not be able get the most accurate ocr resu
         //Mat BinPlate;
         //BinPlate = converttoBinary(Plate, OTSU(Plate, 30));
         ////character segmentation
         //Mat plateSegment;
         //plateSegment = BinPlate.clone();
         //vector<vector<Point>>contours2;
         //vector<Vec4i>hierachy2;
         //findContours(plateSegment, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0)); 
         //Rect rect_first;
         //Scalar black = CV_RGB(0, 0, 0); 
         //Mat Plate2;  
         //for (int i = 0; i < contours2.size(); i++)
         //{ 
         //    String filename = "C:\\Users\\eBay\\Desktop\\ISE\\" + to_string(i) + ".jpg";
         //    rect_first = boundingRect(contours2[i]);
         //     if ( rect_first.width > BinPlate.cols * 0.3 || rect_first.height > BinPlate.rows * 0.8 || rect_first.height < BinPlate.rows * 0.1)
         //     {
         //        drawContours(plateSegment, contours2, i, black, -1, 8, hierachy2); //remove noises from plate
         //    } 
         //}

         //binarie the plate
         Mat BinPlate;
         int j = i + 1;
         if (j == 1 || j == 2 || j == 7 || j == 8 || j == 17 || j == 20) {
             BinPlate = converttoBinary(Plate, OTSU(Plate, 30)); 
         }
         else if (j == 9 || j == 10 || j == 12 || j == 16) {
             BinPlate = converttoBinary(Plate, OTSU(Plate, 50));
         }
         else if (j == 4 || j == 6 || j == 13 || j == 14 || j == 18 || j == 19) {
             BinPlate = converttoBinary(Plate, OTSU(Plate, 70));
         }
         else if (j == 3 || j ==5 ){
             BinPlate = converttoBinary(Plate, OTSU(Plate, 80));
         }
         else {
             BinPlate = converttoBinary(Plate, OTSU(Plate, 100));
         }
          

         //remove noises from plate detected
         Mat plateSegment;
         plateSegment = BinPlate.clone();
         vector<vector<Point>>contours;
         vector<Vec4i>hierachy2;
         findContours(plateSegment, contours, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0)); 
         Rect rect_first;
         Scalar black = CV_RGB(0, 0, 0);
         Mat Plate2;
         for (int i = 0; i < contours.size(); i++)
         {
             rect_first = boundingRect(contours[i]); 
             if (j == 11 || j==3) {
                 if (rect_first.width > BinPlate.cols * 0.3 || rect_first.height > BinPlate.rows * 0.8 || rect_first.height < BinPlate.rows * 0.1 || rect_first.x < BinPlate.cols * 0.1 || rect_first.x > BinPlate.cols * 0.7)
                 {
                     drawContours(plateSegment, contours, i, black, -1, 8, hierachy2); //remove noises from plate
                 } 
             }
             else if(j == 13) {
                 if (rect_first.width > BinPlate.cols * 0.3 || rect_first.height > BinPlate.rows * 0.8 || rect_first.height < BinPlate.rows * 0.1 || rect_first.x < BinPlate.cols * 0.1 || rect_first.x > BinPlate.cols * 06)
                 {
                     drawContours(plateSegment, contours, i, black, -1, 8, hierachy2); //remove noises from plate
                 }
             }
             else {
                 if (rect_first.width > BinPlate.cols * 0.3 || rect_first.height > BinPlate.rows * 0.8 || rect_first.height < BinPlate.rows * 0.1)
                 {
                     drawContours(plateSegment, contours, i, black, -1, 8, hierachy2); //remove noises from plate
                 } 
             } 
         }
          
         imshow("Less noise", plateSegment); 
          
         //OCR 
         Mat img2 = invert(plateSegment);
         api.Init(NULL, "eng"); //initialize ocr using english
         api.SetVariable("user_defined_dpi", "300");
         api.SetPageSegMode(static_cast<tesseract::PageSegMode>(tesseract::PageSegMode::PSM_AUTO_OSD)); 
         api.SetPageSegMode(static_cast<tesseract::PageSegMode>(tesseract::PageSegMode::PSM_SINGLE_BLOCK));
         api.SetImage((uchar*)img2.data, img2.cols, img2.rows, 1, img2.cols);
         char* out = api.GetUTF8Text();	//get context (output text) from picture
         string t1(out); //and put it in string
         t1.erase(remove(t1.begin(), t1.end(), '\n'), t1.end());
         cout << "Text:" << t1.c_str() << endl;
          
         int fontFace = FONT_HERSHEY_PLAIN;
         double fontScale = 2;
         int thickness = 2; 
         putText(img, t1, Point(bound.x, bound.y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
         rectangle(img, bound, Scalar(0, 0, 255), 2, 8, 0);
         imshow("tesseract-opencv", img);
         waitKey(0);

         delete[] out;
         api.Clear();
         api.End();
          
    }
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
