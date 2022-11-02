#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <fstream> 
#include <sstream>
#include <string>  
#include <cstring>
#include <streambuf> 
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <omp.h>
#include <chrono>
#include "cxx-prettyprint/prettyprint.hpp" // for debug

//
//This program is inspired by an article named "Best-Buddies Similarity for Robust Template Matching CVPR2015"
//After reading this passage, I started to realize this method described in it by using OPENCV and c++.
//And here is the source code I wrote.
//"BBS" is short for "Best-Buddies Similarity", which is a useful, robust, and parameter-free similarity measure between two sets of points.
//BBS is based on counting the number of Best-Buddies Pairs (BBPs)-pairs of points in source and target sets, 
//where each point is the nearest neighbor of the other. 
//BBS has several key features that make it robust against complex geometric deformations and high levels of outliers, 
//such as those arising from background clutter and occlusions. 
//And the output of this source code on the challenging real-world dataset is amazingly precise, far beyond my previous expectation.
//

int gamma_, verbose;

//Gaussian lowpass filter
float Gaussian[]{ 0.0277, 0.1110, 0.0277, 0.1110, 0.4452, 0.1110, 0.0277, 0.1110, 0.0277 };

//convert the image's info into a matrix and store them as a 2-dim vector list.
cv::Mat Im2col(cv::Mat src, int pz){ // input or template / pz / pz
    int cols = ceil(src.rows / pz) * ceil(src.cols / pz); // 160 * 90 = 14400 or 15 * 7 = 105
    cv::Mat ans(pz * pz, cols, CV_32FC3);
    cv::Point2d ref;
    std::cout << "(debug)cols of ans: " << cols << std::endl;

    int col = 0; // col of ans
    for (int j = 0; j < (src.rows - src.rows % pz); j += pz){
        for (int i = 0; i < (src.cols - src.cols % pz); i += pz){
            for (int k = 0; k < (pz * pz); k++){ // row of ans
                ref.y = j + k / pz; ref.x = i + k % pz;
                ans.at<cv::Vec3f>(k, col)[0] = src.at<cv::Vec3f>(ref.y, ref.x)[0];
                ans.at<cv::Vec3f>(k, col)[1] = src.at<cv::Vec3f>(ref.y, ref.x)[1];
                ans.at<cv::Vec3f>(k, col)[2] = src.at<cv::Vec3f>(ref.y, ref.x)[2];
            }
            col++;
        }
    }
    std::cout << "(debug)col of for loop: " << col << std::endl;

    return ans;
}

//the main code
int main(int argc, char *argv[]){

    int mode;
    mode = 1;
    std::string TName, IName, TxtName, resultName, output_name, logT, logI;
    cv::Mat RESR, RESG, RESB;
    cv::Mat RESR2, RESG2, RESB2;
    verbose = 0;
    int pz = 3;

    // Check Options
    for( int idx = 1; idx < argc; idx++ ){
        if( !strcmp( argv[idx], "-gamma" )) gamma_ = atoi( argv[++idx] );
        else if( !strcmp( argv[idx], "-pz" )) pz = atoi( argv[++idx] );
        else if( !strcmp( argv[idx], "-tmp" )) TName = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-i" )) IName = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-txt" )) TxtName = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-res" )) resultName = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-log" )) output_name = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-logT" )) logT = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-logI" )) logI = std::string( argv[++idx] );
        else if( !strcmp( argv[idx], "-v" )) verbose = atoi( argv[++idx] );
        else if( !strcmp( argv[idx], "-mode" )) mode = atoi( argv[++idx] );
    }

    /*===============================================================*/
    cv::Mat Ts, Is; // Template image / Input image
    cv::Mat T, I; // Template image / Input image <CV_8UC3 -> CV_32FC3>
    cv::Mat TMat, IMat; // Template image / Input image (Divided into patches) <CV_32FC3>
    std::vector<int> Tcut(4, 0); // [0]x of the top left / [1]y of the top left / [2]width / [3]height
    /*===============================================================*/

    std::cout << "#############################################" << std::endl;
    std::cout << "Template image: " << TName << std::endl;
    std::cout << "Input image: " << IName << std::endl;
    std::cout << "Input text: " << TxtName << std::endl;
    std::cout << "#############################################" << std::endl;
    
    Ts = cv::imread(TName);
    if(Ts.empty()){
        std::cerr << "Cannot load template image." << std::endl;
        return -1;
    }
    Is = cv::imread(IName);
    if(Is.empty()){
        std::cerr << "Cannot load input image." << std::endl;
        return -1;
    }

    std::ifstream input(TxtName);
    if(mode == 1) input >> Tcut[0] >> Tcut[1] >> Tcut[2] >> Tcut[3];
    if(mode == 0){
        std::string temp, temp2;
        getline(input, temp);
        std::istringstream ss(temp);
        int i = 0;
        do{
            ss >> Tcut[i++];
        }while (getline(ss, temp2, ','));
    }
    std::cout << "Text: " << Tcut[0] << " " << Tcut[1] << " " << Tcut[2] << " " << Tcut[3] << std::endl;

    //clipping pictures
    if ((Tcut[2] % pz) < (pz / 2)) Tcut[2] -= (Tcut[2] % pz); // cutoff
    else Tcut[2] += (pz - Tcut[2] % pz);                      // round up
    if ((Tcut[3] % pz) < (pz / 2)) Tcut[3] -= (Tcut[3] % pz); // cutoff
    else Tcut[3] += (pz - Tcut[3] % pz);                      // round up
    std::cout << "Text (After clipping):" << Tcut[0] << " " << Tcut[1] << " " << Tcut[2] << " " << Tcut[3] << std::endl;

    if(mode == 0) T = Ts(cv::Rect(Tcut[0], Tcut[1], Tcut[2], Tcut[3]));
    if(mode == 1) T = Ts;
    I = Is(cv::Rect(0, 0, (Is.cols - Is.cols % pz), (Is.rows - Is.rows % pz))); // cutoff (Input image size must be divisible by 3 as with template image)

    //cv::imwrite(logI, I);
    //cv::imwrite(logT, T);

    // Normalize the image from 0 to 1.
    T.convertTo(T, CV_32FC3, 1.0 / 255.0);  I.convertTo(I, CV_32FC3, 1.0 / 255.0);

    // loop over image pixels
    std::cout << "Size of T: (" << T.rows << " x " << T.cols << ")" << std::endl;
    std::cout << "Size of I: (" << I.rows << " x " << I.cols << ")" << std::endl;
    
    // Divided into patches
    TMat = Im2col(T, pz);   IMat = Im2col(I, pz);

    int N = TMat.cols; // 105
    int rowT = T.rows; // 45
    int colT = T.cols; // 21
    int rowI = I.rows; // 270
    int colI = I.cols; // 480


    //pre compute spatial distance component
    std::vector<std::vector<float>> Dxy(TMat.cols, std::vector<float>(TMat.cols));
    std::vector<std::vector<float>> Dxy2(TMat.cols, std::vector<float>(TMat.cols));
    std::vector<std::vector<float>> Drgb(TMat.cols, std::vector<float>(TMat.cols));
    std::vector<std::vector<float>> Drgb_prev(TMat.cols, std::vector<float>(TMat.cols));
    std::vector<std::vector<float>> D(TMat.cols, std::vector<float>(TMat.cols));
    std::vector<std::vector<float>> D_r(TMat.cols, std::vector<float>(TMat.cols));
    std::vector<std::vector<float>> BBS(I.rows, std::vector<float>(I.cols));

    //Drgb's buffer
    int bufSize = I.rows - T.rows; // 270 - 45 = 225
    std::vector<std::vector<std::vector<float>>> Drgb_buffer(TMat.cols, std::vector<std::vector<float>>(TMat.cols, std::vector<float>(bufSize)));
    //Drgb_buffer.resize(N);
    /*for (int i = 0; i < static_cast<int>(Drgb_buffer.size()); i++){
        Drgb_buffer[i].resize(N);
        for (int j = 0; j < static_cast<int>(Drgb_buffer[i].size()); j++) Drgb_buffer[i][j].resize(bufSize);
    }*/

    std::vector<float> xx, yy;
    for (int i = 0; (pz * i) < T.cols; i++){
        float n = pz * i * 3.0039;
        for (int j = 0; (pz * j) < T.rows; j++){
            float m = pz * j * 0.0039;
            xx.push_back(n);
            yy.push_back(m);
        }
    }
    std::cout << "(debug)size of xx: " << xx.size() << std::endl;
    //std::cout << "--- xx: " << xx << std::endl;
    //std::cout << "--- yy: " << yy << std::endl;
    
    for (int j = 0; j < static_cast<int>(yy.size()); j++){
        for (int i = 0; i < static_cast<int>(xx.size()); i++){
            Dxy[j][i] = pow((xx[i] - xx[j]), 2) + pow((yy[i] - yy[j]), 2);
        }
    }
    //std::cout << "--- Dxy: " << Dxy << std::endl;

    std::vector<std::vector<int>> IndMat(I.rows / pz, std::vector<int>(I.cols / pz));
    //IndMat.resize(I.rows / pz); // 90
    //for (int i = 0; i < static_cast<int>(IndMat.size()); i++) IndMat[i].resize(I.cols / pz); // 160

    int n = 0;
    for (int j = 0; j < (I.cols / pz); j++){
        for (int i = 0; i < (I.rows / pz); i++){
            IndMat[i][j] = n++; // Nmax is 14399
        }
    }



    //std::chrono::system_clock::time_point start, end;
    //start = std::chrono::system_clock::now();

    //#pragma omp parallel for
    
    for (int coli = 0; coli < (I.cols / pz - T.cols / pz + 1); coli++){ // 154
        for (int rowi = 0; rowi < (I.rows / pz - T.rows / pz + 1); rowi++){ // 76
            cv::Mat PMat(9, TMat.cols, CV_32FC3);
            std::vector<int> v;
            std::vector<float> w;
            for (int j = coli; j < (coli + T.cols / pz); j++){
                for (int i = rowi; i < (rowi + T.rows / pz); i++){
                    v.push_back(IndMat[i][j]);
                }
            }
            int ptv = 0;
            for (int ix = 0; ix < N; ix++){
                for (int jx = 0; jx < 9; jx++){
                    PMat.at<cv::Vec3f>(jx, ix)[0] = IMat.at<cv::Vec3f>(jx, v[ptv])[0];
                    PMat.at<cv::Vec3f>(jx, ix)[1] = IMat.at<cv::Vec3f>(jx, v[ptv])[1];
                    PMat.at<cv::Vec3f>(jx, ix)[2] = IMat.at<cv::Vec3f>(jx, v[ptv])[2];
                }
                ptv++;
            }

            //compute distance matrix
            for (int idxP = 0; idxP < N; idxP++){
                cv::Mat Temp(9, N, CV_32FC3);
                for (int i = 0; i < Temp.cols; i++){
                    for (int j = 0; j < Temp.rows; j++){
                        Temp.at<cv::Vec3f>(j, i)[0] = pow(((-TMat.at<cv::Vec3f>(j, i)[0] + PMat.at<cv::Vec3f>(j, idxP)[0])*Gaussian[j]), 2);
                        Temp.at<cv::Vec3f>(j, i)[1] = pow(((-TMat.at<cv::Vec3f>(j, i)[1] + PMat.at<cv::Vec3f>(j, idxP)[1])*Gaussian[j]), 2);
                        Temp.at<cv::Vec3f>(j, i)[2] = pow(((-TMat.at<cv::Vec3f>(j, i)[2] + PMat.at<cv::Vec3f>(j, idxP)[2])*Gaussian[j]), 2);
                    }
                }
                for (int jx = 0; jx < N; jx++){
                    float res = 0;
                    for (int ix = 0; ix < 9; ix++){
                        if (D[ix][jx] < 1e-4) D[ix][jx] = 0;
                        res += Temp.at<cv::Vec3f>(ix, idxP)[0];
                        res += Temp.at<cv::Vec3f>(ix, idxP)[1];
                        res += Temp.at<cv::Vec3f>(ix, idxP)[2];
                    }
                    Drgb[jx][idxP] = res;
                }
            }

            //make the reversed matrix of distance matrix
            for (int ix = 0; ix < N; ix++){
                for (int jx = 0; jx < N; jx++){
                    //calculate distance
                    D[ix][jx] = Dxy[ix][jx] * gamma_ + Drgb[ix][jx];
                    if (D[ix][jx] < 1e-4) D[ix][jx] = 0;
                    D_r[jx][ix] = D[ix][jx];
                }
            }

            //compute the BBS value of this point
            std::vector<float> minVal1, minVal2;
            std::vector<int> idx1, idx2, ii1, ii2;

            for (int ix = 0; ix < N; ix++){
                auto min1 = min_element(begin(D[ix]), end(D[ix]));
                minVal1.push_back(*min1);
                idx1.push_back(distance(begin(D[ix]), min1));

                ii1.push_back((ix * N) + idx1[ix]);
            }
            for (int ix = 0; ix < N; ix++){
                auto min2 = min_element(begin(D_r[ix]), end(D_r[ix]));
                minVal2.push_back(*min2);
                idx2.push_back(distance(begin(D_r[ix]), min2));
                ii2.push_back((ix * N) + idx2[ix]);
            }

            std::vector<std::vector<int>> IDX_MAT1, IDX_MAT2;
            IDX_MAT1.resize(N);
            IDX_MAT2.resize(N);
            for (int i = 0; i < N; i++){
                IDX_MAT1[i].resize(N);
                IDX_MAT2[i].resize(N);
            }
            int sum, sum2, pt1, pt2;
            sum = sum2 = pt1 = pt2 = 0;
            for (int ix = 0; ix < N; ix++){
                for (int jx = 0; jx < N; jx++){
                    IDX_MAT1[ix][jx] = 0;
                    IDX_MAT2[ix][jx] = 999;
                    if ((pt1 < N) && ((ix * N + jx) == ii1[pt1])){
                        IDX_MAT1[ix][jx] = 1;
                        pt1++;
                    }
                    if ((pt2 < N) && ((jx * N + ix) == ii2[pt2])){
                        IDX_MAT2[ix][jx] = 1;
                        pt2++;
                    }
                    if (IDX_MAT2[ix][jx] == IDX_MAT1[ix][jx])
                        sum += 1;
                }
            }	
            BBS[rowi][coli] = sum;
            //cout << coli << " " << rowi << endl;
        }
    }

    //end = std::chrono::system_clock::now();
    //const double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() *0.001;
    //std::cout << "Time: " << time << " [msec]" << std::endl;

    float max = BBS[0][0];
    int markRow, markCol;
    markCol = markRow = 0;

    // search max score
    for (int i = 0; i < I.rows; i++){
        for (int j = 0; j < I.cols; j++){
            if (BBS[i][j] >= max){
                max = BBS[i][j];
                markRow = i;
                markCol = j;
            }
        }
    }

    if(verbose){
        //Initialize the output iamge and .txt files
        std::cout << output_name << std::endl;

        std::ofstream output(output_name);
        output << markRow * pz << " " << markCol * pz << std::endl;
        output.close();
    }

    cv::Mat OUTPUT1, OUTPUT2, OUTPUT3;
    cv::Mat Is2, Ts2;
    if(mode == 1) Is2 = cv::imread(IName, 1);
    if(mode == 0){
        Ts2 = cv::imread(TName, 1);
        Is2 = cv::imread(IName, 1);
    }
    RESR = cv::Mat_<uchar>(rowI, colI);
    RESG = cv::Mat_<uchar>(rowI, colI);
    RESB = cv::Mat_<uchar>(rowI, colI);
    RESR2 = cv::Mat_<uchar>(rowI, colI);
    RESG2 = cv::Mat_<uchar>(rowI, colI);
    RESB2 = cv::Mat_<uchar>(rowI, colI);

    for( int j = 0; j < rowI; j++ ) {
        for( int i = 0; i < colI; i++ ) {
            if(mode == 0){
                RESR.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2];
                RESG.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1];
                RESB.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0];
                RESR2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[2];
                RESG2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[1];
                RESB2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[0];
            }else{
                RESR.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2];
                RESG.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1];
                RESB.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0];
                RESR2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2];
                RESG2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1];
                RESB2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0];
            }
        }
    }

    //mark rectangle
    int si, sj, ei, ej;
    si  = markCol * pz;
    sj  = markRow * pz;
    ei  = si + colT;
    ej  = sj + rowT;

    //Rect-Yellow
    cv::rectangle(RESR,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1);
    cv::rectangle(RESG,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1);
    cv::rectangle(RESB,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(0), 1);
    cv::rectangle(RESR,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1);
    cv::rectangle(RESG,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1);
    cv::rectangle(RESB,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(0), 1);

    //Rect-Blue
    cv::rectangle(RESR,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(50), 1);
    cv::rectangle(RESG,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(255), 1);
    cv::rectangle(RESB,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1);
    
    //Rect-Yellow
    cv::rectangle(RESR,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1);
    cv::rectangle(RESG,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1);
    cv::rectangle(RESB,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(0), 1);
    cv::rectangle(RESR,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1);
    cv::rectangle(RESG,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1);
    cv::rectangle(RESB,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(0), 1);

    std::vector<cv::Mat> color_img1;
    color_img1.push_back(RESB);
    color_img1.push_back(RESG);
    color_img1.push_back(RESR);
    merge(color_img1, OUTPUT1);

    si  = Tcut[0];
    sj  = Tcut[1];
    ei  = Tcut[0] + Tcut[2];
    ej  = Tcut[1] + Tcut[3];

    //Rect-Yellow
    cv::rectangle(RESR2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1);
    cv::rectangle(RESG2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1);
    cv::rectangle(RESB2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(0), 1);
    cv::rectangle(RESR2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1);
    cv::rectangle(RESG2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1);
    cv::rectangle(RESB2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(0), 1);

    //Rect-Red
    cv::rectangle(RESR2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1);
    cv::rectangle(RESG2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1);
    cv::rectangle(RESB2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(255), 1);
    
    //Rect-Yellow
    cv::rectangle(RESR2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1);
    cv::rectangle(RESG2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1);
    cv::rectangle(RESB2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(0), 1);
    cv::rectangle(RESR2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1);
    cv::rectangle(RESG2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1);
    cv::rectangle(RESB2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(0), 1);

    std::vector<cv::Mat> color_img2;
    color_img2.push_back(RESB2);
    color_img2.push_back(RESG2);
    color_img2.push_back(RESR2);
    merge(color_img2, OUTPUT2);

    hconcat(OUTPUT1, OUTPUT2, OUTPUT3);
    std::cout << resultName << std::endl << std::endl;
    imwrite(resultName, OUTPUT3);
    
    return 0;
}