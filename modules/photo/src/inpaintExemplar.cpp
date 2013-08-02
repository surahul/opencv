/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/photo/photo_c.h"

namespace cv {


static Mat inputImage;
static Mat sourceMask;
static Mat mask,updatedMask;
static Mat result;
static Mat workImage;
static Mat sourceRegion;
static Mat targetRegion;
static Mat originalSourceRegion;
static Mat gradientX;
static Mat gradientY;
static Mat confidence;
static Mat data;
static Mat LAPLACIAN_KERNEL,NORMAL_KERNELX,NORMAL_KERNELY;
static Point2i bestMatchUpperLeft,bestMatchLowerRight;
static std::vector<Point> fillFront;
static std::vector<Point2f> normals;
static int halfPatchWidth;
static int targetIndex;



static bool checkEnd(){
    for(int x=0;x<sourceRegion.cols;x++){
        for(int y=0;y<sourceRegion.rows;y++){
            if(sourceRegion.at<uchar>(y,x)==0){
                return true;
               }
        }

    }
    return false;
}

static void getPatch(Point2i &centerPixel, Point2i &upperLeft, Point2i &lowerRight){
    int x,y;
    x=centerPixel.x;
    y=centerPixel.y;

    int minX=std::max(x-halfPatchWidth,0);
    int maxX=std::min(x+halfPatchWidth,workImage.cols-1);
    int minY=std::max(y-halfPatchWidth,0);
    int maxY=std::min(y+halfPatchWidth,workImage.rows-1);


    upperLeft.x=minX;
    upperLeft.y=minY;

    lowerRight.x=maxX;
    lowerRight.y=maxY;
}



static void updateMats(){
    Point2i targetPoint=fillFront.at(targetIndex);
    Point2i a,b;
    getPatch(targetPoint,a,b);
    int width=b.x-a.x+1;
    int height=b.y-a.y+1;

    for(int x=0;x<width;x++){
        for(int y=0;y<height;y++){
            if(sourceRegion.at<uchar>(a.y+y,a.x+x)==0){
                workImage.at<Vec3b>(a.y+y,a.x+x)=workImage.at<Vec3b>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                gradientX.at<float>(a.y+y,a.x+x)=gradientX.at<float>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                gradientY.at<float>(a.y+y,a.x+x)=gradientY.at<float>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                confidence.at<float>(a.y+y,a.x+x)=confidence.at<float>(targetPoint.y,targetPoint.x);
                sourceRegion.at<uchar>(a.y+y,a.x+x)=1;
                targetRegion.at<uchar>(a.y+y,a.x+x)=0;
                updatedMask.at<uchar>(a.y+y,a.x+x)=0;
            }
        }
    }


}

static void computeBestPatch(){
    double minError=9999999999999999,bestPatchVarience=9999999999999999;
    Point2i a,b;
    Point2i currentPoint=fillFront.at(targetIndex);
    Vec3b sourcePixel,targetPixel;
    double meanR,meanG,meanB;
    double difference,patchError;
    bool skipPatch;
    getPatch(currentPoint,a,b);

    int width=b.x-a.x+1;
    int height=b.y-a.y+1;
    for(int x=0;x<=workImage.cols-width;x++){
        for(int y=0;y<=workImage.rows-height;y++){
            patchError=0;
            meanR=0;meanG=0;meanB=0;
            skipPatch=false;

            for(int x2=0;x2<width;x2++){
                for(int y2=0;y2<height;y2++){
                    if(originalSourceRegion.at<uchar>(y+y2,x+x2)==0||sourceMask.at<uchar>(y+y2,x+x2)==0){
                        skipPatch=true;
                        break;
                     }


                    if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0)
                        continue;

                    sourcePixel=workImage.at<Vec3b>(y+y2,x+x2);
                    targetPixel=workImage.at<Vec3b>(a.y+y2,a.x+x2);

                    for(int i=0;i<3;i++){
                        difference=sourcePixel[i]-targetPixel[i];
                        patchError+=difference*difference;
                    }
                    meanB+=sourcePixel[0];meanG+=sourcePixel[1];meanR+=sourcePixel[2];


                }
                if(skipPatch)
                    break;
            }

            if(skipPatch)
                continue;
            if(patchError<minError){
                minError=patchError;
                bestMatchUpperLeft=Point2i(x,y);
                bestMatchLowerRight=Point2i(x+width-1,y+height-1);

                double patchVarience=0;
                for(int x2=0;x2<width;x2++){
                    for(int y2=0;y2<height;y2++){
                        if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0){
                            sourcePixel=workImage.at<Vec3b>(y+y2,x+x2);
                            difference=sourcePixel[0]-meanB;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[1]-meanG;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[2]-meanR;
                            patchVarience+=difference*difference;
                        }

                    }
                }
                bestPatchVarience=patchVarience;

            }else if(patchError==minError){
                double patchVarience=0;
                for(int x2=0;x2<width;x2++){
                    for(int y2=0;y2<height;y2++){
                        if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0){
                            sourcePixel=workImage.at<Vec3b>(y+y2,x+x2);
                            difference=sourcePixel[0]-meanB;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[1]-meanG;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[2]-meanR;
                            patchVarience+=difference*difference;
                        }

                    }
                }
                if(patchVarience<bestPatchVarience){
                    minError=patchError;
                    bestMatchUpperLeft=Point2i(x,y);
                    bestMatchLowerRight=Point2i(x+width-1,y+height-1);
                    bestPatchVarience=patchVarience;
                }
            }
    }
    }


}


static void computeTarget(){

    targetIndex=0;
    float maxPriority=0;
    float priority=0;
    Point2i currentPoint;
    for(int i=0;i<fillFront.size();i++){
        currentPoint=fillFront.at(i);
        priority=data.at<float>(currentPoint.y,currentPoint.x)*confidence.at<float>(currentPoint.y,currentPoint.x);
        if(priority>maxPriority){
            maxPriority=priority;
            targetIndex=i;
        }
    }

}

static void computeData(){

    for(int i=0;i<fillFront.size();i++){
        Point2i currentPoint=fillFront.at(i);
        Point2i currentNormal=normals.at(i);
        data.at<float>(currentPoint.y,currentPoint.x)=std::fabs(gradientX.at<float>(currentPoint.y,currentPoint.x)*currentNormal.x+gradientY.at<float>(currentPoint.y,currentPoint.x)*currentNormal.y)+.001;
    }
}


static void computeConfidence(){
    Point2i a,b;
    for(int i=0;i<fillFront.size();i++){
        Point2i currentPoint=fillFront.at(i);
        getPatch(currentPoint,a,b);
        float total=0;
        for(int x=a.x;x<=b.x;x++){
            for(int y=a.y;y<=b.y;y++){
                if(targetRegion.at<uchar>(y,x)==0){
                    total+=confidence.at<float>(y,x);
                }
            }
        }
        confidence.at<float>(currentPoint.y,currentPoint.x)=total/((b.x-a.x+1)*(b.y-a.y+1));
    }
}

static void computeFillFront(){


    Mat sourceGradientX,sourceGradientY,boundryMat;
    filter2D(targetRegion,boundryMat,CV_32F,LAPLACIAN_KERNEL);
    filter2D(sourceRegion,sourceGradientX,CV_32F,NORMAL_KERNELX);
    filter2D(sourceRegion,sourceGradientY,CV_32F,NORMAL_KERNELY);
    fillFront.clear();
    normals.clear();
    for(int x=0;x<boundryMat.cols;x++){
        for(int y=0;y<boundryMat.rows;y++){

            if(boundryMat.at<float>(y,x)>0){
                fillFront.push_back(Point2i(x,y));

                float dx=sourceGradientX.at<float>(y,x);
                float dy=sourceGradientY.at<float>(y,x);
                Point2f normal(dy,-dx);
                float tempF=std::sqrt((normal.x*normal.x)+(normal.y*normal.y));
                if(tempF!=0){

                normal.x=normal.x/tempF;
                normal.y=normal.y/tempF;

                }
                normals.push_back(normal);

            }
        }
    }


}


static void initializeMats(){
    threshold(mask,confidence,10,255,CV_THRESH_BINARY);
    threshold(confidence,confidence,2,1,CV_THRESH_BINARY_INV);
    confidence.convertTo(confidence,CV_32F);

    sourceRegion=confidence.clone();
    sourceRegion.convertTo(sourceRegion,CV_8U);
    originalSourceRegion=sourceRegion.clone();

    threshold(mask,targetRegion,10,255,CV_THRESH_BINARY);
    threshold(targetRegion,targetRegion,2,1,CV_THRESH_BINARY);
    targetRegion.convertTo(targetRegion,CV_8U);

    threshold(sourceMask,sourceMask,10,255,CV_THRESH_BINARY);
    threshold(sourceMask,sourceMask,2,1,CV_THRESH_BINARY);
    sourceMask.convertTo(sourceMask,CV_8U);



    data=Mat(inputImage.rows,inputImage.cols,CV_32F,Scalar::all(0));


    LAPLACIAN_KERNEL=Mat::ones(3,3,CV_32F);
    LAPLACIAN_KERNEL.at<float>(1,1)=-8;
    NORMAL_KERNELX=Mat::zeros(3,3,CV_32F);
    NORMAL_KERNELX.at<float>(1,0)=-1;
    NORMAL_KERNELX.at<float>(1,2)=1;
    transpose(NORMAL_KERNELX,NORMAL_KERNELY);



}


static void calculateGradients(){
    Mat srcGray;
    cvtColor(workImage,srcGray,CV_BGR2GRAY);

    Scharr(srcGray,gradientX,CV_16S,1,0);
    convertScaleAbs(gradientX,gradientX);
    gradientX.convertTo(gradientX,CV_32F);


    Scharr(srcGray,gradientY,CV_16S,0,1);
    convertScaleAbs(gradientY,gradientY);
    gradientY.convertTo(gradientY,CV_32F);






    for(int x=0;x<sourceRegion.cols;x++){
        for(int y=0;y<sourceRegion.rows;y++){

            if(sourceRegion.at<uchar>(y,x)==0){
                gradientX.at<float>(y,x)=0;
                gradientY.at<float>(y,x)=0;
            }/*else
            {
                if(gradientX.at<float>(y,x)<255)
                    gradientX.at<float>(y,x)=0;
                if(gradientY.at<float>(y,x)<255)
                    gradientY.at<float>(y,x)=0;
            }*/

        }
    }
    gradientX/=255;
    gradientY/=255;
}

static void inpaint(){

    //namedWindow("updatedMask");
    //namedWindow("inpaint");
    //namedWindow("gradientX");
    //namedWindow("gradientY");

    initializeMats();
    calculateGradients();
    bool stay=true;

    while(stay){

        computeFillFront();
        computeConfidence();
        computeData();
        computeTarget();
        computeBestPatch();
        updateMats();
        stay=checkEnd();

    //    imshow("updatedMask",updatedMask);
    //    imshow("inpaint",workImage);
    //    imshow("gradientX",gradientX);
    //    imshow("gradientY",gradientY);
    //    waitKey(2);
    }
    result=workImage.clone();


    //namedWindow("confidence");
    //imshow("confidence",confidence);
}


void inpaint(Mat & _inputImage,Mat & _mask,Mat & _result,int _halfPatchWidth){

    inputImage=_inputImage.clone();
    mask=_mask.clone();
    updatedMask=_mask.clone();
    workImage=_inputImage.clone();
    halfPatchWidth=_halfPatchWidth;
    sourceMask=Mat::ones(_mask.size(),_mask.type())*255;
    inpaint();
    _result=workImage.clone();
}

void inpaint(Mat & _inputImage,Mat & _mask,Mat & _sourceMask,Mat & _result,int _halfPatchWidth){

    inputImage=_inputImage.clone();
    mask=_mask.clone();
    updatedMask=_mask.clone();
    workImage=_inputImage.clone();
    halfPatchWidth=_halfPatchWidth;
    sourceMask=_sourceMask.clone();
    inpaint();
    _result=workImage.clone();
}

//int checkValidInputs(){
//    if(inputImage.type()!=CV_8UC3)
//        return ERROR_INPUT_MAT_INVALID_TYPE;
//    if(mask.type()!=CV_8UC1)
//        return ERROR_MASK_INVALID_TYPE;
//    if(sourceMask.type()!=CV_8UC1)
//        return ERROR_SOURCEMASK_INVALID_TYPE;
//    if(!CV_ARE_SIZES_EQ(&mask,&inputImage))
//        return ERROR_MASK_INPUT_SIZE_MISMATCH;
//    if(!CV_ARE_SIZES_EQ(&sourceMask,&inputImage))
//        return ERROR_SOURCEMASK_INPUT_SIZE_MISMATCH;
//    if(halfPatchWidth==0)
//        return ERROR_HALF_PATCH_WIDTH_ZERO;
//    return CHECK_VALID;
//}


}










