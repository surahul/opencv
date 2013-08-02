/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "opencv2/core/core.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

cv::Mat image,originalImage,inpaintMask,sourceMask;
cv::Point prevPt(-1,-1);
int thickness=5;
static void onMouse( int event, int x, int y, int flags, void* )
{
    if(event == cv::EVENT_LBUTTONUP||!(flags & cv::EVENT_FLAG_LBUTTON) )
        prevPt = cv::Point(-1,-1);
    else if( event == cv::EVENT_LBUTTONDOWN )
        prevPt = cv::Point(x,y);
    else if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON) )
    {
        cv::Point pt(x,y);
        if( prevPt.x < 0 )
            prevPt = pt;
        cv::line( inpaintMask, prevPt, pt, cv::Scalar::all(255), thickness, 8, 0 );
        cv::line( image, prevPt, pt, cv::Scalar::all(255), thickness, 8, 0 );
        prevPt = pt;
        cv::imshow("image", image);
    }
}




int main(int argc, char *argv[])
{

    //we expect three arguments.
    //the first is the image path.
    //the second is the mask path.
    //the third argument is the halfPatchWidth
    //the fourth argument is sourceMask path

    //in case source mask path is not specified the whole image is searched for Exempler patches.
    //in case halPatchWidth is not specified we use a default value of 4.
    //in case only image path is speciifed, we use manual marking of mask over the image.
    //in case image name is also not specified , we use default image default.jpg.



    int halfPatchWidth=4;
    bool applySourceMask=false;
    if(argc>=5){
        sourceMask=cv::imread(argv[4],CV_LOAD_IMAGE_GRAYSCALE);
        if(sourceMask.data)
            applySourceMask=true;

    }
    if(argc>=4)
    {
        std::stringstream ss;
        ss<<argv[3];
        ss>>halfPatchWidth;
    }

    char* imageName = argc >= 2 ? argv[1] : (char*)"default.jpg";

    originalImage=cv::imread(imageName,CV_LOAD_IMAGE_COLOR);

    if(!originalImage.data){
        std::cout<<std::endl<<"Error unable to open input image"<<std::endl;
        return 0;
    }

    image=originalImage.clone();



    bool maskSpecified=false;
    char* maskName;
    if(argc >= 3){
       maskName=argv[2];
       maskSpecified=true;
    }

    if(maskSpecified){
        inpaintMask=cv::imread(maskName,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat result;
        if(applySourceMask)
            cv::inpaint(originalImage,inpaintMask,sourceMask,result,halfPatchWidth);
        else
            cv::inpaint(originalImage,inpaintMask,result,halfPatchWidth);
        cv::imwrite("result.jpg",result);
        cv::namedWindow("result");
        cv::imshow("result",result);
        cv::waitKey();

    }
    else
    {
        std::cout<<std::endl<<"mask not specified , mark manually on input image"<<std::endl;
        inpaintMask = cv::Mat::zeros(image.size(), CV_8U);
        cv::namedWindow( "image", 1 );
        cv::imshow("image", image);
        cv::setMouseCallback( "image", onMouse, 0 );

        for(;;)
            {
                char c = (char)cv::waitKey();

                if( c == 'e' )
                    break;

                if( c == 'r' )
                {
                    inpaintMask = cv::Scalar::all(0);
                    image=originalImage.clone();
                    cv::imshow("image", image);
                }

                if( c == 'i' || c == ' ' )
                {
                    cv::Mat result;
                    cv::inpaint(originalImage,inpaintMask,result,halfPatchWidth);
                    cv::imwrite("result.jpg",result);
                    inpaintMask = cv::Scalar::all(0);
                    cv::namedWindow("result");
                    cv::imshow("result",result);



                }
                if(c=='s'){
                    thickness++;
                    std::cout<<std::endl<<"Thickness = "<<thickness;
                }
                if(c=='a'){
                    thickness--;
                    std::cout<<std::endl<<"Thickness = "<<thickness;
                }
                if(thickness<3)
                    thickness=3;
                if(thickness>12)
                    thickness=12;
            }

    }





    return 0;
}
