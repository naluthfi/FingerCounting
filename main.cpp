#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main(int argc, const char** argv)
{
	//init
	Ptr<BackgroundSubtractor>bg_model=createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	Mat fgMask, bgImg, fgImg;

	//setup Video Capture device and link it to the first capture device
	VideoCapture captureDevice;
	captureDevice.open(1);

	//setup	image files used in the capture process
	Mat captureFrame;
	Mat grayscaleFrame;
	Mat thresholdFrame;

	//create a window to present the results
	namedWindow("Kamera", 1);
	//namedWindow("BG", 1);
	//namedWindow("TH", 1);

	//create a loop to capture and find faces
	while(true)
	{
		//capture a new image frame
		captureDevice>>captureFrame;

		//resize frame
		cv::Mat resized;
		cv::resize(captureFrame, resized, cv::Size(800, 600));

		//draw rectangle
		cv::rectangle(
 	 		resized,
    		cv::Point(480, 20),
    		cv::Point(780, 320),
    		cv::Scalar(255, 255, 255)
		);

		//crop image
		Mat ROI(resized, Rect(480,20,300,300));
		Mat croppedImage;

		// Copy the data into new matrix
		ROI.copyTo(croppedImage);

		//convert captured image to grayscale and  equalize
		cvtColor(croppedImage, grayscaleFrame, CV_BGR2GRAY);
		equalizeHist(grayscaleFrame, grayscaleFrame);

		GaussianBlur(grayscaleFrame,grayscaleFrame,Size(19,19),0.0,0);
        threshold(grayscaleFrame, thresholdFrame,0,255,THRESH_BINARY_INV+THRESH_OTSU);

        if(fgMask.empty()){
        	fgMask.create(croppedImage.size(), croppedImage.type());
    	}
    	bg_model->apply(croppedImage, fgMask, true ? -1:0);
    	GaussianBlur(fgMask, fgMask, Size(11,11), 3.5,3.5);
              
        // threshold mask to saturate at black and white values
        threshold(fgMask, fgMask, 10,255,THRESH_BINARY);

        // create black foreground image
        fgImg = Scalar::all(0);

        // Copy source image to foreground image only in area with white mask
        croppedImage.copyTo(fgImg, fgMask);

        //Get background image
        bg_model->getBackgroundImage(bgImg);

        // Show the results
        imshow("foreground mask", fgMask);
        imshow("foreground image", fgImg);

        Mat Temp, img;
		fgMask.convertTo(Temp, CV_8UC1);
		bilateralFilter(Temp, img, 5, 20, 20);
		vector<vector<Point> > contours;
		vector<Vec4i> hier;
    	cv::findContours(Temp, contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    	cv::Mat contourImage(Temp.size(), CV_8UC3, cv::Scalar(0,0,0));
    	for (size_t idx = 0; idx < contours.size(); idx++) {
			cv::drawContours(contourImage, contours, idx, Scalar(255, 255, 255), 2, 8, hier);

			//Rect box = boundingRect(contours[idx]); 
			//rectangle(resized, box, Scalar(0,255,0));
		}

        //convex hull
        double largest_area = 0.0;
        int largest_contour_index = 0;

		if (!contours.empty())
        {
            // Find largest contour
            for (size_t i = 0; i < contours.size(); i++)
            {
                double a = contourArea(contours[i], false);
                if (a > largest_area)
                {
                    largest_area = a;
                    largest_contour_index = i;
                }
            }

            // Draw largest contors
            Mat3b contour(fgMask.rows, fgMask.cols, Vec3b(0, 0, 0));
            drawContours(contour, contours, largest_contour_index, Scalar(255, 255, 255), CV_FILLED);

            // Find convex hull of largest contour
            vector<Point>hull;
            convexHull(contours[largest_contour_index], hull, CV_CLOCKWISE, true);

            // Draw the convex hull
            vector<vector<Point> > tmp;
            tmp.push_back(hull);
            drawContours(contour, tmp, 0, Scalar(0, 0, 255), 3);

            imshow("Contour", contour);
        }

		//print output
		imshow("Kamera", resized);
		//imshow("TH", thresholdFrame);
    	//imshow("Contour", contourImage);

		//pause for 33ms
		waitKey(33);
	}

	return 0;
}
