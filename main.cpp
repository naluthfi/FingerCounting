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

        Mat coba, coba2;
        erode(fgMask,coba, 0, Point(-1, -1), 2, 1, 1);
        dilate(coba,coba2, 0, Point(-1, -1), 2, 1, 1);


        //Get background image
        bg_model->getBackgroundImage(bgImg);

        // Show the results

        imshow("coba", coba2);

        Mat Temp, img;
		coba2.convertTo(Temp, CV_8UC1);
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
            Mat3b contour(coba2.rows, coba2.cols, Vec3b(0, 0, 0));
            drawContours(contour, contours, largest_contour_index, Scalar(255, 255, 255), 2, 8, hier);

            // Find convex hull of largest contour
            vector<Point>hulll;
            convexHull(contours[largest_contour_index], hulll, CV_CLOCKWISE, true);

            // Draw the convex hull
            vector<vector<Point> > tmp;
            tmp.push_back(hulll);
            //drawContours(contour, tmp, 0, Scalar(0, 0, 255), 3);

            //give point
        //convex hulls
        vector<vector<Point> >hull(contours.size());
        vector<vector<int> > hullsI(contours.size()); 
        vector<vector<Vec4i> > defects(contours.size());
        for (int i = 0; i < contours.size(); i++)
        {
            convexHull(contours[i], hull[i], false);
            convexHull(contours[i], hullsI[i], false); 
            if(hullsI[i].size() > 3 )            
            {
                convexityDefects(contours[i], hullsI[i], defects[i]);
            }
        }
        //REQUIRED contour is detected,then convex hell is found and also convexity defects are found and stored in defects

        if (largest_area>100){
            drawContours(contour, hull, largest_contour_index, Scalar(255, 0, 255), 3, 8, vector<Vec4i>(), 0, Point());

            /// Draw convexityDefects
            for(int j=0; j<defects[largest_contour_index].size(); ++j)
            {
                const Vec4i& v = defects[largest_contour_index][j];
                float depth = v[3] / 256;
                int count=0;
                if (depth > 10 && depth < 100) //  filter defects by depth
                {
                    int startidx = v[0]; Point ptStart(contours[largest_contour_index][startidx]);
                    int endidx = v[1]; Point ptEnd(contours[largest_contour_index][endidx]);
                    int faridx = v[2]; Point ptFar(contours[largest_contour_index][faridx]);

                    line(contour, ptStart, ptEnd, Scalar(0, 255, 0), 1);
                    line(contour, ptStart, ptFar, Scalar(0, 255, 0), 1);
                    line(contour, ptEnd, ptFar, Scalar(0, 255, 0), 1);
                    circle(contour, ptFar, 4, Scalar(0, 0, 255), 2);
                    count++;
                }
                printf("ada %d jari \n", count);
            }
        }

            imshow("Contour", contour);
        }

        

		//print output
		imshow("Kamera", resized);
        //ximshow("foreground mask", fgMask);
		//imshow("TH", thresholdFrame);
    	//imshow("Contour", contourImage);

		//pause for 33ms
		waitKey(33);
	}

	return 0;
}
