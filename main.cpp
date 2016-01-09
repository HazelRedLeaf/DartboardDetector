/******************************
 * COMS30121 mm13354,la13815  *
 ******************************/

// header inclusion for linux lab machines
#include "/usr/include/opencv2/objdetect/objdetect.hpp"
#include "/usr/include/opencv2/opencv.hpp"
#include "/usr/include/opencv2/core/core.hpp"
#include "/usr/include/opencv2/highgui/highgui.hpp"
#include "/usr/include/opencv2/imgproc/imgproc.hpp"

// header inclusion for SNOWY
/*
#include "/usr/local/opencv-2.4/include/opencv2/objdetect/objdetect.hpp"
#include "/usr/local/opencv-2.4/include/opencv2/opencv.hpp"
#include "/usr/local/opencv-2.4/include/opencv2/core/core.hpp"
#include "/usr/local/opencv-2.4/include/opencv2/highgui/highgui.hpp"
#include "/usr/local/opencv-2.4/include/opencv2/imgproc/imgproc.hpp"
*/

// header inclusion for windows
/*
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
*/

// standard c++ and c inclusions
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>

// namespaces
using namespace std;
using namespace cv;

/** Function Headers */
void violaJones( Mat frame);
Mat circlesHoughDisplay( Mat frame, Mat &houghSpace3D);
Mat linesHoughDisplay( Mat frame);
vector<Point> circlesHoughDetect( Mat &frame, Mat &houghSpace3D);
Mat linesHoughDetect( Mat &frame, int magThreshold, int houghThreshold);
void sobel( Mat &frame, Mat &magnitude, Mat &direction);
void thresholdFunction( Mat &inputImage, Mat &outputImage, int threshold);
Mat circlesHoughSpace( Mat &thresholdedMagnitude, Mat &direction);
void transform3Dto2D( Mat &houghSpace3D,	Mat &houghSpace2D);
vector<Point> findCircleCentres(Mat &houghSpace2D);
Mat linesHoughSpace( Mat &thresholdedMagnitude, Mat &direction);
Mat getLines( Mat &houghSpace, Mat &frame);
Point* intersection( Point p1,	Point p2,	Point p3,	Point p4);
vector<Point> combineLineCircle( Mat &circles, Mat &frame);
int drawDartboards( Mat &circles, Mat &frame);
Mat closeCentres( Mat &circleSuspects);
bool isDartboardHere( Mat &image, Point topLeft, Point botRight);

// Global variables
// linux XML
String cascade_name = "dartcascade/cascade.xml";
// windows XML
//String cascade_name = "cascade.xml";
CascadeClassifier cascade;
const bool IMWRITE = true; //save working images or not
int height, width, depth, maxP;


int main(int argc, const char** argv)
{
	// read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name))
	{
		printf("--(!)Error loading\n");
		return -1;
	}

	// Detect Dartboards and Display Result
	int numBoards;
	Mat houghSpace3D, circleSuspects;

	circleSuspects = circlesHoughDisplay(frame, houghSpace3D);

  // reduce the number of suspected circles by grouping close pixels
  circleSuspects = closeCentres(circleSuspects);

	// check if the circles contain a dartboard according to corner and line detection
  combineLineCircle(circleSuspects, frame);

	// draw and count the dartboards
  numBoards = drawDartboards(circleSuspects, frame);

	if(numBoards != 1) cout << numBoards << " dartboards found." << endl; //maybe output locations too?
	else cout << numBoards << " dartboard found." << endl;

	// Save Result Image
	cv::imwrite("detected.jpg", frame);

	return 0;
}

/* @function detectAndDisplay
** @param frame is the input image
** @return void - the the output of the function is drawn onto the input
*/
void violaJones(Mat frame) // rename to Viola Jones
{
	// variables declaration
	std::vector<Rect> dartboards; //was "faces"
	Mat frame_gray; // grayscale image holder

	// prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// perform Viola-Jones Object Detection to detect dartboards
	cascade.detectMultiScale(frame_gray, dartboards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// print number of dartboards found
	std::cout << "Viola-Jones dartboards: " << dartboards.size() << std::endl;

	// draw a bounding box around dartboards found
	for (int i = 0; i < dartboards.size(); i++)
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar(0, 255, 0), 2);
	}

}

/* @function circlesHoughDisplay finds the circles in the image
** @param frame is the input image
** @param houghSpace3D is used to store the 3D hough space found for circles
** @return circleSuspects a matrix which stores the circles as the radius of the circle at its centre
*/
Mat circlesHoughDisplay(Mat frame, Mat &houghSpace3D)
{
	Mat frame_gray; // grayscale image holder

	// prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// perform Hough Transform to find circles
	vector<Point> circleCentres = circlesHoughDetect(frame_gray, houghSpace3D);

  // find the radius of the circles whose centres have been found
	Mat circleSuspects = Mat(height, width, CV_64F, 0.0);

	//find radii
	for (int i = 0; i < circleCentres.size(); i++)
	{
		double x = circleCentres.at(i).x;
		double y = circleCentres.at(i).y;

		int biggestR = 0;
		//take the largest radius whose has more than 10 votes
		for (int r = 5; r<2*depth; r++)
		{
			double votes = houghSpace3D.at<double>(y, x, r);
			if (votes > 10) biggestR = r;
		}
		circleSuspects.at<double>(y,x) = biggestR;
	}

	return circleSuspects;
}

/* @function linesHoughDisplay - orchestrates the finding of lines in the image
** @param frame is the input image
** @return linePaths is the lines detected by the Hough Space on a black image representation of the input image
*/
Mat linesHoughDisplay(Mat frame)
{
	Mat frame_gray; // grayscale image holder

	// prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Perform Hough Transform to find lines
	//24 is threshold for magnitude, 15 for the hough
	Mat houghSpace;
	houghSpace = linesHoughDetect(frame_gray, 24, 15);

	// draw lines found on a black representation of the input image
	Mat linePaths;
	linePaths = getLines(houghSpace, frame);

	return linePaths;
}

/* @function circlesHoughDetect - orchestrates the finding of the circle hough space
** @param frame is the input image
** @param houghSpace3D is the output circle Hough Space
** @return circleCentres
*/
vector<Point> circlesHoughDetect(Mat &frame, Mat &houghSpace3D)
{
	Mat magnitude, direction, thresholdedMagnitude;

	// call to sobel function
	sobel(frame, magnitude, direction);

	// threshold the magnitude
	thresholdFunction(magnitude, thresholdedMagnitude, 30);

	//un comment to save the magbitude as an image
	//cv::imwrite("debugImages/circlesThresholdedMagnitude.jpg", thresholdedMagnitude);

	// find the 3D Hough Space
	houghSpace3D = circlesHoughSpace(thresholdedMagnitude, direction);

	// turn the 3D Hough Space into 2D space (for display)
	Mat houghSpace2D;
	transform3Dto2D(houghSpace3D, houghSpace2D);

	// count the detected circles
	vector<Point> circleCentres;
	circleCentres = findCircleCentres(houghSpace2D);

	return circleCentres;
}

/* @function linesHoughDetect orchestrates the finding of the hough space for lines
** @param frame is the input image
** @param magThreshold the value to threshold the magnitudes
** @param houghThreshold the value to threshold the hough space
** @return houghSpace the matrix holding the hough space for lines
*/
Mat linesHoughDetect(Mat &frame, int magThreshold, int houghThreshold)
{
	Mat magnitude, direction, thresholdedMagnitude;

	// call to sobel function
	sobel(frame, magnitude, direction);

	// find a threshold for the magnitude and produce a thresholded image
	thresholdFunction(magnitude, thresholdedMagnitude, magThreshold);

	// find the 2D Hough Space of lines
	Mat houghSpace;
	houghSpace = linesHoughSpace(thresholdedMagnitude, direction);

	// display the Hough Space of the detected lines - make brighter for display
	cv::imwrite("debugImages/linesHoughSpace.jpg", houghSpace);

	// threshold the Hough Space before returning it
	thresholdFunction(houghSpace, houghSpace, houghThreshold);

	return houghSpace;
}

/* @function sobel performs sobel edge detection on the image provided
** @param frame is the input image
** @param magnitude the output image for edge magnitudes
** @param direction the output image for edge directions
** @return void
*/
void sobel(Mat &frame, Mat &magnitude, Mat &direction)
{
	// initialise the output matrices
	magnitude.create(frame.size(), CV_64F);
	direction.create(frame.size(), CV_64F);

	// intitialising the direction matrix to display
	Mat directionOutput;
	directionOutput.create(frame.size(), CV_64F);

	// kernels for convolution
	Mat dXkernel = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat dYkernel = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	// create a padded version of the input frame
	int kernelRadiusX = (dXkernel.size[0] - 1) / 2;
	int kernelRadiusY = (dXkernel.size[1] - 1) / 2;
	Mat paddedFrame;
	copyMakeBorder(frame, paddedFrame,kernelRadiusX, kernelRadiusX,
		kernelRadiusY, kernelRadiusY,	BORDER_REPLICATE);

	//perform convolution with the kernels
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			// initialise vars for the sums
			// over the 2 coordinates of the image
			double sumOverX = 0.0, sumOverY = 0.0;

			//convolve
			for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
			{
				for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
				{
					// get the indices
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernels
					int imageval = (int)paddedFrame.at<uchar>(imagex, imagey);
					int kernalXval = dXkernel.at<int>(kernelx, kernely);
					int kernalYval = dYkernel.at<int>(kernelx, kernely);

					// do the convolution multiplication
					sumOverX += imageval * kernalXval;
					sumOverY += imageval * kernalYval;
				}
			}

			// calculate magnitude, normalising for display
			double magnitudeRange = 4 * 255 * sqrt(2);
			double magnitudeCalculation = sqrt(sumOverX*sumOverX + sumOverY*sumOverY);
			magnitudeCalculation = (magnitudeCalculation * 255.0 / magnitudeRange);
			magnitude.at<double>(i, j) = magnitudeCalculation;

			// get the direction of each edge, in radians
			double directionInRadians = atan2(sumOverY, sumOverX);
			direction.at<double>(i, j) = directionInRadians;

			//get normalised direction for output (currently unused)
			double directionInDegrees = directionInRadians * 180.0 / 3.14159265; // ranges from -180 to 180
			directionInDegrees += 180; // shift: so it ranges from 0 to 360
			directionInDegrees = directionInDegrees*255.0 / 360;
			directionOutput.at<double>(i, j) = directionInDegrees;
		}
	}
}

/* @function thresholdFunction sets pixels above the threshold to white, and the rest to black
** @param inputImage input matrix to be thresholded
** @param outputImage output matrix with only values 0 and 255
** @param threshold the value above which pixels are considered significant
** @return void
*/
void thresholdFunction(Mat &inputImage, Mat &outputImage, int threshold)
{
	// initialise the thresholded magnitude
	outputImage.create(inputImage.size(), CV_64F);

	// go through the magnitude image
	// and threshold it
	for (int y = 0; y < inputImage.rows; y++)
	{
		for (int x = 0; x < inputImage.cols; x++)
		{
			// check if a pixel is below the threshold value
			if (inputImage.at<double>(y, x) < threshold)
			{
				// if below threshold, turn pixel black
				outputImage.at<double>(y, x) = 0;
			}
			else
			{
				// if above threshold, turn pixel white
				outputImage.at<double>(y, x) = 255;
			}
		}
	}
}

/* @function circlesHoughSpace finds the 3D hough space for circles using thresholded magnitude and direction passed in
** @param thresholdedMagnitude matrix holding the thresholded magnitude
** @param @direction matrix holding the direction of each edge in radians
** @return houghSpace 3 dimensional matrix circle hough space
*/
Mat circlesHoughSpace(Mat &thresholdedMagnitude, Mat &direction)
{
	//assign glocal dimensions
	height = direction.rows;
	width = direction.cols;
	depth = 80; // number of radii to consider!
	int dimensions[] = { height, width, depth };

	// initialise 3D matrix fillled with 0s
	Mat houghSpace = Mat(3, dimensions, CV_64F, cv::Scalar(0));

	int houghPlusX, houghMinusX, houghPlusY, houghMinusY;

	// go over all pixels of the thresholded magnitude
	for (int y = 0; y < thresholdedMagnitude.rows; y++)
	{
		for (int x = 0; x < thresholdedMagnitude.cols; x++)
		{
			// if pixel is white, there's a strong edge
			if (thresholdedMagnitude.at<double>(y, x) == 255)
			{
				// for all sizes of radii considered
				for (int r = 30; r < depth; r++)
				{
					double angleRadians = direction.at<double>(y, x);
					double angleDegrees = angleRadians / CV_PI * 180;

					// calculate points in both directions from the pixel
					houghPlusX = int(x + r * cos(angleRadians));
					houghPlusY = int(y + r * sin(angleRadians));
					houghMinusX = int(x - r * cos(angleRadians));
					houghMinusY = int(y - r * sin(angleRadians));

          // increment Hough Space
					if (houghPlusX > 0 && houghPlusY > 0 && houghPlusX < width && houghPlusY < height)
					{
						houghSpace.at<double>(houghPlusY, houghPlusX, r) += 1;
					}

					if (houghMinusX > 0 && houghMinusY > 0 && houghMinusX < width && houghMinusY < height)
					{
						houghSpace.at<double>(houghMinusY, houghMinusX, r) += 1;
					}
				}
			}
		}
	}
	return houghSpace;
}

/* @function transform3Dto2D convert the 3D Hough Space into 2D space for display
** @param houghSpace3D input hough space
** @param houghSpace2D ouput 2 dimensional hough space
** @return void
*/
void transform3Dto2D(Mat &houghSpace3D, Mat &houghSpace2D)
{
	// initialise an output matrix filled with 0s
	// height and width of the two matrices are the same
	houghSpace2D = Mat(height, width, CV_64F, 0.0);

	// sum that radii at each point to remove the third dimension
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int r = 30; r < depth; r++)
			{
				double currentRadius = houghSpace3D.at<double>(y, x, r);
				houghSpace2D.at<double>(y, x) += currentRadius;
			}
		}
	}

	//save the image
	//cv::imwrite("debugImages/circleHoughSpace2D.jpg", houghSpace2D);

	// scale the houghspace values so that the maximum is represented by 255,
	// set all values below the middle of the range to 0
	// this approach is geared towards finding circle centres

	// get the minimum and maximum value of the hough space
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(houghSpace2D, &minVal, &maxVal, &minLoc, &maxLoc);

	// the range of all the Hough Space values
	double range = maxVal - minVal;
	double bin = range / 4;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			double currentPixel = houghSpace2D.at<double>(y, x);

			// QUADRATIC MODEL + DECISION TREE

			// if current pixel falls in the top bin
			if (currentPixel <= maxVal && currentPixel >(2 * bin + minVal))
			{
				// square current pixel's value, and scale down so it fits in the image range
				houghSpace2D.at<double>(y, x) = 255 * ((currentPixel*currentPixel ) / (maxVal*maxVal));
			}
			else
			{
				// set all pixels with a low value to black
				houghSpace2D.at<double>(y, x) = 0;
			}
		}
	}
}

/* @function findCircleCentres extracts centres from matrix circle representation and returns them as a vector of points
** @param houghSpace2D 2 dimensional version of the circle hough space
** @return circleCentres vector of points corresponding to the centres of the circles
*/
vector<Point> findCircleCentres(Mat &houghSpace2D)
{
	Mat centres; vector<Point> circleCentres;

	//centres now holds the specfic centeres which we need to find to find the most likely radii
	thresholdFunction(houghSpace2D, centres, 50);

	//need to get localMAX
	for (int x = 0; x < centres.cols; x++) {
		for (int y = 0; y < centres.rows; y++) {
			Point centre;
			if (centres.at<double>(y, x) == 255) {
				centre.x = x; centre.y = y;
				circleCentres.push_back(centre);
			}
		}
	}

	return circleCentres;
}

/* @function linesHoughSpace - finds the Hough Space of lines
** @param thresholdedMagnitude mat holding the thresholded magnitude
** @param direction mat holding the edge direction at each pixel in radians
** @return houghSpace the hough space for lines, theta in the y direction, p in the x
*/
Mat linesHoughSpace(Mat &thresholdedMagnitude, Mat &direction)
{
	height = direction.rows;
	width = direction.cols;
	maxP = int(sqrt(height*height + width*width));

	// initialise 2D array fillled with 0s
	Mat houghSpace = Mat(360, maxP, CV_64F, 1.0);

	// go through all pixels of the thresholded magnitude
	for (int y = 0; y < thresholdedMagnitude.rows; y++)
	{
		for (int x = 0; x < thresholdedMagnitude.cols; x++)
		{
			//at each point on a strong edge
			if (thresholdedMagnitude.at<double>(y, x) == 255)
			{
				// turn the pixel's current direction from radians to degrees
				int currentDirection = int(direction.at<double>(y, x) * 180 / CV_PI);

				// omit horizontal and vertical lines as they are noise
				if (currentDirection == 0 || currentDirection == 90 || currentDirection == 180 || currentDirection == 270)
					continue;

				//must be divisor of 360 for the for loop wrap around
				int deltaTheta = 6;

				// calculate the min and max angle of line to be added to the hough Space
				// adding a range of lines to account for potential direction inaccuracies
				// thresholding the line hough space covers for these extra lines aded
				int minTheta = (currentDirection - deltaTheta + 360) % 360;
				int maxTheta = (minTheta + 2 * deltaTheta) % 360;

				// consider all angles within the minimum and maximum offset;
				// ensure no angle is bigger than 360 degrees (instead - wrap around)
				for (int theta = minTheta; theta != maxTheta; theta = (theta + 1) % 360)
				{
					// calculate distance from current pixel to the line
					int p = int(x * cos(theta * CV_PI / 180) + y * sin(theta * CV_PI / 180));

					// increment hough space for this pixel
					houghSpace.at<double>(theta, p) += 1;
				}
			}
		}
	}

	return houghSpace;
}

/*@function getLines translates the lines hough space to lines on the image
**@param houghSpace thresholded line hough space
**@param frame the orignal image
**@return intersectingLines each co-ordinate in this matrix stores how many lines traversed it
*/
Mat getLines(Mat &houghSpace, Mat &frame)
{
	// lineMat holds the pixels traversed per line inspected
	// intersectingLines holds acculmulated line paths and is returned from the function
	Mat lineMat(frame.size(), CV_32S, Scalar(0));
	Mat linePaths(frame.size(), CV_32S, Scalar(0));

	// go over each point in the (thresholded) hough space of the lines
	for (int theta = 0; theta < 360; theta++)
	{
		for (int p = 0; p < maxP; p++)
		{
			//each spike in the hough space represents a line on the image
			if (houghSpace.at<double>(theta, p) == 255)
			{
				// initialise two points for start and end of the line drawn
				Point p1, p2;
				// if sin of an angle is 0, the line is horizontal,
				// so calculate x first and set values of y
				if (sin(theta*CV_PI / 180) == 0)
				{
					// x coordinate of both points is equal
					p1.x = p2.x = int(p / cos(theta*CV_PI / 180));

					// y coordinates from #check left to right of original image
					p1.y = 0;
					p2.y = height;
				}
				else // calculate y and set values of x
				{
					// from top to bottom of the image
					p1.x = 0;
					p2.x = width;
					// calculate y coordinates of the two points
					p1.y = int(p / sin(theta*CV_PI / 180));
					p2.y = int((p - width*cos(theta*CV_PI / 180)) / sin(theta*CV_PI / 180));
				}

				//draw a line on a black matrix image
				cv::line(lineMat, p1, p2, Scalar(1), 1);
				linePaths += lineMat;
			}
		}
	}

	return linePaths;
}

/* @function combineLineCircle NEEDS NEW NAME, DOESNT EVEN USE LINES
** @param circles a matrix which at suspected centres, holds the radius of each suspected circle, 0 otherwise
** @param image passed to isDartboardHere for verifying image segments
** @return dartboards vector of points which are the suspected circles #change DOESNT MAKE SENSE TO RETURN THAT
*/
vector<Point> combineLineCircle(Mat &circles, Mat &image)
{
	vector<Point> dartboards;
	bool isFound;
	for (int y = 0; y < circles.rows; y++)
	{
		for (int x = 0; x < circles.cols; x++)
		{
		  double radius = circles.at<double>(y,x);
		  if (radius > 0)
		  {
		    //calculate the corners taking into account image bounds
		    Point topLeft, botRight;
	      if(x-radius > 0) topLeft.x = x - radius;
	      else topLeft.x = 0;

        if(y-radius > 0) topLeft.y = y - radius;
	      else topLeft.y = 0;

        if(x+radius < circles.cols) botRight.x = x + radius;
	      else botRight.x = circles.cols;

        if(y+radius < circles.rows) botRight.y = y + radius;
	      else botRight.y = circles.rows;

		    //find out if there really is a dartboard at this co-ordinate
		    isFound = isDartboardHere(image, topLeft, botRight);
		    if(isFound) //dartboard found
		    {
		      Point centre = (topLeft + botRight) * 0.5;
		      dartboards.push_back(centre);
		      cout << "dartboard found at: " << centre << endl;
		    }
		    else //no longer consider this centre
		    {
		      circles.at<double>(y,x) = 0;
		    }
		  }
		}
	}
	return dartboards;
}

/* @function isDartboardHere returns true if a dartboard is found in the specified area
** of the image, defined by two points passed in as parameters
** @param image the whole original image, from which the relevant segment is specified
** @param topLeft is a point defining the top left corner of the image segment to check
** @param botRight is the point defining the bottom right corner of the image segment to check
** @return isFound true if a dartboard is found in the area specified
*/
bool isDartboardHere(Mat &image, Point topLeft, Point botRight)
{
  //initialise return value
  bool isFound = false;

  //segment dimensions
  int segWidth = botRight.x - topLeft.x;
  int segHeight = botRight.y - topLeft.y;

  int diameter = (segWidth+segHeight)/2;
  Point centre = (topLeft + botRight) * 0.5;
	Mat segment = image(Rect(topLeft,botRight));


  // convert to grayscale and normalise lighting for finding lines
  Mat frame_gray;
	cvtColor(segment, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

  // corner detection
	Mat segmentConv;
	int count = 0;

	// copy the greyscale segment into another matrix of the relevant type
	frame_gray.convertTo(segmentConv, CV_32FC1);
	// create a zero matrix to save the results after corner detection
	Mat corners = Mat::zeros(segmentConv.size(), CV_32FC1);

	// use cornerHarris for corner detection
	cornerHarris(segmentConv, corners, 4, 3, 0.12, BORDER_DEFAULT);

	// normalise and scale the result matrix
	Mat cornersNorm, cornersNormScaled;
	normalize(corners, cornersNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(cornersNorm, cornersNormScaled);

	// Draw a circle around corners
	// and increment count of corners detected
	for (int y = 0; y < cornersNorm.rows; y++)
	{
		for (int x = 0; x < cornersNorm.cols; x++)
		{
			if ((int) cornersNormScaled.at<float>(y, x) > 5)
			{
				circle(cornersNormScaled, Point(x, y), 5, Scalar(0), 2, 8, 0);
				count++;
		}}
	}

	// if not enough corners are found, return false without looking for lines
	if (count < 5 )
		return isFound;

  //find lines with a lower threshold than ordinary due to targeting
  Mat houghSpace = linesHoughDetect(frame_gray, 10, 8);
  Mat linePaths = getLines(houghSpace, segment);

  //slack is the radius defining the distance from the centre within which we accept intersecting lines
  //corresponding to actual dartboard dimensions 32/340 www.darts501.com/Boards.html
	//#change 10 to 32/340 corresponds to +- 1 bullseye
  int slack = ceil( diameter / 10 );
  int minIntersects = 2;
  //check within slack bounds for enough lines intersecting
  for (int i = centre.x - slack; i < centre.x + slack; i++)
  {
    for (int j = centre.y - slack; j < centre.y + slack; j++)
    {
      if (i > topLeft.x && j > topLeft.y && i < botRight.x && j < botRight.y)
      {
	      if (linePaths.at<int>(j,i) > minIntersects)
	      {
		      isFound = true;
	      }
      }
    }
  }

  return isFound;
}

/* @function drawDartboards draws a pink bounding box around each dartboard found
** @param dartboards is a matrix that holds the radius of a dartboard at the co-ordinate of its centre
** @param image is the original image that gets drawn on
** @return numBoards the number of dartboards drawn
*/
int drawDartboards(Mat &dartboards, Mat &image)
{
  int numBoards = 0;
  for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
	    double radius = dartboards.at<double>(y,x);
	    if(radius > 0)
	    {
	      Point topLeft, botRight;
	      topLeft.x = x - radius;
		    topLeft.y = y - radius;

		    botRight.x = x + radius;
		    botRight.y = y + radius;

		    //uncomment to draw out-of-bounds boxes on the edge of the image
		     /*
		    if (topLeft.x < 0)
					topLeft.x = 0;

				if (botRight.x > width - 1)
					botRight.x = width - 1;

				if (topLeft.y < 0)
					topLeft.y = 0;

				if (botRight.y > height - 1)
					botRight.y = height - 1;
		    */

		    //draws the pink bounding box of thickness = 2 onto the image
		    rectangle(image, topLeft, botRight, Scalar(255, 0 , 255), 2);
	      numBoards++;
	    }
	  }
	}
	return numBoards;
}

/* @function closeCentres finds the highest value pixel within a local area and assigns its value to the pixel in the middle
** @param circleSuspects a matrix holding the brightest areas in the circle Hough Space
** @return a matrix holding the brightest few points assumed to be circle centres
*/
Mat closeCentres(Mat &circleSuspects)
{
	// initialise a matrix holder with 0s
	Mat closeCentres(circleSuspects.size(), CV_64F, Scalar(0));

	int localArea = 35;

	// go through the local area
	for (int y = localArea; y < circleSuspects.rows - localArea; y++)
	{
		for (int x = localArea; x < circleSuspects.cols - localArea; x++)
		{
			// if the centre pixel is 0, no cluster is assumed
			if (circleSuspects.at<double>(y, x) != 0)
			{
				double pixelSum = 0;
				double pixelCount = 0;
				double maxVotedPixel = 0;

				// compare the centre pixel with all surrounding pixels
				for (int j = y - localArea; j < y + localArea; j++)
				{
					for (int i = x - localArea; i < x + localArea; i++)
					{
						// do not compare the centre pixel with itself
						if (j == y && i == x)	continue;

						// the centre pixel's value is smaller than its neighbour's pixel value
						if (circleSuspects.at<double>(y, x) < circleSuspects.at<double>(j, i))
						{
							maxVotedPixel = circleSuspects.at<double>(j, i);
							// add the smaller value to the accummulated sum
							pixelSum += circleSuspects.at<double>(y, x);
							if (circleSuspects.at<double>(j, i) != 0)
								pixelCount++;
							// put the larger value in the centre
							circleSuspects.at<double>(y, x) = circleSuspects.at<double>(j, i);
							// set the neighbour value to 0
							circleSuspects.at<double>(j, i) = 0;
						}
						else
						{
							// add the larger value to the accummulated sum
							pixelSum += circleSuspects.at<double>(j, i);
							if (circleSuspects.at<double>(j, i) != 0)
								pixelCount++;
							maxVotedPixel = circleSuspects.at<double>(y, x);
							// set the neighbour value to 0
							circleSuspects.at<double>(j, i) = 0;
						}
					}
				}
				// average the value in the centre
				circleSuspects.at<double>(y, x) = pixelSum / pixelCount;
			}
		}
	}

	for (int y = 0; y < circleSuspects.rows; y++)
	{
		for (int x = 0; x < circleSuspects.cols; x++)
		{
			closeCentres.at<double>(y, x) = circleSuspects.at<double>(y, x);
			//this if is no longer necessary?
			if (closeCentres.at<double>(y, x) != 0)
			{
				//cout << " At (" << x << "," << y << ") = " << closeCentres.at<double>(y, x) << endl;
			}
		}
	}
	return closeCentres;
}
