/**
    Opencv example code: overlay an image on a chessboard after computing homography matrix
    Enrique Marin
    88enrique@gmail.com
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(){

    // Variables
    VideoCapture capture;
    Mat frame, overlay;

    // Open windows
    namedWindow("video", CV_WINDOW_NORMAL);
    namedWindow("overlay", CV_WINDOW_NORMAL);

    // Open image file
    Mat pic = imread("../Images/stewie.jpg", CV_LOAD_IMAGE_COLOR);

    // Open video file
    capture.open("../Videos/chessboard-1.avi");
    //capture.open(0);

    // Check that the video was opened
    if (!capture.isOpened()){
        cout << "Cannot open video device or file!" << endl;
        return -1;
    }

    // Read new frame
    capture.read(frame);
    if (frame.empty())
        return -1;

    // Setting size variables
    overlay = Mat::zeros(pic.cols, pic.rows, CV_8UC3);
    int width_frame = frame.cols;
    int height_frame = frame.rows;
    int width_pic = pic.cols;
    int height_pic = pic.rows;

    // Read the video
    while(true){

        // Read new frame
        capture.read(frame);
        if (frame.empty())
            break;

        // Chesboard size
        Size patternsize(9,6); //interior number of corners
        vector<Point2f> corners; //this will be filled by the detected corners

        // Find corners on a chessboard pattern
        bool patternfound = findChessboardCorners(frame, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        //drawChessboardCorners(frame, patternsize, Mat(corners), patternfound);

        // If chessboard was found
        if (patternfound){

            // Set correspondences for homography
            vector<Point2f> world;
            vector<Point2f> img;
            world.push_back(Point2f(0,height_pic));
            img.push_back(Point2f(corners.at(8)));
            world.push_back(Point2f(width_pic,height_pic));
            img.push_back(Point2f(corners.at(0)));
            world.push_back(Point2f(0,0));
            img.push_back(Point2f(corners.at(53)));
            world.push_back(Point2f(width_pic,0));
            img.push_back(Point2f(corners.at(45)));

            // Find homography matrix
            // H:world->image; H-1:image->world
            Mat H = findHomography(world, img, CV_RANSAC);

            // Apply homography to the image
            warpPerspective(pic, overlay, H, cvSize(width_frame, height_frame));

            // Merge pic into frame image
            Mat thresh;
            Mat aux;
            threshold(overlay, thresh, 1, 255, CV_THRESH_BINARY_INV);
            bitwise_and(thresh, frame, aux);
            bitwise_or(aux, overlay, aux);
            overlay = aux;

            // Transform points from image to transformed image
            vector<Point2f> img_points;     // Points in image
            vector<Point2f> world_points;   // Points transformed
            img_points.push_back(Point2f(0,0));
            img_points.push_back(Point2f(pic.cols,0));
            img_points.push_back(Point2f(pic.cols,pic.rows));
            img_points.push_back(Point2f(0,pic.rows));

            // Apply homography (transformation)
            perspectiveTransform(img_points, world_points, H);

            // Draw the bounding box of the pic
            line(overlay, world_points.at(0), world_points.at(1), cvScalar(255,0,0));
            line(overlay, world_points.at(1), world_points.at(2), cvScalar(255,0,0));
            line(overlay, world_points.at(2), world_points.at(3), cvScalar(255,0,0));
            line(overlay, world_points.at(3), world_points.at(0), cvScalar(255,0,0));
        }

        // Show frame
        imshow("video", frame);
        imshow("overlay", overlay);

        if ((cvWaitKey(10) & 255) == 27) break;
    }

    // Release memory
    frame.release();
    overlay.release();

    return 0;
}


