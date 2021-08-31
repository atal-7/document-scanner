// document scanner using OpenCV

#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

Mat gscale, blr, canny,dil, iwarp;
vector<Point> initialpt, docpoints;
float w = 420, h = 596;

Mat pre_process(Mat img) {
	cvtColor(img, gscale, COLOR_BGR2GRAY);
	//adaptiveThreshold(gscale, gscale, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 9, 15);
	GaussianBlur(gscale, blr, Size(3, 3), 3, 0);
	//bilateralFilter(img,blr, 9, 75, 75);   //this slower than Gaussian
	Canny(blr, canny, 25, 75);
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate(canny, dil, kernel);
	return canny;
}

vector<Point> getcontours(Mat img) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conpoly(contours.size());
	vector<Point> lrg;
	int maxA=0;
	for (int i = 0; i < contours.size(); ++i) {
		int area = contourArea(contours[i]);
		string type;
		if (area > 1000) {
			float peri = arcLength(contours[i], 1);
			approxPolyDP(contours[i], conpoly[i], 0.02 * peri, 1);
			if (area > maxA  && conpoly[i].size()==4) {
				maxA = area;
				lrg = { conpoly[i][0],conpoly[i][1] ,conpoly[i][2],conpoly[i][3] };
			}
			drawContours(img, conpoly, i, Scalar(0, 255, 0), 2);
		}
	}
	return lrg;
}

vector<Point> rorder(vector<Point> pts) {
	vector<Point> pts2;
	vector<int> summ, subb;
	for (int k = 0; k < 4;++k) {
		summ.push_back(pts[k].x + pts[k].y);
		subb.push_back(pts[k].x - pts[k].y);
	}
	pts2.push_back(pts[min_element(summ.begin(), summ.end()) - summ.begin()]); //0
	pts2.push_back(pts[max_element(subb.begin(), subb.end()) - subb.begin()]); //1
	pts2.push_back(pts[min_element(subb.begin(), subb.end()) - subb.begin()]); //2
	pts2.push_back(pts[max_element(summ.begin(), summ.end()) - summ.begin()]); //3

	return pts2;
}

Mat getwarp(Mat img, vector<Point> points, float w, float h) {
	Point2f src[4] = { points[0],points[1],points[2],points[3] };     //2f floating upto 2 decimal places
	Point2f dst[4] = { {0.0f,0.0f} ,{w,0.0f}, {0.0f,h}, {w,h} };
	Mat matrix= getPerspectiveTransform(src, dst);
	warpPerspective(img, iwarp,matrix,Point(w,h));
	return iwarp;
}

int main() {
	
	Mat im = imread("Resources/pa.jpg");
	resize(im, im, Size(), 0.5, 0.5);

	//preprocessor
	Mat i2;
	i2=pre_process(im);

	//find contours
	initialpt=getcontours(i2);
	docpoints=rorder(initialpt);
	for (int i = 0; i < 4; ++i) {
		cout << initialpt[i]<<endl;
	}
	for (int i = 0; i < 4; ++i) {
		cout << docpoints[i] << endl;
	}

	//warp
	iwarp = getwarp(im, docpoints, w, h);

	//crop
	int cval = 10;
	Rect roi(cval, cval, w - (2 * cval), h - (2 * cval));
	Mat icrop = iwarp(roi);

	imshow("original", im);
	//imshow("pre", i2);
	imshow("scanned doc", icrop);
	
	waitKey(0);
}