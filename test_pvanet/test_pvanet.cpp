#include "pvanet.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void draw_boxes(Mat& image, vector<pair<string, vector<float> > >& boxes) {
	for (int i = 0; i < boxes.size(); i++) {
		char text[128];
		sprintf(text, "%s(%.2f)", boxes[i].first.c_str(), boxes[i].second[4]);
		if (boxes[i].second[4] >= 0.8) {
			rectangle(image, Rect(round(boxes[i].second[0]), round(boxes[i].second[1]), round(boxes[i].second[2] - boxes[i].second[0] + 1), round(boxes[i].second[3] - boxes[i].second[1] + 1)), Scalar(0, 0, 255), 2);
		}
		else {
			rectangle(image, Rect(round(boxes[i].second[0]), round(boxes[i].second[1]), round(boxes[i].second[2] - boxes[i].second[0] + 1), round(boxes[i].second[3] - boxes[i].second[1] + 1)), Scalar(255, 0, 0), 1);
		}
		putText(image, text, Point(round(boxes[i].second[0]), round(boxes[i].second[1] + 15)), 2, 0.5, cv::Scalar(0, 0, 0), 2);
		putText(image, text, Point(round(boxes[i].second[0]), round(boxes[i].second[1] + 15)), 2, 0.5, cv::Scalar(255, 255, 255), 1);
	}
}

int main(int argc, char** argv) {
	pvanet_init("PVANET7.0.1.pt", "PVANET7.0.1.caffemodel", 0);
	Mat image = imread("004545.jpg");
	vector<pair<string, vector<float> > > boxes;
	pvanet_detect(image.data, image.cols, image.rows, image.step.p[0], boxes);
	draw_boxes(image, boxes);
	imshow("test_pvanet", image);
	waitKey(0);
	pvanet_release();
	return 0;
}
