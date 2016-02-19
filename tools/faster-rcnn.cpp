#ifdef RUN_TIME
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/net.hpp"

#ifdef _MSC_VER
#define round(x) ((int)((x) + 0.5))
#endif

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::string;
using caffe::vector;
using std::ostringstream;
using namespace cv;
using namespace caffe;

DEFINE_string(gpu, "0",
	"Optional; run in GPU mode on given device IDs separated by ','."
	"Use '-gpu all' to run on all available GPUs. The effective training "
	"batch size is multiplied by the number of devices.");
DEFINE_string(model, "",
	"The model definition protocol buffer text file..");
DEFINE_string(weights, "",
	"Optional; the pretrained weights to initialize finetuning, "
	"separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(cam, "0", "camera index");
DEFINE_string(width, "", "camera input width");
DEFINE_string(height, "", "camera input height");
DEFINE_string(img, "", "image path");
DEFINE_string(vid, "", "video path");
DEFINE_string(db, "", "db path");


static void get_gpus(vector<int>* gpus) {
	if (FLAGS_gpu == "all") {
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) {
			gpus->push_back(i);
		}
	}
	else if (FLAGS_gpu.size()) {
		vector<string> strings;
		boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); ++i) {
			gpus->push_back(boost::lexical_cast<int>(strings[i]));
		}
	}
	else {
		CHECK_EQ(gpus->size(), 0);
	}
}

void draw_boxes(Mat& image, vector<pair<string, vector<float> > >& boxes, float time) {
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
		if (time > 0) {
			sprintf(text, "%.3f sec", time);
			putText(image, text, Point(10, 10), 2, 0.5, cv::Scalar(0, 0, 0), 2);
			putText(image, text, Point(10, 10), 2, 0.5, cv::Scalar(255, 255, 255), 1);
		}
	}
}

void test_stream(Net<float>& caffe_net, VideoCapture& vc) {
	Mat image;
	vector<pair<string, vector<float> > > boxes;
	clock_t tick0 = clock();
	float time = 0;
	while (1){
		vc >> image;
		if (image.empty()) break;

		caffe_net.Detect(image.data, image.cols, image.rows, image.step.p[0], boxes);

		clock_t tick1 = clock();
		if (time == 0) {
			time = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
		}
		else {
			time = time * 0.9 + (float)(tick1 - tick0) / CLOCKS_PER_SEC * 0.1;
		}
		tick0 = tick1;

		draw_boxes(image, boxes, time);
		imshow("faster-rcnn", image);
		if (waitKey(1) == 27) break; //ESC
	}
}

void test_image(Net<float>& caffe_net, Mat& image) {
	vector<pair<string, vector<float> > > boxes;

	caffe_net.Detect(image.data, image.cols, image.rows, image.step.p[0], boxes);

	draw_boxes(image, boxes, 0);

	imshow("faster-rcnn", image);
	waitKey(0);
}

int test(const char* command) {
	CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
	CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
		
	// Instantiate the caffe net.
	vector<int> gpus;
	get_gpus(&gpus);
	Net<float> caffe_net(FLAGS_model, FLAGS_weights, gpus);

	if (strcmp(command, "live") == 0) {
		VideoCapture vc(boost::lexical_cast<int>(FLAGS_cam));
		if (!vc.isOpened()) {
			LOG(ERROR) << "Cannot open camera(" << FLAGS_cam << ")";
			return -1;
		}
		vc.set(CV_CAP_PROP_FRAME_WIDTH, boost::lexical_cast<int>(FLAGS_width));
		vc.set(CV_CAP_PROP_FRAME_HEIGHT, boost::lexical_cast<int>(FLAGS_height));
		test_stream(caffe_net, vc);
	}
	else if (strcmp(command, "snapshot") == 0) {
		Mat image = imread(FLAGS_img);
		if (!image.data) {
			LOG(ERROR) << "Cannot open image: " << FLAGS_img;
			return -1;
		}
		test_image(caffe_net, image);
	}
	else if (strcmp(command, "video") == 0) {
		VideoCapture vc(boost::lexical_cast<int>(FLAGS_vid));
		if (!vc.isOpened()) {
			LOG(ERROR) << "Cannot open video: " << FLAGS_vid;
			return -1;
		}
		test_stream(caffe_net, vc);
	}
	else if (strcmp(command, "database") == 0) {
		FILE* fp = fopen(FLAGS_db.c_str(), "r");
		if (!fp) {
			LOG(ERROR) << "Cannot open db: " << FLAGS_db;
			return -1;			
		}
		char path[256];
		while (fgets(path, 256, fp)) {
			path[strlen(path) - 1] = '\0';
			Mat image = imread(path);
			if (!image.data) {
				LOG(ERROR) << "Cannot open image: " << path;
				continue;
			}
			test_image(caffe_net, image);
		}
		fclose(fp);
	}

	destroyAllWindows();

	return 0;
}

int main(int argc, char** argv) {
	// Print output to stderr (while still logging).
	FLAGS_alsologtostderr = 1;
	// Usage message.
	gflags::SetUsageMessage("command line brew\n"
		"usage: caffe <command> <args>\n\n"
		"commands:\n"
		"  live           \n"
		"  snapshot       \n"
		"  video          \n"
		"  database       \n");
	// Run tool or show usage.
	caffe::GlobalInit(&argc, &argv);
	
	if (argc == 2) {
		test(argv[1]);
	}
	else {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/faster-rcnn");
	}

	return 0;
}
#else
#include <stdio.h>
int main(int argc, char** argv) {
	printf("For runttime code!\n");
}
#endif