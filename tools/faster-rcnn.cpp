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
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/util/gpu_nms.hpp"

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

void set_input(Net<float>& caffe_net, Mat& image) {
	int im_size_min = min(image.cols, image.rows);
	int im_size_max = max(image.cols, image.rows);
	float im_scale = (float)caffe_net.net_param().scale() / im_size_min;
	if (round(im_scale * im_size_max) > caffe_net.net_param().max_size()) {
		im_scale = (float)caffe_net.net_param().max_size() / im_size_max;
	}
	float im_scale_x = (float)((int)(image.cols * im_scale / caffe_net.net_param().multiple()) * caffe_net.net_param().multiple()) / image.cols;
	float im_scale_y = (float)((int)(image.rows * im_scale / caffe_net.net_param().multiple()) * caffe_net.net_param().multiple()) / image.rows;

	Mat resized_image;
	resize(image, resized_image, Size(round(image.cols * im_scale_x), round(image.rows * im_scale_y)));

	Blob<float>& data = *caffe_net.blob_by_name("data");
	data.Reshape(1, 3, resized_image.rows, resized_image.cols);
	float* data_ = data.mutable_cpu_data();	
	for (int i = 0; i < resized_image.rows; i++) {
		for (int j = 0; j < resized_image.cols; j++) {
			data_[((0 * 3 + 0) * resized_image.rows + i) *  resized_image.cols + j] = (float)resized_image.at<Vec3b>(i, j)[0] - caffe_net.net_param().mean_value(0);
			data_[((0 * 3 + 1) * resized_image.rows + i) *  resized_image.cols + j] = (float)resized_image.at<Vec3b>(i, j)[1] - caffe_net.net_param().mean_value(1);
			data_[((0 * 3 + 2) * resized_image.rows + i) *  resized_image.cols + j] = (float)resized_image.at<Vec3b>(i, j)[2] - caffe_net.net_param().mean_value(2);
		}
	}

	Blob<float>& im_info = *caffe_net.blob_by_name("im_info");
	float* im_info_ = im_info.mutable_cpu_data();
	im_info_[0] = resized_image.rows;
	im_info_[1] = resized_image.cols;
	im_info_[2] = im_scale_x;
	im_info_[3] = im_scale_y;
	im_info_[4] = im_scale_x;
	im_info_[5] = im_scale_y;
}

void get_output(Net<float>& caffe_net, vector<BoundingBox<float> >& boxes) {
	Blob<float>& im_info = *caffe_net.blob_by_name("im_info");
	const float* im_info_ = im_info.cpu_data();
	Blob<float>& rois = *caffe_net.blob_by_name("rois");
	const float* rois_ = rois.cpu_data();
	Blob<float>& bbox_pred = *caffe_net.blob_by_name("bbox_pred");
	const float* bbox_pred_ = bbox_pred.cpu_data();
	Blob<float>& cls_prob = *caffe_net.blob_by_name("cls_prob");
	const float* cls_prob_ = cls_prob.cpu_data();
	
	int img_w = im_info_[1];
	int img_h = im_info_[0];
	float im_scale_x = im_info_[2];
	float im_scale_y = im_info_[3];
	int min_size = caffe_net.layer_by_name("proposal")->layer_param().proposal_param().min_size();
	int min_w = min_size * im_scale_x;
	int min_h = min_size * im_scale_y;

	boxes.clear();

	for (int i = 1; i < cls_prob.shape(1); i++) {
		vector<BoundingBox<float> > box(cls_prob.shape(0));
		for (int j = 0; j < cls_prob.shape(0); j++) {
			int rois_idx = j * 5;
			int cls_prob_idx = j * cls_prob.shape(1) + i;
			int bbox_pred_idx = j * bbox_pred.shape(1) + i * 4;
			box[j].x1 = rois_[rois_idx + 1] / im_scale_x;
			box[j].y1 = rois_[rois_idx + 2] / im_scale_y;
			box[j].x2 = rois_[rois_idx + 3] / im_scale_x;
			box[j].y2 = rois_[rois_idx + 4] / im_scale_y;
			box[j].score = cls_prob_[cls_prob_idx];
			box[j].transform_box(bbox_pred_[bbox_pred_idx + 0], bbox_pred_[bbox_pred_idx + 1], bbox_pred_[bbox_pred_idx + 2], bbox_pred_[bbox_pred_idx + 3], img_w, img_h, min_w, min_h);
		}
		std::sort(box.begin(), box.end());

		int* keep = (int*)calloc(box.size(), sizeof(int));
		int num_out = 0;
		float* sorted_dets = (float*)calloc(box.size() * 5, sizeof(float));
		for (int j = 0; j < box.size(); j++) {
			sorted_dets[j * 5 + 0] = box[j].x1;
			sorted_dets[j * 5 + 1] = box[j].y1;
			sorted_dets[j * 5 + 2] = box[j].x2;
			sorted_dets[j * 5 + 3] = box[j].y2;
			sorted_dets[j * 5 + 4] = box[j].score;
		}

		_nms(keep, &num_out, sorted_dets, box.size(), 5, caffe_net.net_param().nms_thresh());

		for (int j = 0; j < num_out; j++) {
			if (box[keep[j]].score >= caffe_net.net_param().conf_thresh()) {
				box[keep[j]].class_idx = i - 1;
				boxes.push_back(box[keep[j]]);
			}
		}

		free(keep);
		free(sorted_dets);
	}

}

void draw_boxes(Net<float>& caffe_net, Mat& image, vector<BoundingBox<float> >& boxes, float time) {
	for (int i = 0; i < boxes.size(); i++) {
		char text[128];
		sprintf(text, "%s(%.2f)", caffe_net.net_param().class_name(boxes[i].class_idx).c_str(), boxes[i].score);
		if (boxes[i].score >= 0.8) {
			rectangle(image, Rect(round(boxes[i].x1), round(boxes[i].y1), round(boxes[i].x2 - boxes[i].x1 + 1), round(boxes[i].y2 - boxes[i].y1 + 1)), Scalar(0, 0, 255), 2);
		}
		else {
			rectangle(image, Rect(round(boxes[i].x1), round(boxes[i].y1), round(boxes[i].x2 - boxes[i].x1 + 1), round(boxes[i].y2 - boxes[i].y1 + 1)), Scalar(255, 0, 0), 1);
		}
		putText(image, text, Point(round(boxes[i].x1), round(boxes[i].y1 + 15)), 2, 0.5, cv::Scalar(0, 0, 0), 2);
		putText(image, text, Point(round(boxes[i].x1), round(boxes[i].y1 + 15)), 2, 0.5, cv::Scalar(255, 255, 255), 1);
		if (time > 0) {
			sprintf(text, "%.3f sec", time);
			putText(image, text, Point(10, 10), 2, 0.5, cv::Scalar(0, 0, 0), 2);
			putText(image, text, Point(10, 10), 2, 0.5, cv::Scalar(255, 255, 255), 1);
		}
	}
}

void test_stream(Net<float>& caffe_net, VideoCapture& vc) {
	Mat image;
	vector<BoundingBox<float> > boxes;
	clock_t tick0 = clock();
	float time = 0;
	while (1){
		vc >> image;
		if (image.empty()) break;

		set_input(caffe_net, image);
		caffe_net.ForwardFromTo(0, caffe_net.layers().size() - 1);
		get_output(caffe_net, boxes);

		clock_t tick1 = clock();
		if (time == 0) {
			time = (float)(tick1 - tick0) / CLOCKS_PER_SEC;
		}
		else {
			time = time * 0.9 + (float)(tick1 - tick0) / CLOCKS_PER_SEC * 0.1;
		}
		tick0 = tick1;

		draw_boxes(caffe_net, image, boxes, time);
		imshow("faster-rcnn", image);
		if (waitKey(1) == 27) break; //ESC
	}
}

void test_image(Net<float>& caffe_net, Mat& image) {
	vector<BoundingBox<float> > boxes;

	set_input(caffe_net, image);
	caffe_net.ForwardFromTo(0, caffe_net.layers().size() - 1);
	get_output(caffe_net, boxes);

	draw_boxes(caffe_net, image, boxes, 0);

	imshow("faster-rcnn", image);
	waitKey(0);
}

int test(const char* command) {
	CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
	CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() != 0) {
		LOG(INFO) << "Use GPU with device ID " << gpus[0];
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	// Instantiate the caffe net.
	Net<float> caffe_net(FLAGS_model, FLAGS_weights);

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
