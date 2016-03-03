// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/util/gpu_nms.hpp"

#ifdef _MSC_VER
#define round(x) ((int)((x) + 0.5))
#endif

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
static void _get_new_rois(const Blob<Dtype>& cls_prob, const Blob<Dtype>& bbox_pred, const Blob<Dtype>& im_info, const Blob<Dtype>& rois,
			const int min_size, const float conf_thresh, const float nms_thresh,
			vector<vector<Dtype> >& new_rois) {
	const Dtype* im_info_ = im_info.cpu_data();
	const int img_w = (int)im_info_[1];
	const int img_h = (int)im_info_[0];
	const Dtype im_scale_x = im_info_[2];
	const Dtype im_scale_y = im_info_[3];
	const int min_w = min_size * im_scale_x;
	const int min_h = min_size * im_scale_y;

	const Dtype* cls_prob_ = cls_prob.cpu_data();
	const Dtype* bbox_pred_ = bbox_pred.cpu_data();
	const Dtype* rois_ = rois.cpu_data();
	new_rois.clear();
	for (int i = 1; i < cls_prob.shape(1); i++) {
		vector<BoundingBox<Dtype> > box(cls_prob.shape(0));
		for (int j = 0; j < cls_prob.shape(0); j++) {
			int rois_idx = j * 5;
			int cls_prob_idx = j * cls_prob.shape(1) + i;
			int bbox_pred_idx = j * bbox_pred.shape(1) + i * 4;
			box[j].x1 = rois_[rois_idx + 1];
			box[j].y1 = rois_[rois_idx + 2];
			box[j].x2 = rois_[rois_idx + 3];
			box[j].y2 = rois_[rois_idx + 4];
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

		_nms(keep, &num_out, sorted_dets, box.size(), 5, nms_thresh);

		for (int j = 0; j < num_out; j++) {
			if (box[keep[j]].score >= conf_thresh) {
                                vector<Dtype> new_roi;
				new_roi.push_back(box[keep[j]].x1);
				new_roi.push_back(box[keep[j]].y1);
				new_roi.push_back(box[keep[j]].x2);
				new_roi.push_back(box[keep[j]].y2);
				new_roi.push_back(box[keep[j]].score);
				new_rois.push_back(new_roi);
			}
		}

		free(keep);
		free(sorted_dets);
	}
}

template <typename Dtype>
void ReproposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ReproposalParameter proposal_param = this->layer_param_.reproposal_param();
  nms_thresh_ = proposal_param.nms_thresh();
  min_size_ = proposal_param.min_size();
  conf_thresh_ = proposal_param.conf_thresh();

  // rois blob : holds R regions of interest, each is a 5 - tuple
  // (n, x1, y1, x2, y2) specifying an image batch index n and a
  // rectangle(x1, y1, x2, y2)
  vector<int> shape;
  shape.push_back(1); shape.push_back(5);
  top[0]->Reshape(shape);

  // scores blob : holds scores for R regions of interest
  if (top.size() > 1) {
    shape.pop_back();
    top[1]->Reshape(shape);
  }
}

template <typename Dtype>
void ReproposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	vector<vector<Dtype> > new_rois;
	_get_new_rois(*bottom[0], *bottom[1], *bottom[2], *bottom[3], min_size_, conf_thresh_, nms_thresh_, new_rois);

	vector<int> shape;
	shape.push_back(new_rois.size()); shape.push_back(5);
	top[0]->Reshape(shape);
	Dtype* top0 = top[0]->mutable_cpu_data();
	for (int i = 0; i < new_rois.size(); i++) {
		top0[i * 5 + 0] = 0;
		top0[i * 5 + 1] = new_rois[i][0];
		top0[i * 5 + 2] = new_rois[i][1];
		top0[i * 5 + 3] = new_rois[i][2];
		top0[i * 5 + 4] = new_rois[i][3];
	}
	if (top.size() > 1) {
		shape.pop_back();
		top[1]->Reshape(shape);
		Dtype* top1 = top[1]->mutable_cpu_data();
		for (int i = 0; i < new_rois.size(); i++) {
			top1[i] = new_rois[i][4];
		}
	}
	new_rois.clear();
}

#ifdef CPU_ONLY
STUB_GPU(ReproposalLayer);
#endif

INSTANTIATE_CLASS(ReproposalLayer);
REGISTER_LAYER_CLASS(Reproposal);

}  // namespace caffe
