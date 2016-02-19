// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <stdio.h>
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

static void _whctrs(const float anchor[], float& w, float& h, float& x_ctr, float& y_ctr) {
	w = anchor[2] - anchor[0] + 1;
	h = anchor[3] - anchor[1] + 1;
	x_ctr = anchor[0] + 0.5f * (w - 1);
	y_ctr = anchor[1] + 0.5f * (h - 1);
}

static void _mkanchors(const vector<float>& ws, const vector<float>& hs, float x_ctr, float y_ctr, vector<float>& anchors) {
	for (int i = 0; i < ws.size(); i++){
		anchors.push_back(x_ctr - 0.5f * (ws[i] - 1));
		anchors.push_back(y_ctr - 0.5f * (hs[i] - 1));
		anchors.push_back(x_ctr + 0.5f * (ws[i] - 1));
		anchors.push_back(y_ctr + 0.5f * (hs[i] - 1));
	}
}

static void _ratio_enum(const float anchor[], const vector<float>& ratios, vector<float>& anchors) {
	float w, h, x_ctr, y_ctr;
	_whctrs(anchor, w, h, x_ctr, y_ctr);
	float size = w * h;
	vector<float> ws, hs;
	for (int i = 0; i < ratios.size(); i++) {
		ws.push_back(round(sqrt(size / ratios[i])));
		hs.push_back(round(ws[i] * ratios[i]));
	}
	_mkanchors(ws, hs, x_ctr, y_ctr, anchors);
}

static void _scale_enum(const float anchor[], const vector<float>& scales, vector<float>& anchors) {
	float w, h, x_ctr, y_ctr;
	_whctrs(anchor, w, h, x_ctr, y_ctr);
	vector<float> ws, hs;
	for (int i = 0; i < scales.size(); i++) {
		ws.push_back(w * scales[i]);
		hs.push_back(h * scales[i]);
	}
	_mkanchors(ws, hs, x_ctr, y_ctr, anchors);
}

static void generate_anchors(int base_size, const vector<float>& ratios, const vector<float>& scales, vector<float>& anchors) {
	float base_anchor[4] = { 0, 0, base_size - 1, base_size - 1 };
	vector<float> ratio_anchors;
	_ratio_enum(base_anchor, ratios, ratio_anchors);
	for (int i = 0; i < ratio_anchors.size() / 4; i++) {
		_scale_enum(&ratio_anchors[i * 4], scales, anchors);
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ProposalParameter proposal_param = this->layer_param_.proposal_param();
  feat_stride_ = proposal_param.feat_stride();
  base_size_ = proposal_param.base_size();
  for (int i = 0; i < proposal_param.ratio_size(); i++) ratios_.push_back(proposal_param.ratio(i));
  for (int i = 0; i < proposal_param.scale_size(); i++) scales_.push_back(proposal_param.scale(i));
  pre_nms_topn_ = proposal_param.pre_nms_topn();
  post_nms_topn_ = proposal_param.post_nms_topn();
  nms_thresh_ = proposal_param.nms_thresh();
  min_size_ = proposal_param.min_size();

  for (int i = 0; i < proposal_param.copys(); i++)
    generate_anchors(base_size_, ratios_, scales_, anchors_);
  num_anchors_ = anchors_.size() / 4;

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
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";

	const Dtype* scores = bottom[0]->cpu_data();
	const Dtype* bbox_deltas = bottom[1]->cpu_data();
	const Dtype* im_info = bottom[2]->cpu_data();

	int width = bottom[0]->width();
	int height = bottom[0]->height();
	vector<BoundingBox<Dtype> > proposals;
	for (int k = 0; k < num_anchors_; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {			
				Dtype x1 = j * feat_stride_ + anchors_[k * 4 + 0];
				Dtype y1 = i * feat_stride_ + anchors_[k * 4 + 1];
				Dtype x2 = j * feat_stride_ + anchors_[k * 4 + 2];
				Dtype y2 = i * feat_stride_ + anchors_[k * 4 + 3];
				
				Dtype dx = bbox_deltas[(((0 * num_anchors_ + k) * 4 + 0) * height + i) * width + j];
				Dtype dy = bbox_deltas[(((0 * num_anchors_ + k) * 4 + 1) * height + i) * width + j];
				Dtype dw = bbox_deltas[(((0 * num_anchors_ + k) * 4 + 2) * height + i) * width + j];
				Dtype dh = bbox_deltas[(((0 * num_anchors_ + k) * 4 + 3) * height + i) * width + j];
				Dtype score = scores[((0 * num_anchors_ * 2 + num_anchors_ + k) * height + i) * width + j];
				
				BoundingBox<Dtype> proposal(x1, y1, x2, y2, score);

				if (proposal.transform_box(dx, dy, dw, dh, im_info[1], im_info[0], min_size_ * im_info[2], min_size_ * im_info[3])) {
					proposals.push_back(proposal);
				}
			}
		}
	}
	std::sort(proposals.begin(), proposals.end());

	if (pre_nms_topn_ > 0) {
		while (proposals.size() > pre_nms_topn_) proposals.pop_back();
	}

	int* keep = (int*)calloc(proposals.size(), sizeof(int));
	int num_out = 0;
	float* sorted_dets = (float*)calloc(proposals.size() * 5, sizeof(float));
	for (int i = 0; i < proposals.size(); i++) {
		sorted_dets[i * 5 + 0] = proposals[i].x1;
		sorted_dets[i * 5 + 1] = proposals[i].y1;
		sorted_dets[i * 5 + 2] = proposals[i].x2;
		sorted_dets[i * 5 + 3] = proposals[i].y2;
		sorted_dets[i * 5 + 4] = proposals[i].score;
	}

	_nms(keep, &num_out, sorted_dets, proposals.size(), 5, nms_thresh_);

	int nproposals = min<int>(num_out, post_nms_topn_);
	vector<int> shape;
	shape.push_back(nproposals); shape.push_back(5);
	top[0]->Reshape(shape);
	Dtype* top0 = top[0]->mutable_cpu_data();
	for (int i = 0; i < nproposals; i++) {
		top0[i * 5 + 0] = 0;
		top0[i * 5 + 1] = proposals[keep[i]].x1;
		top0[i * 5 + 2] = proposals[keep[i]].y1;
		top0[i * 5 + 3] = proposals[keep[i]].x2;
		top0[i * 5 + 4] = proposals[keep[i]].y2;
	}
	if (top.size() > 1) {
		shape.pop_back();
		top[1]->Reshape(shape);
		Dtype* top1 = top[1]->mutable_cpu_data();
		for (int i = 0; i < nproposals; i++) {
			top1[i] = proposals[keep[i]].score;
		}
	}

        FILE* fp = fopen("bottom.txt", "w");
        for (int k = 0; k < bottom[0]->count(); ++k)
            fprintf(fp, "%.6f\n", scores[k]);
        fclose(fp);
        fp = fopen("bbox.txt", "w");
        for (int k = 0; k < bottom[1]->count(); ++k)
            fprintf(fp, "%.6f\n", bbox_deltas[k]);
        fclose(fp);
        fp = fopen("im_info.txt", "w");
        for (int k = 0; k < bottom[2]->count()+1; ++k)
            fprintf(fp, "%.6f\n", im_info[k]);
        fclose(fp);
        fp = fopen("roi.txt", "w");
        for (int k = 0; k < nproposals; ++k) {
          for (int i = 1; i < 5; ++i)
            fprintf(fp, "%.2f ", top0[k * 5 + i]);
          fprintf(fp, "\n");
        }
        fclose(fp);
        fp = fopen("shapes.txt", "w");
        fprintf(fp, "scores: %d %d %d\n", bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
        fprintf(fp, "bboxes: %d %d %d\n", bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
        fclose(fp);

	free(keep);
	free(sorted_dets);
}

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
