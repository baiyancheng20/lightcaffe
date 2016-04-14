// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_LAYERS_HPP_
#define CAFFE_FAST_RCNN_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* ROIPoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class ROIPoolingLayer : public Layer<Dtype> {
 public:
  explicit ROIPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Dtype spatial_scale_;
  Blob<int> max_idx_;
};

template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  /**
   * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  Blob<Dtype> ones_;
  bool has_weights_;
  Dtype sigma2_;
};

template <typename Dtype>
class BoundingBox {
public:
  Dtype x1, y1, x2, y2;
  Dtype score;

  BoundingBox() {
    this->x1 = 0; this->y1 = 0; this->x2 = 0; this->y2 = 0;
    this->score = 0;
  }

  BoundingBox(Dtype x1, Dtype y1, Dtype x2, Dtype y2) {
    this->x1 = x1; this->y1 = y1; this->x2 = x2; this->y2 = y2;
    this->score = 0;
  }

  BoundingBox(Dtype x1, Dtype y1, Dtype x2, Dtype y2, Dtype score) { 
    this->x1 = x1; this->y1 = y1; this->x2 = x2; this->y2 = y2; 
    this->score = score;
  }  

  bool operator<(BoundingBox other) const { return score > other.score; }

  bool transform_box(Dtype dx, Dtype dy, Dtype dw, Dtype dh, int im_w, int im_h, Dtype min_w, Dtype min_h) {
    Dtype w = x2 - x1 + 1.0f;
    Dtype h = y2 - y1 + 1.0f;
    Dtype ctr_x = x1 + 0.5f * w;
    Dtype ctr_y = y1 + 0.5f * h;

    Dtype pred_ctr_x = dx * w + ctr_x;
    Dtype pred_ctr_y = dy * h + ctr_y;
    Dtype pred_w = exp(dw) * w;
    Dtype pred_h = exp(dh) * h;

    x1 = pred_ctr_x - 0.5f * pred_w;
    y1 = pred_ctr_y - 0.5f * pred_h;
    x2 = pred_ctr_x + 0.5f * pred_w;
    y2 = pred_ctr_y + 0.5f * pred_h;

    x1 = std::max<Dtype>(std::min<Dtype>(x1, im_w - 1), 0);
    y1 = std::max<Dtype>(std::min<Dtype>(y1, im_h - 1), 0);
    x2 = std::max<Dtype>(std::min<Dtype>(x2, im_w - 1), 0);
    y2 = std::max<Dtype>(std::min<Dtype>(y2, im_h - 1), 0);

    w = x2 - x1 + 1.0f;
    h = y2 - y1 + 1.0f;

    if (w >= min_w && h >= min_h) {
      return true;
    }
    return false;
  }
};

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
 public:
  explicit ProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //LOG(FATAL) << "Reshaping happens during the call to forward.";
  }

  virtual inline const char* type() const { return "ProposalLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	  LOG(FATAL) << "This layer does not propagate gradients.";
  }

  int feat_stride_;
  int base_size_;
  vector<float> ratios_;
  vector<float> scales_;
  int pre_nms_topn_;
  int post_nms_topn_;
  float nms_thresh_;
  int min_size_;
  vector<float> anchors_;
  int num_anchors_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
