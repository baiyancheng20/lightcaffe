#include <vector>
#include <stdio.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
#ifdef _MSC_VER
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n), false);
#else
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
#endif
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }

  if (this->layer_param_.convolution_param().logging()) {
    FILE* fp = fopen("../test/data/temp/conv_bottom0.txt", "w");
    const Dtype* data = bottom[0]->cpu_data();
    for (int n = 0; n < bottom[0]->count(); ++n) {
      fprintf(fp, "%.6f\n", data[n]);
    }
    fclose(fp);
    fp = fopen("../test/data/temp/conv_param0.txt", "w");
    data = this->blobs_[0]->cpu_data();
    for (int n = 0; n < this->blobs_[0]->count(); ++n) {
      fprintf(fp, "%.6f\n", data[n]);
    }
    fclose(fp);
    if (this->bias_term_) {
      fp = fopen("../test/data/temp/conv_param1.txt", "w");
      data = this->blobs_[1]->cpu_data();
      for (int n = 0; n < this->blobs_[1]->count(); ++n) {
        fprintf(fp, "%.6f\n", data[n]);
      }
      fclose(fp);
    }
    fp = fopen("../test/data/temp/conv_top0.txt", "w");
    data = top[0]->cpu_data();
    for (int n = 0; n < top[0]->count(); ++n) {
      fprintf(fp, "%.4f\n", data[n]);
    }
    fclose(fp);
    fp = fopen("../test/data/temp/conv_shapes.txt", "w");
    fprintf(fp, "bottom: %d %d %d %d\n", bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
    fprintf(fp, "top: %d %d %d %d\n", top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width());
    fprintf(fp, "kernel: %d %d\n", this->kernel_h_, this->kernel_w_);
    fprintf(fp, "stride: %d %d\n", this->stride_h_, this->stride_w_);
    fprintf(fp, "pad: %d %d\n", this->pad_h_, this->pad_w_);
    fclose(fp);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
