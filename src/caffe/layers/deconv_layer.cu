#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }

  if (this->layer_param_.convolution_param().logging()) {
    string layer_name = this->layer_param_.name();
    string blob_name = layer_name + "_bottom0";
    this->LoggingData(blob_name.c_str(), *bottom[0]);
    blob_name = layer_name + "_param0";
    this->LoggingData(blob_name.c_str(), *this->blobs_[0]);
    if (this->bias_term_) {
      blob_name = layer_name + "_param1";
      this->LoggingData(blob_name.c_str(), *this->blobs_[1]);
    }
    blob_name = layer_name + "_top0";
    this->LoggingData(blob_name.c_str(), *top[0]);
  }
/*
  const Dtype* data = bottom[0]->cpu_data();
  FILE* fp = fopen("../test/data/temp/deconv_bottom0.txt", "w");
  for (int i = 0; i < bottom[0]->count(); ++i) {
    fprintf(fp, "%.6f\n", data[i]);
  }
  fclose(fp);
  fp = fopen("../test/data/temp/deconv_bottom0.bin", "wb");
  if (fwrite(data, sizeof(Dtype), bottom[0]->count(), fp) != bottom[0]->count()) {
    printf("Error while writing deconv_bottom0\n");
  }
  fclose(fp);
  data = this->blobs_[0]->cpu_data();
  fp = fopen("../test/data/temp/deconv_param0.txt", "w");
  for (int i = 0; i < this->blobs_[0]->count(); ++i) {
    fprintf(fp, "%.6f\n", data[i]);
  }
  fclose(fp);
  fp = fopen("../test/data/temp/deconv_param0.bin", "wb");
  if (fwrite(data, sizeof(Dtype), this->blobs_[0]->count(), fp) != this->blobs_[0]->count()) {
    printf("Error while writing deconv_param0\n");
  }
  fclose(fp);
  if (this->bias_term_) {
    data = this->blobs_[1]->cpu_data();
    fp = fopen("../test/data/temp/deconv_param1.txt", "w");
    for (int i = 0; i < this->blobs_[1]->count(); ++i) {
      fprintf(fp, "%.6f\n", data[i]);
    }
    fclose(fp);
    fp = fopen("../test/data/temp/deconv_param1.bin", "wb");
    if (fwrite(data, sizeof(Dtype), this->blobs_[1]->count(), fp) != this->blobs_[1]->count()) {
      printf("Error while writing deconv_param1\n");
    }
    fclose(fp);
  }
  data = top[0]->cpu_data();
  fp = fopen("../test/data/temp/deconv_top0.txt", "w");
  for (int i = 0; i < top[0]->count(); ++i) {
    fprintf(fp, "%.6f\n", data[i]);
  }
  fclose(fp);
  fp = fopen("../test/data/temp/deconv_top0.bin", "wb");
  if (fwrite(data, sizeof(Dtype), top[0]->count(), fp) != top[0]->count()) {
    printf("Error while writing deconv_top0\n");
  }
  fclose(fp);
*/
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(top_diff + top[i]->offset(n),
              bottom_data + bottom[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->forward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n),
              this->param_propagate_down_[0]);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DeconvolutionLayer);

}  // namespace caffe
