#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void PrecisionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Parse params
  PrecisionParameter param = this->layer_param_.precision_param();
  
  exponent = this->layer_param_.precision_param().exponent();
  num_bits = this->layer_param_.precision_param().num_bits();
  
  Dtype scale_factor = exponent * 2.0;
  int fraction_bits = num_bits - 1;

  fast_mult_1 = pow(2, fraction_bits) / scale_factor;
  fast_mult_2 = scale_factor / pow(2, fraction_bits);

  LOG(INFO) << "exponent: " << exponent << "\t bit_precision: " << num_bits;
}

template <typename Dtype>
void PrecisionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
void PrecisionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  // NOT IMPLEMENTED
}

template <typename Dtype>
void PrecisionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  // NOT IMPLEMENTED
}


#ifdef CPU_ONLY
STUB_GPU(PrecisionLayer);
#endif

INSTANTIATE_CLASS(PrecisionLayer);
REGISTER_LAYER_CLASS(Precision);

}  // namespace caffe
