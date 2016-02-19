#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/*
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(round,
    y[index] = __int2float_rn( __float2int_rd( x[index] ) ) );

*/
/*
template <typename Dtype>
__global__ void round_kernel(const int n, const Dtype* a,
    const Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __int2float_rn(__float2int_rd(a[index]));
  }
}

template <>
void caffe_gpu_round<float>(const int N, const float* a, float* y) {
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}
*/

template <typename Dtype>
void PrecisionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  if (bottom[0] != top[0]) {
    caffe_copy(count, bottom_data, top_data);
  }

  caffe_gpu_scal(count, fast_mult_1, top_data);
  caffe_gpu_round<Dtype>(count, top_data, top_data);
  caffe_gpu_scal(count, fast_mult_2, top_data);
}

template <typename Dtype>
void PrecisionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
    // 

}

INSTANTIATE_LAYER_GPU_FUNCS(PrecisionLayer);


}  // namespace caffe
