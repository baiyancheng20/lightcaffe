#include <boost/thread.hpp>
#include "caffe/layer.hpp"
#include <stdio.h>

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template <typename Dtype>
void Layer<Dtype>::LoggingData(const char* blob_name, Blob<Dtype>& blob) {
  char filename[1024];
  sprintf(filename, "../test/data/temp/%s.bin", blob_name);

  FILE* fp = fopen(filename, "wb");

  int ndim = blob.num_axes();
  int total_size = 1;
  fwrite(&ndim, sizeof(int), 1, fp);
  for (int i = 0; i < ndim; ++i) {
    int shape_i = blob.shape(i);
    total_size *= shape_i;
    fwrite(&shape_i, sizeof(int), 1, fp);
  }

  if (total_size != blob.count()) {
    printf("[ERROR] Size mismatch! %d != %d\n", total_size, blob.count());
  }

  Dtype* data = blob.mutable_cpu_data();
  fwrite(data, sizeof(Dtype), blob.count(), fp);

  fclose(fp);
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
