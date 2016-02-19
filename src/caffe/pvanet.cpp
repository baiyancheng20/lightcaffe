#include "caffe/pvanet.hpp"
#include "caffe/net.hpp"

static caffe::Net<float>* caffe_net = NULL;

void pvanet_init(const std::string& model_file, const std::string& weights_file, int gpu_id) {
  if(caffe_net == NULL) {
    std::vector<int> gpus;
    gpus.push_back(gpu_id);
    caffe_net = new caffe::Net<float>(model_file, weights_file, gpus);
  }
}

void pvanet_release() {
  if(caffe_net) {
    delete caffe_net;
    caffe_net = NULL;
  }
}

void pvanet_detect(const unsigned char* image_data, int width, int height, int stride, std::vector<std::pair<std::string, std::vector<float> > >& boxes) {
  caffe_net->Detect(image_data, width, height, stride, boxes);
}
