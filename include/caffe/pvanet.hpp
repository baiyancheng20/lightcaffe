#ifndef __PVANET_HPP__
#define __PVANET_HPP__

#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

void pvanet_init(const std::string& model_file, const std::string& weights_file, int gpu_id);
void pvanet_release();
void pvanet_detect(const unsigned char* image_data, int width, int height, int stride, std::vector<std::pair<std::string, std::vector<float> > >& boxes);

#ifdef __cplusplus
}
#endif

#endif
