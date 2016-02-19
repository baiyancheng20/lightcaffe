#ifdef __cplusplus
extern "C" {
#endif

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh);

#ifdef __cplusplus
}
#endif
