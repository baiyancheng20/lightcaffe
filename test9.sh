#!/bin/bash
#./tools/test_net.py --gpu 0 --def ../faster-rcnn/P_SRPN_MT_A9_PVANET7.1.1_rt.pt --net ../faster-rcnn/P_SRPN_MT_A9_PVANET7.1.1_iter_600000.caffemodel --cfg experiments/cfgs/faster_rcnn_once_25anc.yml --imdb pvtdb:voc2007test:20
./tools/test_net.py --gpu 0 --def ../faster-rcnn/P_SRPN_MT_A9_PVANET7.1.1_rt_compressed.pt --net ../faster-rcnn/P_SRPN_MT_A9_PVANET7.1.1_iter_600000_compressed.caffemodel --cfg experiments/cfgs/faster_rcnn_once_25anc.yml --imdb pvtdb:voc2007test:20
