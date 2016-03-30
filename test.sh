#!/bin/bash
#./tools/test_net.py --gpu 0 --def SRPN_PVANET7.1.1_faster_rcnn_rt_compressed.pt --net SRPN_PVANET7.1.1_faster_rcnn_once_iter_900000_compressed.caffemodel --cfg experiments/cfgs/faster_rcnn_once_25anc.yml --imdb pvtdb:voc2007test:20
./tools/test_net.py --gpu 0 --def ../test/meta/pva/7.1.1/full/SRPN_PVANET7.1.1_faster_rcnn_test_rt.pt --net ../test/data/pva/7.1.1/full/SRPN_PVANET7.1.1_faster_rcnn_once_iter_900000.caffemodel --cfg experiments/cfgs/faster_rcnn_once_25anc.yml --imdb pvtdb:voc2007test:20
