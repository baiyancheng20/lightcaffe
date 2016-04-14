#!/bin/bash
# Usage parse_log_pvt.sh faster_rcnn.log
# It creates the following four text files, each containing a table:
#     caffe.log.stage1 (columns: '#Iters Seconds BboxLoss ClsLoss')

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log_pvt.sh /path/to/your.log"
exit
fi
LOG=`basename $1`

# split logs into multiple files for each stage
csplit -sk $LOG '/Stage /' {9999}
rm xx00 xx02 xx05
mv xx01 $LOG.stage1_1rpn.log
mv xx03 $LOG.stage1_2rcnn.log
mv xx04 $LOG.stage2_1rpn.log
mv xx06 $LOG.stage2_2rcnn.log

# generate input files for plotting


# stage1_rpn
# For extraction of time since this line contains the start time
grep '] Creating training ' $LOG.stage1_1rpn.log > aux.txt
grep ', loss = ' $LOG.stage1_1rpn.log >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ': rpn_loss_bbox = ' $LOG.stage1_1rpn.log | awk '{print $11}' > aux1.txt
grep ': rpn_cls_loss = ' $LOG.stage1_1rpn.log | awk '{print $11}' > aux2.txt

# Extracting elapsed seconds
$DIR/extract_seconds_pvt.py aux.txt aux3.txt

# Generating
echo '#Iters Seconds BboxLoss ClsLoss'> $LOG.stage1_1rpn.log.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.stage1_1rpn.log.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt

# stage1_rcnn
# For extraction of time since this line contains the start time
grep '] Creating training ' $LOG.stage1_2rcnn.log > aux.txt
grep ', loss = ' $LOG.stage1_2rcnn.log >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ': loss_bbox = ' $LOG.stage1_2rcnn.log | awk '{print $11}' > aux1.txt
grep ': loss_cls = ' $LOG.stage1_2rcnn.log | awk '{print $11}' > aux2.txt

# Extracting elapsed seconds
$DIR/extract_seconds_pvt.py aux.txt aux3.txt

# Generating
echo '#Iters Seconds BboxLoss ClsLoss'> $LOG.stage1_2rcnn.log.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.stage1_2rcnn.log.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt



# stage2_rpn
# For extraction of time since this line contains the start time
grep '] Creating training ' $LOG.stage2_1rpn.log > aux.txt
grep ', loss = ' $LOG.stage2_1rpn.log >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ': rpn_loss_bbox = ' $LOG.stage2_1rpn.log | awk '{print $11}' > aux1.txt
grep ': rpn_cls_loss = ' $LOG.stage2_1rpn.log | awk '{print $11}' > aux2.txt

# Extracting elapsed seconds
$DIR/extract_seconds_pvt.py aux.txt aux3.txt

# Generating
echo '#Iters Seconds BboxLoss ClsLoss'> $LOG.stage2_1rpn.log.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.stage2_1rpn.log.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt



# stage2_rcnn
# For extraction of time since this line contains the start time
grep '] Creating training ' $LOG.stage2_2rcnn.log > aux.txt
grep ', loss = ' $LOG.stage2_2rcnn.log >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ': loss_bbox = ' $LOG.stage2_2rcnn.log | awk '{print $11}' > aux1.txt
grep ': loss_cls = ' $LOG.stage2_2rcnn.log | awk '{print $11}' > aux2.txt

# Extracting elapsed seconds
$DIR/extract_seconds_pvt.py aux.txt aux3.txt

# Generating
echo '#Iters Seconds BboxLoss ClsLoss'> $LOG.stage2_2rcnn.log.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.stage2_2rcnn.log.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt

rm $LOG.stage1_1rpn.log
rm $LOG.stage1_2rcnn.log
rm $LOG.stage2_1rpn.log
rm $LOG.stage2_2rcnn.log
