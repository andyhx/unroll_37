#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

OUTPUT_DIR=/home/scw4750/github/unrolling/zero/A_Rnet/A_Rnet_freeze_r_with_sofmax/lmdb
#the dir that stores train.txt val.txt train_pair.txt val_pair.txt
DATA=/home/scw4750/github/unrolling/zero/A_Rnet/A_Rnet_freeze_r_with_sofmax

TOOLS=build/tools

#the true dir that store data.because the infomation in train.txt val.txt is not the absolute route.
TRAIN_DATA_ROOT=/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/data/test_img/

rm -r $OUTPUT_DIR/freeze_r_with_softmax_lmdb

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=100 \
    --resize_width=100 \
    --gray=true  \
    $TRAIN_DATA_ROOT \
    $DATA/shuffle_300w.txt \
    $OUTPUT_DIR/freeze_r_with_softmax_lmdb

echo "Creating train pair  lmdb..."


echo "Done."


