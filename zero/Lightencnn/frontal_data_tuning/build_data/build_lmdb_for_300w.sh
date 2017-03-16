#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

OUTPUT_DIR=/home/scw4750/github/unrolling/zero/Lightencnn/unroll_frontal_data_fine_tuning/data_lmdb
#the dir that stores train.txt val.txt train_pair.txt val_pair.txt
DATA=/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/data

TOOLS=build/tools

#the true dir that store data.because the infomation in train.txt val.txt is not the absolute route.
TRAIN_DATA_ROOT=/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/frontal_layer/data/test_true_frontal/
OUTPUT_NAME=$OUTPUT_DIR/300w
if [ -d "$OUTPUT_NAME" ]; then
  rm -r $OUTPUT_NAME
fi


# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=100
  RESIZE_WIDTH=100
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating image lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --gray=true  \
    --shuffle=true \
    $TRAIN_DATA_ROOT \
    $DATA/shuffle_300w.txt \
    $OUTPUT_NAME

echo "Done."


