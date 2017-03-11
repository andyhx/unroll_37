#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

OUTPUT_DIR=zero/test_Rnet/lmdb
#the dir that stores train.txt val.txt train_pair.txt val_pair.txt
DATA=zero/test_Rnet/make_data

TOOLS=build/tools

#the true dir that store data.because the infomation in train.txt val.txt is not the absolute route.
TRAIN_DATA_ROOT=/home/scw4750/github/r_net/
VAL_DATA_ROOT=/home/scw4750/github/r_net/
rm -r $OUTPUT_DIR/*

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

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --gray=true  \
    $TRAIN_DATA_ROOT \
    $DATA/merge_probe.txt \
    $OUTPUT_DIR/unrolling_probe_lmdb

echo "Creating train pair  lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=128 \
    --resize_width=128 \
    --gray=true  \
    $TRAIN_DATA_ROOT \
    $DATA/merge_gallery.txt \
    $OUTPUT_DIR/unrolling_gallery_lmdb

echo "Done."


