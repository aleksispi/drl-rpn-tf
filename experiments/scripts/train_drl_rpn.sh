#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3 # only vgg16 supported currently
WEIGHTS=$4
USE_HIST=$5 # whether to use class-specific context aggregation
DET_START=$6 # when to start detector-tuning (alternate policy, detector training)
USE_POST=$7 # whether to train posterior class-probability adjustments (assumes pretrained drl-RPN model)
ITERS=$8 # number of iterations (images to iterate) in training
SET_IDX=$9 # consider refactoring this out (used to run on several work stations)

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:9:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    DRL_RPN_STEPSIZE="90000"
    NBR_CLASSES="21"
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    DRL_RPN_STEPSIZE="90000"
    NBR_CLASSES="21"
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712_test)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval+voc_2007_test"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    DRL_RPN_STEPSIZE="90000"
    NBR_CLASSES="21"
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[400000]"
    DRL_RPN_STEPSIZE="390000"
    NBR_CLASSES="81"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# Set up paths according to your own system
# Below MAIN_PATH is used when saving trained weights, whereas WEIGHTS_PATH
# is used for loading existing weights
case $SET_IDX in
  1)
    MAIN_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/set${SET_IDX}
    WEIGHTS_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/data/pretrained_models/${WEIGHTS}
    ;;
  2)
    MAIN_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/set${SET_IDX}
    WEIGHTS_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/data/pretrained_models/${WEIGHTS}
    ;;
  3)
    MAIN_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/set${SET_IDX}
    WEIGHTS_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/data/pretrained_models/${WEIGHTS}
    ;;
  4)
    MAIN_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/set${SET_IDX}
    WEIGHTS_PATH=/home/aleksis/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/data/pretrained_models/${WEIGHTS}
    ;;
  5)
    MAIN_PATH=/media/aleksis/B872DFD372DF950A/phd/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/set${SET_IDX}
    WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/data/pretrained_models/${WEIGHTS}
    ;;
  6)
    MAIN_PATH=/media/aleksis/B872DFD372DF950A/phd/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/set${SET_IDX}
    WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/faster_rcnn_tf_data_and_output/tf-faster-rcnn/rl/data/pretrained_models/${WEIGHTS}
    ;;
  *)
    echo "The set-idx / weight path does not exist!"
    exit
    ;;
esac

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight ${WEIGHTS_PATH} \
      --main ${MAIN_PATH} \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/drl-rpn-${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --use_hist ${USE_HIST} \
      --det_start ${DET_START} \
      --use_post ${USE_POST} \
      --set_idx ${SET_IDX} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            NBR_CLASSES ${NBR_CLASSES} TRAIN.STEPSIZE ${STEPSIZE} \
            DRL_RPN_TRAIN.STEPSIZE ${DRL_RPN_STEPSIZE} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight ${WEIGHTS_PATH} \
      --main ${MAIN_PATH} \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/drl-rpn-${NET}.yml \
      --net ${NET} \
      --use_hist ${USE_HIST} \
      --det_start ${DET_START} \
      --use_post ${USE_POST} \
      --set_idx ${SET_IDX} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            NBR_CLASSES ${NBR_CLASSES} TRAIN.STEPSIZE ${STEPSIZE} \
            DRL_RPN_TRAIN.STEPSIZE ${DRL_RPN_STEPSIZE} ${EXTRA_ARGS}
  fi
fi
