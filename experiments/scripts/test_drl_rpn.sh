#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
USE_HIST=$3 # whether to use class-specific context aggregation (must = 1 for all models on google drive)
USE_POST=$4 # whether to use posterior class-probability adjustments
NBR_FIX=$5 # <= 0: auto-stop; >= 1: enforce exactly that nbr fixations / image

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
NET=vgg16

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    NBR_CLASSES="21"
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    NBR_CLASSES="21"
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712_test)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval+voc_2007_test"
    TEST_IMDB="voc_2012_test"
    ITERS=110000
    NBR_CLASSES="21"
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    NBR_CLASSES="81"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# Set up base weights paths according to your own system
case ${DATASET} in
  pascal_voc_0712_test)
    WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/drl-rpn-voc2007-2012-trainval-plus-2007test/vgg16_2012_drl_rpn_iter_110000.ckpt
    ;;
  *)
    WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/drl-rpn-voc2007-2012-trainval/vgg16_drl_rpn_iter_110000.ckpt
    ;;
esac

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${WEIGHTS_PATH} \
    --cfg experiments/cfgs/drl-rpn-${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --use_hist ${USE_HIST} \
    --use_post ${USE_POST} \
    --nbr_fix ${NBR_FIX} \
    --set NBR_CLASSES ${NBR_CLASSES} ANCHOR_SCALES ${ANCHORS} \
          ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${WEIGHTS_PATH} \
    --cfg experiments/cfgs/drl-rpn-${NET}.yml \
    --net ${NET} \
    --use_hist ${USE_HIST} \
    --use_post ${USE_POST} \
    --nbr_fix ${NBR_FIX} \
    --set NBR_CLASSES ${NBR_CLASSES} ANCHOR_SCALES ${ANCHORS} \
          ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
fi