#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
USE_HIST=$3 # whether to use class-specific context aggregation (likely want = 1)
DET_START=$4 # when to start detector-tuning (alternate policy, detector training, I used 20000)
USE_POST=$5 # whether to train posterior class-probability adjustments (assumes pretrained drl-RPN model (i.e. first set USE_POST=0))
ITERS=$6 # number of iterations (images to iterate) in training

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
NET=vgg16

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
# Below SAVE_PATH is used when saving trained weights, whereas WEIGHTS_PATH
# is used for loading existing weights
case ${DATASET} in
  pascal_voc_0712_test)
    SAVE_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/output-weights/drl-rpn-voc2007-2012-trainval+2007test/
    case ${USE_POST} in
      0)
        WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/fr-rcnn-voc2007-2012-trainval+2007test/vgg16_2012_faster_rcnn_iter_180000.ckpt
        ;;
      *)
        WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/drl-rpn-voc2007-2012-trainval+2007test/vgg16_2012_drl_rpn_iter_110000.ckpt
        ;;
    esac
    ;;
  *)
    SAVE_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/output-weights/drl-rpn-voc2007-2012-trainval/
    case ${USE_POST} in
      0)
        WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/fr-rcnn-voc2007-2012-trainval/vgg16_faster_rcnn_iter_180000.ckpt
        ;;
      *)
        WEIGHTS_PATH=/media/aleksis/B872DFD372DF950A/phd/drl-rpn/drl-rpn-voc2007-2012-trainval/vgg16_drl_rpn_iter_110000.ckpt
        ;;
    esac
    ;;
esac

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight ${WEIGHTS_PATH} \
      --save ${SAVE_PATH} \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/drl-rpn-${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --use_hist ${USE_HIST} \
      --det_start ${DET_START} \
      --use_post ${USE_POST} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            NBR_CLASSES ${NBR_CLASSES} TRAIN.STEPSIZE ${STEPSIZE} \
            DRL_RPN_TRAIN.STEPSIZE ${DRL_RPN_STEPSIZE} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight ${WEIGHTS_PATH} \
      --save ${SAVE_PATH} \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/drl-rpn-${NET}.yml \
      --net ${NET} \
      --use_hist ${USE_HIST} \
      --det_start ${DET_START} \
      --use_post ${USE_POST} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
            NBR_CLASSES ${NBR_CLASSES} TRAIN.STEPSIZE ${STEPSIZE} \
            DRL_RPN_TRAIN.STEPSIZE ${DRL_RPN_STEPSIZE} ${EXTRA_ARGS}
  fi
fi
