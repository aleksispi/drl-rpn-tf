from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import time
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options (for Faster R-CNN, but see drl-RPN a few 100 lines down)
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.00025

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [80000]

# Iteration intervals for showing the loss during training,
# on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images,
# introduced merely for saving GPU memory
__C.TRAIN.ASPECT_GROUPING = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'vgg16_drl_rpn'

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Whether to use all ground truth bounding boxes for training, 
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged
# as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

#
# Testing options (for Faster R-CNN, but see drl-RPN a few 10's of lines down)
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = True

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals
# to select
__C.TEST.RPN_TOP_N = 5000

#
# drl-RPN options
#

# TRAINING
__C.DRL_RPN_TRAIN = edict()

# Learning rate etc
__C.DRL_RPN_TRAIN.LEARNING_RATE = 0.00002
__C.DRL_RPN_TRAIN.GAMMA = 0.20 # Factor for reducing the learning rate
__C.DRL_RPN_TRAIN.STEPSIZE = 90000
__C.DRL_RPN_TRAIN.DISPLAY = 50
__C.DRL_RPN_TRAIN.BATCH_SIZE = 50

# Use horizontally-flipped images during training?
__C.DRL_RPN_TRAIN.USE_FLIPPED = True

# When start training of detector (-1: never train detector, >=0: train from
# that onwards)?
__C.DRL_RPN_TRAIN.DET_START = -1

# Can skip a number of images in training if desired (e.g. if starting
# from a later checkpoint file)
__C.DRL_RPN_TRAIN.IMG_START_IDX = -1

# Fixation penalty(ies) used in training
__C.DRL_RPN_TRAIN.BETAS = [0.05, 0.35]

# Use baseline?
__C.DRL_RPN_TRAIN.USE_BL = True

# Moving average factor 
__C.DRL_RPN_TRAIN.MA_WEIGHT = 0.0005

# Use final NMS'd detection boxes in rewards, or use before final NMS?
__C.DRL_RPN_TRAIN.REW_AFTER_NMS = False

# IoU-thresholds in rewards
__C.DRL_RPN_TRAIN.IOU_THRESH = 0.5

# Settings for posterior class-probability adjustment learning
__C.DRL_RPN_TRAIN.USE_POST = 0
__C.DRL_RPN_TRAIN.POST_LR = 0.001
__C.DRL_RPN_TRAIN.POST_SS = [80000]
__C.DRL_RPN_TRAIN.POST_BETAS = [0.05, 0.35]

# TESTING
__C.DRL_RPN_TEST = edict()

# Randomness during testing
__C.DRL_RPN_TEST.RANDOM_DONE = False
__C.DRL_RPN_TEST.RANDOM_FIX = False

# Fixation penalty beta
__C.DRL_RPN_TEST.BETA = 0.10

# Run a certain number of fixations in drl-RPN?
__C.DRL_RPN_TEST.NBR_FIX = 0 # 0 = automatic stopping

# GENEREAL
__C.DRL_RPN = edict()

# Use class-specific history?
__C.DRL_RPN.USE_HIST = True

# Use class-specific history to perform posterior class-probability adjustment?
# (Only possible if using class-history to guide search (USE_HIST above))
__C.DRL_RPN.USE_POST = False

# Max-objectness per layer only? (experimental option)
cfg.DRL_RPN.TOPK_OBJNESS = 0

# Use class-agnostic NMS (instead of TOP-K above)?
cfg.DRL_RPN.USE_AGNO = False

# Scaling by which to downscale feature map (increases speed)
__C.DRL_RPN.H_SCALE = 0.5
__C.DRL_RPN.W_SCALE = 0.5

# Fixation rectangle size
__C.DRL_RPN.H_FIXRECT = .25
__C.DRL_RPN.W_FIXRECT = .25

# Class-specific history into M x N bins
__C.DRL_RPN.H_HIST = 3
__C.DRL_RPN.W_HIST = 3

# Max iterations per trajectory used in training
__C.DRL_RPN.MAX_ITER_TRAJ = 13
__C.DRL_RPN.MAX_ITER_TRAJ_FLT = float(__C.DRL_RPN.MAX_ITER_TRAJ)

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default pooling mode, only 'crop' is available
__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Base feature map dimensionality
__C.DIMS_BASE = 512

# Number of classes
__C.NBR_CLASSES = 21

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Number of anchors
__C.NBR_ANCHORS = len(__C.ANCHOR_SCALES) * len(__C.ANCHOR_RATIOS)

# Number of non-hist RL channels
__C.DIMS_NONHIST = __C.DIMS_BASE + 2 * __C.NBR_ANCHORS

# Number of auxiliary inputs
__C.DIMS_AUX = 2 * __C.NBR_ANCHORS + __C.NBR_CLASSES

# Total number of RL input channels
__C.DIMS_TOT = __C.DIMS_NONHIST + __C.NBR_CLASSES 

# Number of filters for the RPN layer
__C.RPN_CHANNELS = 512

# If using COCO-trained 81-class detector on PASCAL, below is the mapping
# to corresponding 21 PASCAL-classes
__C.COCO_TO_PASCAL = [0, 5, 2, 15, 9, 40, 6, 3, 16, 57, 20, 61,
                      17, 18, 4, 1, 59, 19, 58, 7, 63]

# Set afterwards
__C.CLASS_NAMES = tuple([])

# This is used in testing to "undo" mean-std normalizations
__C.STDS_BBOX = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS),
                        (cfg.NBR_CLASSES))
__C.MEANS_BBOX = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS),
                         (cfg.NBR_CLASSES))


def get_output_dir(imdb, weights_filename, main_path=None):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  if main_path is None:
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
      weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
  else:
    if weights_filename is not None:
      outdir = osp.join(main_path, 'output', __C.EXP_DIR, imdb.name,
                        weights_filename)
    else:
      outdir = osp.join(main_path, 'output', __C.EXP_DIR, imdb.name)
    print(outdir)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)

  # If we changed something from file, must recompute things which depend
  # on each other

  # Number of anchors
  __C.NBR_ANCHORS = len(__C.ANCHOR_SCALES) * len(__C.ANCHOR_RATIOS)

  # Number of non-hist RL channels
  __C.DIMS_NONHIST = __C.DIMS_BASE + 2 * __C.NBR_ANCHORS

  # Number of auxiliary inputs
  __C.DIMS_AUX = 2 * __C.NBR_ANCHORS + __C.NBR_CLASSES

  # Total number of RL input channels
  __C.DIMS_TOT = __C.DIMS_NONHIST + __C.NBR_CLASSES

  # This is used in testing to "undo" mean-std normalizations
  __C.STDS_BBOX = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS),
                          (cfg.NBR_CLASSES))
  __C.MEANS_BBOX = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS),
                           (cfg.NBR_CLASSES))


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value

  # If we changed something from list, must recompute things which depend
  # on each other

  # Number of anchors
  __C.NBR_ANCHORS = len(__C.ANCHOR_SCALES) * len(__C.ANCHOR_RATIOS)

  # Number of non-hist RL channels
  __C.DIMS_NONHIST = __C.DIMS_BASE + 2 * __C.NBR_ANCHORS

  # Number of auxiliary inputs
  __C.DIMS_AUX = 2 * __C.NBR_ANCHORS + __C.NBR_CLASSES

  # Total number of RL input channels
  __C.DIMS_TOT = __C.DIMS_NONHIST + __C.NBR_CLASSES

  # This is used in testing to "undo" mean-std normalizations
  __C.STDS_BBOX = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS),
                          (cfg.NBR_CLASSES))
  __C.MEANS_BBOX = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS),
                           (cfg.NBR_CLASSES))