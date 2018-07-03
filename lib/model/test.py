# --------------------------------------------------------
# Tensorflow drl-RPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
from time import sleep

from utils.timer import Timer
from utils.blob import im_list_to_blob
from utils.statcoll import StatCollector

from model.config import cfg, get_output_dir, cfg_from_list
from model.nms_wrapper import nms
from model.factory import run_drl_rpn, print_timings, get_image_blob



def im_detect(sess, net, im, timers, im_idx=None, nbr_gts=None):

  # Setup image blob
  blobs = {}
  blobs['data'], im_scales, blobs['im_shape_orig'] = get_image_blob(im)
  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]])

  # Run drl-RPN
  scores, pred_bboxes, timers, stats \
    = run_drl_rpn(sess, net, blobs, timers, 'test', cfg.DRL_RPN_TEST.BETA,
                  im_idx, nbr_gts)

  return scores, pred_bboxes, timers, stats


def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.00):
  """Test a drl-RPN network on an image database."""

  # Set numpy's random seed
  np.random.seed(cfg.RNG_SEED)

  nbr_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(nbr_images)] for _ in range(cfg.NBR_CLASSES)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  _t_drl_rpn = {'init': Timer(), 'fulltraj': Timer(),
                'upd-obs-vol': Timer(), 'upd-seq': Timer(), 'upd-rl': Timer(),
                'action-rl': Timer(), 'coll-traj': Timer()}
  avg_traj = 0.0
  avg_frac = 0.0

  # Create StatCollector (tracks various drl-RPN test statistics)
  stat_strings = ['#fix/img', 'exploration']
  sc = StatCollector(nbr_images, stat_strings, False)

  # Try getting gt-info if available
  try:
    gt_roidb = imdb.gt_roidb()
  except:
    gt_roidb = None

  # Visualize search trajectories?
  do_visualize = cfg.DRL_RPN_TEST.DO_VISUALIZE

  # Can be convenient to run from some other image, especially if visualizing,
  # but having nbr_ims_eval = nbr_images and start_idx = 0 --> regular testing!
  nbr_ims_eval = nbr_images
  start_idx = 0 
  end_idx = start_idx + nbr_ims_eval

  # Test drl-RPN on the test images
  for i in range(start_idx, end_idx):

    # Need to know image index if performing visualizations
    if do_visualize: im_idx = i
    else: im_idx = None

    # Try extracting gt-info for diagnostics (possible for voc 2007)
    if gt_roidb is None:
      nbr_gts = None
    else:
      nbr_gts = gt_roidb[i]['boxes'].shape[0]

    # Detect!
    im = cv2.imread(imdb.image_path_at(i))
    _t['im_detect'].tic()
    scores, boxes, _t_drl_rpn, stats = im_detect(sess, net, im, _t_drl_rpn,
                                                 im_idx, nbr_gts)
    _t['im_detect'].toc()

    # Update and print some stats
    sc.update(0, stats)
    sc.print_stats(False)

    _t['misc'].tic()
    # skip j = 0, because it's the background class
    for j in range(1, cfg.NBR_CLASSES):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, cfg.NBR_CLASSES)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, cfg.NBR_CLASSES):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('\nim_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, nbr_images, _t['im_detect'].average_time,
            _t['misc'].average_time))
    #print_timings(_t_drl_rpn) # uncomment for some timing details!

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir, start_idx, end_idx)