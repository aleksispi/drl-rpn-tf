# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride,
                   anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, cfg.NBR_ANCHORS:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


# --------------------------------------------------------
# All proposals for drl-RPN. Written by Aleksis Pirinen.
# Licensed under The MIT License [see LICENSE for details]
# Originally written by Ross Girshick and Xinlei Chen.
# --------------------------------------------------------
def proposal_layer_all(rpn_bbox_pred, im_info, anchors, rpn_cls_prob=None):
  """
  Simply returns every single RoI; drl-RPN later decides
  which are forwarded to the class-specific module.
  """

  # Get the bounding boxes
  batch_sz, height, width = rpn_bbox_pred.shape[0 : 3]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Create initial (all-zeros) observation RoI volume
  roi_obs_vol = np.zeros((batch_sz, height, width, cfg.NBR_ANCHORS),
                         dtype=np.int32)

  if cfg.DRL_RPN.USE_AGNO:
    # If this branch is used, we only consider RoIs among survivors from
    # class-agnositc NMS when choosing RoIs with drl-RPN

    pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
    nms_thresh = cfg.TEST.RPN_NMS_THRESH

    scores = rpn_cls_prob[:, :, :, cfg.NBR_ANCHORS:]
    scores = scores.reshape((-1, 1))
    keep_ids_all = np.arange(scores.shape[0], dtype=np.int32)

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    keep_ids = keep_ids_all[keep]
    not_keep_ids = np.setdiff1d(keep_ids_all, keep_ids)
  else:
    not_keep_ids = np.zeros((1, 1), dtype=np.int32)

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  rois_all = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return rois_all, roi_obs_vol, not_keep_ids