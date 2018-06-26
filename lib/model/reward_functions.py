# --------------------------------------------------------
# Fixate and Done action reward computations for drl-RPN
# These may differ slightly from the original CVPR 2018 paper,
# but see code for details.
# Licensed under The MIT License [see LICENSE for details]
# Written by Aleksis Pirinen
# --------------------------------------------------------
import numpy as np
from time import sleep
from model.config import cfg
from model.factory import IoU


# Give penalty for all newly evaluated voxels in obs_voxels,
# but also reward any ground-truth instances whose "coverage"
# (beyond 0.5) increases. Also update obs_gt_max_ious,
# which keeps track of each ground-truth instances max-IoU
# so far. Note that this can only happen after the intial
# fixation (no voxels get immediately observed - the agent
# first looks and then decides whether to stop immediately;
# if not, then we may begin observing voxels) 
def reward_fixate(pred_bboxes, gt_bboxes, gt_max_ious):

  if pred_bboxes is None or pred_bboxes.shape[0] == 0:
    return 0, gt_max_ious

  # Compute current max-IoUs with gts
  bbox_max_ious, _ = IoU(pred_bboxes, gt_bboxes)

  # In some (ideally rare) cases, nothing was observed
  if len(bbox_max_ious) == 0:
    return 0, gt_max_ious

  iou_thresh = cfg.DRL_RPN_TRAIN.IOU_THRESH

  # Next, compute any (potential) positive reward obtained by
  # increasing object recall for any instance(s)

  # Compute "axis = 0" max (for potential rewarding), this will
  # for each gt-instance give the best IoU given by the observed voxels
  inst_wise_max_bbox_max_ious = np.max(bbox_max_ious, axis=0)

  # Check what (if any) instances reached above IoU iou_thresh
  above_iou_thresh_idxs = inst_wise_max_bbox_max_ious >= iou_thresh

  # In this case, none of the observed voxels had good enough IoU
  # with any ground-truth instance; thus simply return the
  # fixation penalty
  if np.count_nonzero(above_iou_thresh_idxs) == 0:
    return 0, gt_max_ious

  # Extract corresponding IoUs
  inst_wise_max_bbox_max_ious[inst_wise_max_bbox_max_ious < iou_thresh] = 0
  idxs_new_better_than_prev = inst_wise_max_bbox_max_ious > gt_max_ious

  # Time to compute reward
  reward_fixate \
    = np.sum((inst_wise_max_bbox_max_ious[idxs_new_better_than_prev]
              - gt_max_ious[idxs_new_better_than_prev]))

  # Update gt_max_ious
  gt_max_ious[idxs_new_better_than_prev] \
    = inst_wise_max_bbox_max_ious[idxs_new_better_than_prev]

  # Return
  return reward_fixate, gt_max_ious


# The reward associated with the done action
def reward_done(gt_max_ious):
  iou_thresh = cfg.DRL_RPN_TRAIN.IOU_THRESH
  if np.count_nonzero(gt_max_ious <= iou_thresh) == 0:
    return np.sum(gt_max_ious - iou_thresh) / iou_thresh
  else:
    return np.min(gt_max_ious - iou_thresh) * gt_max_ious.shape[0] / iou_thresh