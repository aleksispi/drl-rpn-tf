# --------------------------------------------------------
# Factory file for drl-RPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from time import sleep

from skimage.transform import resize as resize

from scipy.misc import imsave as imsave
from scipy.misc import imread as imread
from scipy.spatial.distance import cdist as cdist

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.blob import im_list_to_blob

from model.config import cfg
from model.nms_wrapper import nms
from model.bbox_transform import bbox_transform_inv, clip_boxes


def get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors), im_shape


def IoU(bboxes, gt_bboxes):

  # If background is included in bboxes, remove it
  if bboxes.shape[1] > 4 * cfg.NBR_CLASSES:
    bboxes = bboxes[:, 4:]

  # Get some relevant sizes and pre-allocate memory
  nbr_bboxes = bboxes.shape[0]
  tot_nbr_gt_bboxes = gt_bboxes.shape[0]

  # Now we can pre-allocate the memory needed
  bbox_max_ious = np.zeros((nbr_bboxes, tot_nbr_gt_bboxes), dtype=np.float32)

  # Used for indexing appropriately into the columns of
  # bbox_max_ious and gt_max_ious
  ctr_prev = 0
  ctr_curr = 0

  for cls_label in range(1, cfg.NBR_CLASSES):

    # Extract stuff for current class
    bboxes_curr_cls = bboxes[:, (cls_label - 1) * 4:cls_label * 4]
    gt_bboxes_curr_cls = gt_bboxes[gt_bboxes[:, 4] == cls_label][:, 0:4]
    nbr_gt_bboxes = gt_bboxes_curr_cls.shape[0]

    if nbr_gt_bboxes > 0:

      # Increase counter
      ctr_curr += nbr_gt_bboxes

      # Repmat / repeat appropriately for vectorized computations
      gt_bboxes_curr_cls = np.tile(gt_bboxes_curr_cls, [nbr_bboxes, 1])
      bboxes_curr_cls = np.repeat(bboxes_curr_cls,
                                  [nbr_gt_bboxes for _ in range(nbr_bboxes)],
                                  axis=0)

      # Intersection
      ixmin = np.maximum(bboxes_curr_cls[:, 0], gt_bboxes_curr_cls[:, 0])
      iymin = np.maximum(bboxes_curr_cls[:, 1], gt_bboxes_curr_cls[:, 1])
      ixmax = np.minimum(bboxes_curr_cls[:, 2], gt_bboxes_curr_cls[:, 2])
      iymax = np.minimum(bboxes_curr_cls[:, 3], gt_bboxes_curr_cls[:, 3])
      iw = np.maximum(ixmax - ixmin + 1., 0.)
      ih = np.maximum(iymax - iymin + 1., 0.)
      inters = iw * ih

      # Union
      uni = ((bboxes_curr_cls[:, 2] - bboxes_curr_cls[:, 0] + 1.) *
             (bboxes_curr_cls[:, 3] - bboxes_curr_cls[:, 1] + 1.) +
             (gt_bboxes_curr_cls[:, 2] - gt_bboxes_curr_cls[:, 0] + 1.) *
             (gt_bboxes_curr_cls[:, 3] - gt_bboxes_curr_cls[:, 1] + 1.) - inters)

      # IoU
      ious = inters / uni
      ious = np.reshape(ious, [nbr_bboxes, nbr_gt_bboxes])
      
      # Set everything except row-wise maxes (i.e the values by taking
      # max within each row) to zero, to indicate bbox-gt instance assignments
      # (each bounding box is only allowed to "cover"/be assigned to
      # one instance, as is the case also in mAP evaluation etc.)
      ious[ious - np.max(ious, axis=1)[:, np.newaxis] < 0] = 0

      # Insert into set of all ious (all ious contains ious for all the
      # respective, image-existing categories) and update counter/indexer
      bbox_max_ious[:, ctr_prev:ctr_curr] = ious
      ctr_prev = ctr_curr

  # Also compute maximum coverage for each gt instance
  # (for quicker future computations in e.g. RL reward)
  gt_max_ious = np.max(bbox_max_ious, axis=0)

  # Return
  return bbox_max_ious, gt_max_ious


# This helper function initializes the RL part of the network
# (trainable part, i.e., the non-pretrained part of the network)
def init_rl_variables(sess):
  uninitialized_vars = []
  print("Variables which are now initialized (i.e. they were not loaded!):\n")
  for var in tf.global_variables():
    try:
      sess.run(var)
    except tf.errors.FailedPreconditionError:
      print var
      uninitialized_vars.append(var)
  if len(uninitialized_vars) == 0:
    print("\nNo variables needed initialization (all were instead loaded!)\n")
  else:
    print("\nSuccessfully initialized variables!\n")
  sess.run(tf.variables_initializer(uninitialized_vars))


# NMS-on-the-fly (performed per fixation location)
def prepare_cls_hist(height, width, height_im, width_im):
  hist_bin_height = float(height_im) / cfg.DRL_RPN.H_HIST
  hist_bin_width  = float(width_im) / cfg.DRL_RPN.W_HIST
  # bin_ctrs will contain centroids of bins, so that we can assign
  # detections to correct part spatially
  bin_ctrs = np.zeros((cfg.DRL_RPN.H_HIST * cfg.DRL_RPN.W_HIST, 2)) 
  ctr_tot = 0
  for ctr_width in range(cfg.DRL_RPN.W_HIST):
    for ctr_height in range(cfg.DRL_RPN.H_HIST):
      x1_bin = ctr_width * hist_bin_width
      y1_bin = ctr_height * hist_bin_height
      bin_ctrs[ctr_tot, :] = np.array([(2 * x1_bin + hist_bin_width) / 2,
                                       (2 * y1_bin + hist_bin_height) / 2])
      ctr_tot += 1
  rl_in_upsample_height = int(round(float(height) / cfg.DRL_RPN.H_HIST))
  rl_in_upsample_width = int(round(float(width) / cfg.DRL_RPN.W_HIST))

  # Per-time-step NMS survivor index container
  # keeps[0] only used to keep track of survivors at each time-step;
  # whereas entry 1 to (nbr_classes - 1) [1 - 20 for pascal]
  # represent the finally chosen per-class    
  keeps_nms = [[] for _ in range(cfg.NBR_CLASSES)]
  return rl_in_upsample_height, rl_in_upsample_width, keeps_nms, bin_ctrs


# Update class-specific history
def do_hist_update(rl_input, cls_probs_uptonow, pred_bboxes_uptonow, keeps,
                   bin_ctrs, height, width, rl_in_upsamp_height,
                   rl_in_upsamp_width):

  # Initialize
  nms_prod = cfg.DRL_RPN.H_HIST * cfg.DRL_RPN.W_HIST
  cls_history = np.zeros((nms_prod, cfg.NBR_CLASSES))
  ctrs_hist = np.zeros(nms_prod)

  for j in range(1, cfg.NBR_CLASSES):
    if len(keeps[j]) > 0:

      # Extract what remains from class-specific NMS and compute bin
      # assignments in M x M grid for surviving detections
      cls_probs = cls_probs_uptonow[keeps[j], :]
      cls_boxes = pred_bboxes_uptonow[keeps[j], j * 4 : (j + 1) * 4]
      cls_box_ctrs = np.hstack([((cls_boxes[:, 0] + cls_boxes[:, 2]) / 2)\
                                [:, np.newaxis],
                                ((cls_boxes[:, 1] + cls_boxes[:, 3]) / 2)\
                                [:, np.newaxis]])
      bin_assignments = np.argmin(cdist(bin_ctrs, cls_box_ctrs), axis=0)

      # Check if something should be replaced based on current
      for some_idx in range(len(bin_assignments)):
        curr_bin_assignment = bin_assignments[some_idx]
        cls_history[curr_bin_assignment, :] \
          = (ctrs_hist[curr_bin_assignment] \
          * cls_history[curr_bin_assignment, :] \
          + cls_probs[some_idx, :]) \
          / (ctrs_hist[curr_bin_assignment] + 1)
        ctrs_hist[curr_bin_assignment] += 1

  rl_input_copy = np.copy(rl_input)
  if not rl_in_upsamp_height is None:
    for curr_bin_assignment in range(nms_prod):
      ctr_height = curr_bin_assignment % cfg.DRL_RPN.H_HIST
      ctr_width = curr_bin_assignment / cfg.DRL_RPN.W_HIST
      rl_hist_hstart = ctr_height * rl_in_upsamp_height
      rl_hist_hend = min((ctr_height + 1) * rl_in_upsamp_height, height)
      rl_hist_wstart = ctr_width * rl_in_upsamp_width
      rl_hist_wend = min((ctr_width + 1) * rl_in_upsamp_width, width)
      curr_cls_prob_vec = cls_history[curr_bin_assignment, :]\
                            [np.newaxis, np.newaxis, np.newaxis, :]
      rl_input_copy[:, rl_hist_hstart:rl_hist_hend,
                    rl_hist_wstart:rl_hist_wend, -cfg.NBR_CLASSES:] = \
      np.tile(curr_cls_prob_vec, [1, rl_hist_hend - rl_hist_hstart,
                    rl_hist_wend - rl_hist_wstart, 1])
  return rl_input_copy, cls_history.reshape((1, -1))


# Perform local NMS
def _get_nms_keep(keeps, cls_probs_uptonow, pred_bboxes_uptonow, thresh=0.0):
  for j in range(1, cfg.NBR_CLASSES):
    inds = np.where(cls_probs_uptonow[:, j] > thresh)[0]
    cls_scores = cls_probs_uptonow[inds, j]
    cls_boxes = pred_bboxes_uptonow[inds, j * 4:(j + 1) * 4]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                .astype(np.float32, copy=False)
    curr_keep = list(inds[np.asarray(nms(cls_dets, cfg.TEST.NMS),dtype=np.int32)])
    keeps[j] = curr_keep
    keeps[0] += curr_keep
  keeps[0] = list(set(keeps[0]))
  return keeps


# Updates rl state
def update_rl(rl_in, h_start, w_start, h_end, w_end, t, rois_seq,
              cls_probs_seq, bbox_preds_seq, cls_probs_uptonow,
              pred_bboxes_uptonow, keeps, im_shape, bin_ctrs, height,
              width, rl_in_upsamp_height, rl_in_upsamp_width, thresh=0.0):

  if t > 1:
    cls_probs_uptonow = cls_probs_uptonow[keeps[0], :]
    pred_bboxes_uptonow = pred_bboxes_uptonow[keeps[0], :]
    keeps[0] = []

  # Potentially perform per-time-step NMS
  if rois_seq is not None:

    # Current preds at this fix merged with survivors from previous steps
    cls_probs_uptonow = np.vstack([cls_probs_uptonow, cls_probs_seq])
    pred_bboxes = bbox_transform_inv(rois_seq, bbox_preds_seq)
    pred_bboxes = clip_boxes(pred_bboxes, im_shape)
    pred_bboxes_uptonow = np.vstack([pred_bboxes_uptonow, pred_bboxes])

  # Perform on-the-fly NMS (used when performing class-specific history updates)
  keeps = _get_nms_keep(keeps, cls_probs_uptonow, pred_bboxes_uptonow, thresh)

  # Update non-history part of RL state
  rl_in[:, h_start:h_end, w_start:w_end, :cfg.DIMS_NONHIST] = -1

  if rois_seq is not None:
    rl_in, _ = do_hist_update(rl_in, cls_probs_uptonow, pred_bboxes_uptonow,
                              keeps, bin_ctrs, height, width,
                              rl_in_upsamp_height, rl_in_upsamp_width)

  return rl_in, keeps, cls_probs_uptonow, pred_bboxes_uptonow


def _get_rect_coords(fix_h, fix_w, obs_rect_h, obs_rect_w, height, width):
  # The +1 needed due to inclusive in h1, w1, exclusive in h2, w2
  h_start = max(0, fix_h - int(obs_rect_h / 2))
  w_start = max(0, fix_w - int(obs_rect_w / 2))
  h_end = min(height, fix_h + int((obs_rect_h + 1) / 2))
  w_end = min(width, fix_w + int((obs_rect_w + 1) / 2))
  return h_start, w_start, h_end, w_end


# Updates observed voxels (orig)
def update_obs_vol(roi_obs_vol, t, height, width, height_orig, width_orig,
                   fix_h, fix_w, obs_rect_h, obs_rect_w, obs_rect_h_orig,
                   obs_rect_w_orig, h_ratio_orig, w_ratio_orig):

  # Prior to updating, check which voxels were already observed
  # at time 1,2,..., t-1
  already_obs_rois_idxs_orig = roi_obs_vol > 0
  already_obs_rois_orig = roi_obs_vol[already_obs_rois_idxs_orig]

  # Update RoI volume
  fix_h_orig = int(round(fix_h * h_ratio_orig))
  fix_w_orig = int(round(fix_w * w_ratio_orig))
  h_start_orig, w_start_orig, h_end_orig, w_end_orig \
    = _get_rect_coords(fix_h_orig, fix_w_orig, obs_rect_h_orig, obs_rect_w_orig,
                       height_orig, width_orig)
  roi_obs_vol[:, h_start_orig:h_end_orig, w_start_orig:w_end_orig, :] = t
  roi_obs_vol[already_obs_rois_idxs_orig] = already_obs_rois_orig 

  # Need to also keep track of extent of observation rectangle for updating the
  # RL state in a later stage (recall that the spatial extent of the RL state
  # "volume" is not the same as that of the original feature maps, which is why
  # we must keep track of this separately).
  h_start, w_start, h_end, w_end = _get_rect_coords(fix_h, fix_w, obs_rect_h,
                                                    obs_rect_w, height, width) 

  return roi_obs_vol, h_start, w_start, h_end, w_end, h_start_orig, w_start_orig,\
          h_end_orig, w_end_orig


# Check whether to terminate search
def _check_termination(t, done_prob, mode='train'):

  # If running in detector training mode, we do not allow agent to terminate
  # prior to fixating at least once
  if mode == 'train_det' and t == 0:
    return False, None

  # Check whether to run in training / testing mode
  if mode == 'train':
    random_done = True
  else:
    random_done = cfg.DRL_RPN_TEST.RANDOM_DONE

  # Basic termination
  if mode == 'test' and cfg.DRL_RPN_TEST.NBR_FIX > 0:
    if t + 1 == cfg.DRL_RPN_TEST.NBR_FIX:
      return True, False
    return False, False
  else:
    if random_done:
      terminate = np.random.uniform() <= done_prob
    else:
      terminate = done_prob > 0.5

  # Used if max number of fixations reached
  if t == cfg.DRL_RPN.MAX_ITER_TRAJ - 1:
    return True, terminate
  else:
    return terminate, True


# Given spatially softmaxed where-to-fix layer, sample such a location to visit
def sample_fix_loc(fix_prob, mode='train'):

  # Check whether to run in training / testing mode
  if mode == 'train':
    random_fix = True
  else:
    random_fix = cfg.DRL_RPN_TEST.RANDOM_FIX

  # Draw uniform random number for location selection
  if random_fix:
    fix_layer_cumulative = np.cumsum(fix_prob)
    u = np.random.rand()
    while u > fix_layer_cumulative[-1]: # May be round-off errors
      u = np.random.rand()
    first_smaller_than_idx_linear = np.where(u <= fix_layer_cumulative)[0][0]
  else:
    first_smaller_than_idx_linear = np.argmax(fix_prob)

  # Translate back to spatial indexing and form (h,w)-tuple
  fix_loc = np.unravel_index(first_smaller_than_idx_linear, fix_prob.shape)

  # Return (h,w)-tuple
  return fix_loc[0], fix_loc[1], first_smaller_than_idx_linear


# If experimenting with ways of reducing RoIs further, can use this
def _extract_topK_objness_per_channel_rois(roi_obs_vol, rpn_cls_prob_topK, t,
                                           not_keep_ids):
  if not cfg.DRL_RPN.USE_AGNO:
    roi_obs_vol_copy = np.zeros_like(roi_obs_vol)
    roi_obs_vol_copy[rpn_cls_prob_topK] = roi_obs_vol[rpn_cls_prob_topK]
  else:
    roi_obs_vol_copy = np.copy(roi_obs_vol)
  roi_obs_vol_copy = (roi_obs_vol_copy == t).reshape((-1))
  if cfg.DRL_RPN.USE_AGNO:
    roi_obs_vol_copy[not_keep_ids] = 0
  return roi_obs_vol_copy


# Run drl-RPN detector on an image blob
def run_drl_rpn(sess, net, blob, timers, mode, beta, im_idx=None,
                extra_args=None):

  # Extract relevant parts from blob (assume 1 img/batch)
  im_blob = blob['data']
  im_info = blob['im_info']
  if mode == 'test':
    im_shape = blob['im_shape_orig']
    im_scale = im_info[2]
  else:
    im_shape = im_info[:2]
    im_scale = 1.0
    gt_boxes = blob['gt_boxes']

  # Run initial drl-RPN processing (get base feature map etc)
  timers['init'].tic()
  net_conv, rl_in, rl_hid, rois_all, roi_obs_vol, rpn_cls_objness_topK,\
  rpn_cls_objness_vec, height, width, height_orig, width_orig, fix_rect_h,\
  fix_rect_w, h_ratio_orig, w_ratio_orig, fix_rect_h_orig, fix_rect_w_orig,\
  not_keep_ids_agno_nms = net.get_init_rl(sess, im_blob, im_info)

  # Initialize class-specific history if used
  if cfg.DRL_RPN.USE_HIST:
    rl_in_upsamp_height, rl_in_upsamp_width, keeps_nms, bin_ctrs, \
      = prepare_cls_hist(height, width, im_shape[0], im_shape[1])

  # Initialize detection containers
  cls_probs_seqs, bbox_preds_seqs, rois_seqs = [], [], []
  roi_objnesses = []
  timers['init'].toc()

  # Store and return observation canvas, used for diagnostic / visualization of
  # where drl-RPN attention has been placed in the trajectory
  obs_canvas = np.zeros((height_orig, width_orig), dtype=np.float32)
  obs_canvas_all = None

  # If training, intitialize certain containers and other things
  if mode == 'train':
    net.reset_pre_traj()
    gt_max_ious = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    rews_traj = []

  # Run search trajectory
  timers['fulltraj'].tic()
  for t in range(cfg.DRL_RPN.MAX_ITER_TRAJ): 

    # Update RL state
    if t > 0:

      # Update observation volume (used to keep track of where RoIs have been
      # forwarded for class-specific predictions)
      timers['upd-obs-vol'].tic()
      roi_obs_vol, h_start, w_start, h_end, w_end, h_start_orig, w_start_orig,\
          h_end_orig, w_end_orig \
        = update_obs_vol(roi_obs_vol, t, height, width, height_orig, width_orig,
                         fix_h, fix_w, fix_rect_h, fix_rect_w, fix_rect_h_orig,
                         fix_rect_w_orig, h_ratio_orig, w_ratio_orig)
      timers['upd-obs-vol'].toc()

      # Sequential RoI classification
      if cfg.DRL_RPN.TOPK_OBJNESS > 0 or cfg.DRL_RPN.USE_AGNO:
        roi_obs_vec_seq \
          = _extract_topK_objness_per_channel_rois(roi_obs_vol,
                                                   rpn_cls_objness_topK, t,
                                                   not_keep_ids_agno_nms)
      else:
        roi_obs_vec_seq = (roi_obs_vol == t).reshape(-1)
      
      if np.count_nonzero(roi_obs_vec_seq) > 0:

        # Classify RoIs
        timers['upd-seq'].tic()
        rois_seq = rois_all[roi_obs_vec_seq, :]
        cls_probs_seq, bbox_preds_seq = net.seq_rois_pass(sess, net_conv,
                                                          rois_seq,
                                                          mode == 'train_det')

        # Add to collection of all detections
        cls_probs_seqs.append(cls_probs_seq)
        bbox_preds_seqs.append(bbox_preds_seq)
        rois_seqs.append(rois_seq)
        rois_seq = rois_seq[:, 1:5] / im_scale
        try:
          roi_objnesses = np.concatenate([roi_objnesses,
                                          rpn_cls_objness_vec[roi_obs_vec_seq]])
        except:
          pass
        timers['upd-seq'].toc()
      else:
        rois_seq, cls_probs_seq, bbox_preds_seq = None, None, None

      # Update observation canvas (used in search visualization and
      # to keep track of fraction of spatial area covered by agent)
      obs_canvas[np.squeeze(np.sum(roi_obs_vol == t, 3) > 0)] = 1
      if im_idx is not None:
        if obs_canvas_all is None:
          obs_canvas_all = np.copy(obs_canvas[:, :, np.newaxis])
        else:
          curr_canvas = np.zeros_like(obs_canvas)
          curr_canvas[h_start_orig : h_end_orig, w_start_orig : w_end_orig] = 1
          obs_canvas_all = np.concatenate([obs_canvas_all,
                                           curr_canvas[:, :, np.newaxis]], axis=2)

      # Perform sequential pass of RoIs based on chosen regions at time t and
      # update RL state
      if cfg.DRL_RPN.USE_HIST:

        # Update RL state
        timers['upd-rl'].tic()
        if t == 1:
          cls_probs_uptonow = np.zeros((0, cfg.NBR_CLASSES))
          pred_bboxes_uptonow = np.zeros((0, 4 * cfg.NBR_CLASSES))
        rl_in, keeps_nms, cls_probs_uptonow, pred_bboxes_uptonow \
          = update_rl(rl_in, h_start, w_start, h_end, w_end, t, rois_seq,
                      cls_probs_seq, bbox_preds_seq, cls_probs_uptonow,
                      pred_bboxes_uptonow, keeps_nms, im_shape, bin_ctrs,
                      height, width, rl_in_upsamp_height,
                      rl_in_upsamp_width, 0.05)
      else:
        # Update RL state (very simple when not using any history)
        timers['upd-rl'].tic()
        rl_in[:, h_start:h_end, w_start:w_end, :] = -1
      timers['upd-rl'].toc()

      if mode == 'train':

        # Make into final form (want to compute IoU post-bbox regression, so that
        # optimization objective is closer to the final detection task)
        if rois_seq is None:
          pred_bboxes_fix = None
        else:
          pred_bboxes_fix = bbox_transform_inv(rois_seq, bbox_preds_seq)
          pred_bboxes_fix = clip_boxes(pred_bboxes_fix, im_shape)

        # Fixation reward computation
        rew_fix, gt_max_ious = net.reward_fixate(pred_bboxes_fix, gt_boxes,
                                                 gt_max_ious, t, beta)
        rews_traj.append(rew_fix)

    # Action selection (and update of conv-GRU hidden state)
    timers['action-rl'].tic()
    rl_hid, done_prob, fix_prob = net.action_pass(sess, rl_in, rl_hid, t, beta,
                                                  mode=='train')
    timers['action-rl'].toc()

    # Check for termination
    terminate, free_will = _check_termination(t, done_prob[0][0], mode)

    if terminate:
      if mode == 'train':
        rew_done = net.reward_done(fix_prob, t, gt_max_ious, free_will)
        rews_traj.append(rew_done)
        timers['fulltraj'].toc()
        rew_traj = sum(rews_traj)
        frac_gt_covered = float(np.count_nonzero(gt_max_ious >= 0.5)) \
                            / len(gt_max_ious)
        frac_gt = np.sum(gt_max_ious) / len(gt_max_ious)
      break

    # If search has not terminated, sample next spatial location to fixate
    fix_h, fix_w, fix_one_hot = sample_fix_loc(fix_prob, mode)
    if mode == 'train':
      net._ep['fix'].append(fix_one_hot)
  timers['fulltraj'].toc()

  # Potentially we need the cls-hist for posterior nudge
  if cfg.DRL_RPN.USE_POST:
    # If terminating prior to any fixation
    if t == 0:
      cls_hist = np.zeros((1, cfg.NBR_CLASSES * cfg.NBR_ANCHORS))
    else:
      _, cls_hist = do_hist_update(rl_in, cls_probs_uptonow, pred_bboxes_uptonow,
                                   keeps_nms, bin_ctrs, height, width,
                                   rl_in_upsamp_height, rl_in_upsamp_width)
  else:
    cls_hist = None

  # Collect all detections throughout the trajectory
  timers['coll-traj'].tic()
  scores, pred_bboxes, rois, fix_tracker = \
    _collect_detections(rois_seqs, bbox_preds_seqs, cls_probs_seqs, im_shape,
                        im_scale, mode, sess, net, net_conv, cls_hist,
                        roi_objnesses)
  timers['coll-traj'].toc()

  # Save visualization (if desired)
  if im_idx is not None:
    save_visualization(im_blob, im_shape, im_idx, obs_canvas_all, scores,
                       pred_bboxes, fix_tracker, 0, 1)

  # Depending on what mode, return different things
  frac_area = float(np.count_nonzero(obs_canvas)) / np.prod(obs_canvas.shape)
  if mode == 'test':
    if extra_args is not None:
      return scores, pred_bboxes, timers, [t, frac_area, [extra_args, beta, t]]
    else:
      return scores, pred_bboxes, timers, [t, frac_area]
  elif mode == 'train_det':
    return net_conv, rois, gt_boxes, im_info, timers, cls_hist
  else:
    return [rew_traj, rew_done, t, frac_area, frac_gt_covered, frac_gt,
            [gt_boxes.shape[0], beta, t]]


def _collect_detections(rois_seqs, bbox_preds_seqs, cls_probs_seqs, im_shape,
                        im_scale, mode, sess, net, net_conv, cls_hist,
                        roi_objnesses):
  if mode == 'test' or mode == 'train':
    if len(rois_seqs) > 0:
      rois = np.vstack(rois_seqs)
      bbox_pred = np.vstack(bbox_preds_seqs)
      if mode == 'test' and cfg.DRL_RPN.USE_POST:
        save = 10000
        if rois.shape[0] > save:
          roi_idxs = np.arange(len(roi_objnesses))
          roi_idxs = roi_idxs[np.argsort(-roi_objnesses)][:save]
          rois = rois[roi_idxs, :]
          bbox_pred = bbox_pred[roi_idxs, :]
        scores = net.post_hist_nudge(sess, net_conv, rois, cls_hist)
      else:
        scores = np.vstack(cls_probs_seqs)
      rois = rois[:, 1:5] / im_scale
      pred_bboxes = bbox_transform_inv(rois, bbox_pred)
      pred_bboxes = clip_boxes(pred_bboxes, im_shape)

      # Also do sequential collection
      fix_tracker = []
      for i in range(len(rois_seqs)):
        fix_tracker.append(i * np.ones(rois_seqs[i].shape[0], dtype=np.int32))
      fix_tracker = np.hstack(fix_tracker)
    else:
      scores = np.zeros((0, cfg.NBR_CLASSES))
      pred_bboxes = np.zeros((0, 4 * cfg.NBR_CLASSES))
      rois = np.zeros((0, 4))
      fix_tracker = None
    return scores, pred_bboxes, rois, fix_tracker
  else: # mode: 'train_det'
    if len(rois_seqs) > 0:
      rois = np.vstack(rois_seqs)
    else:
      rois = np.zeros((0, 5))
    return None, None, rois, None


def print_timings(timers):
  start = 1000 # burn-in --> may give fairer average runtimes
  print('init-rl: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['init'].get_avg(), timers['init'].get_avg(start),
     timers['init'].diff))
  print('fultraj: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['fulltraj'].get_avg(), timers['fulltraj'].get_avg(start),
     timers['fulltraj'].diff))
  print('upd-vol: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['upd-obs-vol'].get_avg(), timers['upd-obs-vol'].get_avg(start),
     timers['upd-obs-vol'].diff))
  print('upd-seq: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['upd-seq'].get_avg(), timers['upd-seq'].get_avg(start),
     timers['upd-seq'].diff))
  print('upd-rl:  (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['upd-rl'].get_avg(), timers['upd-rl'].get_avg(start),
     timers['upd-rl'].diff))
  print('action:  (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['action-rl'].get_avg(), timers['action-rl'].get_avg(start),
     timers['action-rl'].diff))
  print('col-tra: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
    (timers['coll-traj'].get_avg(), timers['coll-traj'].get_avg(start),
     timers['coll-traj'].diff))


def produce_det_bboxes(im, scores, det_bboxes, fix_tracker, thresh_post=0.80,
                       thresh_pre=0.0):
  """
  Based on the forward pass in a detector, extract final detection
  bounding boxes with class names and class probablity scores
  """
  class_names = cfg.CLASS_NAMES[0]
  class_names = ['bg',' aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car',
                 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'moto', 'person',
                 'plant', 'sheep', 'sofa', 'train', 'tv']
  height, width = im.shape[:2]
  colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0],
            [0.5804, 0, 0.82745], [1, 0, 1], [0, 1, 1],
            [0, 1, 0.498]]
  colors = [[0, 1, 0.498], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0],
            [0.5804, 0, 0.82745], [1, 0, 1], [0, 1, 1]]
  nbr_colors = len(colors)
  col_idx = 0
  names_and_coords = []
  cls_dets_all = []
  for j in range(1, cfg.NBR_CLASSES):
    inds = np.where(scores[:, j] > thresh_pre)[0]
    cls_scores = scores[inds, j]
    cls_bboxes = det_bboxes[inds, j * 4:(j + 1) * 4]
    curr_fix_tracker = fix_tracker[inds]
    cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis]))
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    curr_fix_tracker = curr_fix_tracker[keep]
    keep = cls_scores[keep] > thresh_post
    cls_dets = cls_dets[keep]
    curr_fix_tracker = curr_fix_tracker[keep]
    name = class_names[j]
    for jj in range(cls_dets.shape[0]):
      crop = np.squeeze(cls_dets[jj, :])
      cls_dets_all.append(crop[:4])
      coords = [crop[0], crop[1]]
      names_and_coords.append({'coords': coords,
                               'score': round(crop[4], 2),
                               'class_name': name,
                               'color': colors[col_idx],
                               'fix': curr_fix_tracker[jj]})
    if (cls_dets.shape[0]) > 0:
      col_idx += 1
      col_idx %= nbr_colors
  return cls_dets_all, names_and_coords


def save_visualization(im_blob, im_shape, im_idx, obs_canvas, cls_probs,
                       det_bboxes, fix_tracker, show_all_steps=False,
                       show_text=True):

  # Make sure image in right range
  im = im_blob[0, :, :, :]
  im -= np.min(im)
  im /= np.max(im)
  im = resize(im, (im_shape[0], im_shape[1]), order=1, mode='reflect')

  # BGR --> RGB
  im = im[...,::-1]

  # Make sure obs_canvas has same size as im
  obs_canvas = resize(obs_canvas, (im.shape[0], im.shape[1]), order=1,
                      mode='reflect')

  # Produce final detections post-NMS
  cls_dets, names_and_coords = produce_det_bboxes(im, cls_probs,
                                                  det_bboxes, fix_tracker)

  # Show image
  fig, ax = plt.subplots(1)
  ax.imshow(im)

  # Potentially we may want to show-step-by-step
  if show_all_steps:
    save_ctr = 0
    im_name = 'im' + str(im_idx + 1) + '_' + str(save_ctr) + '.jpg' 
    plt.savefig(im_name)

  # Draw all fixation rectangles
  for i in range(obs_canvas.shape[2]):

    # Extract current stuff
    if np.count_nonzero(obs_canvas[:, :, i]) == 0:
      continue
    nonzeros = np.nonzero(obs_canvas[:, :, i])
    start_x = nonzeros[1][0]
    start_y = nonzeros[0][0]
    end_x = nonzeros[1][-1]
    end_y = nonzeros[0][-1]

    # Show fixation number
    if show_text:
      ax.text(start_x, start_y, "fix " + str(i + 1), color='black', weight='bold',
              fontsize=8,
              horizontalalignment='center', verticalalignment='center',
              bbox=dict(facecolor='white', edgecolor='white', pad=-0.1))


    # Show fixation rectangle
    rect = patches.Rectangle((start_x, start_y), end_x - start_x,
                             end_y - start_y,
                             linewidth=7, edgecolor='w', facecolor='none')
    ax.add_patch(rect)

    # Potentially we may want to show-step-by-step
    if show_all_steps:
      save_ctr += 1
      im_name = 'im' + str(im_idx + 1) + '_' + str(save_ctr) + '.jpg' 
      plt.savefig(im_name)

    # Draw all detection boxes
    for j in range(len(names_and_coords)):

      # Extract current stuff
      fix = names_and_coords[j]['fix']
      if fix != i:
        continue
      coords = names_and_coords[j]['coords']
      score = names_and_coords[j]['score']
      name = names_and_coords[j]['class_name']
      color = names_and_coords[j]['color']
      cls_det = cls_dets[j]

      # Show object category + confidence
      if show_text:
        ax.text(coords[0], coords[1], name + " " + str(score),
                 weight='bold', color='black', fontsize=8,
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', edgecolor='white', pad=-0.1))

      # Show detection bounding boxes
      rect = patches.Rectangle((cls_det[0], cls_det[1]), cls_det[2] - cls_det[0],
                               cls_det[3] - cls_det[1],
                               linewidth=7, edgecolor=color, facecolor='none')
      ax.add_patch(rect)

    # Potentially we may want to show-step-by-step
    if show_all_steps:
      save_ctr += 1
      im_name = 'im' + str(im_idx + 1) + '_' + str(save_ctr) + '.jpg' 
      plt.savefig(im_name)

  # Final save / close of figure
  if ~show_all_steps:
    im_name = 'im' + str(im_idx + 1) + '.jpg' 
    plt.savefig(im_name)
  plt.close()

  # Display success message
  print("Saved image " + im_name + "!\n")
