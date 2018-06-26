# --------------------------------------------------------
# Tensorflow drl-RPN-based Faster R-CNN detector.
# Licensed under The MIT License [see LICENSE for details]
# Partially written* by Aleksis Pirinen, rest is based on original code by
# Xinlei Chen.
# * All parts of drl-RPN by Aleksis Pirinen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
from time import sleep

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer, proposal_layer_all
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer,\
                                              proposal_target_layer_wo_scores
from utils.visualization import draw_bounding_boxes

from model.config import cfg
from model.reward_functions import reward_fixate, reward_done
from model.factory import sample_fix_loc


class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._variables_to_fix = {}


  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] \
                  / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])


  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1],
                                      [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf


  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)


  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_top_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors,
                                     self._num_anchors],
                                    [tf.float32, tf.float32],name="proposal_top")
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])
    return rois, rpn_scores


  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._mode, self._feat_stride,
                                     self._anchors],
                                    [tf.float32, tf.float32], name="proposal")
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])
    return rois, rpn_scores


  def _proposal_layer_all(self, rpn_bbox_pred, rpn_cls_prob=None,
                          name='proposal_all'):
    with tf.variable_scope(name) as scope:
      rois_all, roi_obs_vol, not_keep_ids = tf.py_func(proposal_layer_all,
                                             [rpn_bbox_pred, self._im_info,
                                              self._anchors, rpn_cls_prob],
                                             [tf.float32, tf.int32, tf.int32],
                                             name=name)
      self._predictions['rois_all'] = rois_all
      self._predictions['roi_obs_vol'] = roi_obs_vol
      self._predictions['not_keep_ids'] = not_keep_ids


  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]


  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"),
                             [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) \
                * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) \
                * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                       [pre_pool_size, pre_pool_size],
                                       name="crops")
    return slim.max_pool2d(crops, [2, 2], padding='SAME')


  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)


  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,\
      rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
                                            [rpn_cls_score, self._gt_boxes,
                                             self._im_info, self._feat_stride,
                                             self._anchors, self._num_anchors],
                                            [tf.float32, tf.float32, tf.float32,
                                             tf.float32], name="anchor_target")
      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
    return rpn_labels


  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights,\
      bbox_outside_weights = tf.py_func(proposal_target_layer,
                                        [rois, roi_scores, self._gt_boxes,
                                         self._num_classes],
                                        [tf.float32, tf.float32, tf.float32,
                                         tf.float32, tf.float32, tf.float32],
                                        name="proposal_target")
      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,self._num_classes*4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,
                                     self._num_classes * 4])
      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
      return rois, roi_scores


  def _proposal_target_layer_wo_scores(self, rois, name):
    with tf.variable_scope(name) as scope:
      rois, labels, bbox_targets, bbox_inside_weights,\
      bbox_outside_weights = tf.py_func(proposal_target_layer_wo_scores,
                                        [rois, self._gt_boxes,self._num_classes],
                                        [tf.float32, tf.float32, tf.float32,
                                         tf.float32, tf.float32],
                                        name="proposal_target_wo")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,self._num_classes*4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,
                                     self._num_classes * 4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
      return rois


  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] \
                  / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] \
                  / np.float32(self._feat_stride[0])))
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                          [height, width,
                                           self._feat_stride,
                                           self._anchor_scales,
                                           self._anchor_ratios],
                                          [tf.float32, tf.int32],
                                          name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length


  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights,
                      bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. \
                      / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
    return loss_box


  # Currently only supports training of detector head
  def _add_losses(self, post_hist=False):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      # RCNN, class loss
      if post_hist:
        cls_score = self._predictions['cls_score_hist']
      else:
        cls_score = self._predictions['cls_score_seq']
      label = tf.reshape(self._proposal_targets['labels'], [-1])
      cross_entropy = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                          logits=cls_score, labels=label))
      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred_seq']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets,
                                      bbox_inside_weights,
                                      bbox_outside_weights)
      if post_hist:
        loss = cross_entropy
      else:
        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        loss = cross_entropy + loss_box
      # Want to use regularizer, but only for the actual weights used in the
      # detector updates!
      all_reg_losses = tf.losses.get_regularization_losses()
      if post_hist:
        ii = 0
        while 'post_hist' not in all_reg_losses[ii].name:
          ii += 1
        relevant_reg_losses = all_reg_losses[ii:]
        reg_loss = tf.add_n(relevant_reg_losses, 'regu')
        self._losses['total_loss_hist'] = loss + reg_loss
      else:
        ii = 0
        while 'fc6' not in all_reg_losses[ii].name:
          ii += 1
        relevant_reg_losses = all_reg_losses[ii:]
        reg_loss = tf.add_n(relevant_reg_losses, 'regu')
        self._losses['total_loss'] = loss + reg_loss


  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training,
                      weights_initializer=initializer, scope='rpn_conv/3x3')
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1],
                                trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None,
                                scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2,
                                                'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape,
                                               'rpn_cls_prob_reshape')
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]),
                             axis=1, name='rpn_cls_pred')
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape,
                                       self._num_anchors * 2, 'rpn_cls_prob')
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1],
                                trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None,
                                scope='rpn_bbox_pred')
    if is_training:
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred,'rois')
      rpn_labels = self._anchor_target_layer(rpn_cls_score, 'anchor')
      # Try to have a deterministic order for the computing graph, for
      # reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(rois, roi_scores, 'rpn_rois')
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, 'rois')
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, 'rois')
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois


  def _region_classification(self, fc7, is_training, initializer,
                             initializer_bbox, reuse=None):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                     weights_initializer=initializer,
                                     trainable=is_training, reuse=reuse,
                                     activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training, reuse=reuse,
                                     activation_fn=None, scope='bbox_pred')
    self._predictions['cls_score_seq'] = cls_score
    self._predictions['cls_pred_seq'] = cls_pred
    self._predictions['cls_prob_seq'] = cls_prob
    self._predictions['bbox_pred_seq'] = bbox_pred


  ############# DRL-RPN ADDITIONAL COMPONENTS -- START #########################


  def train_drl_rpn(self, sess, lr_rl, sc, stats):

    # Compute baseline
    if cfg.DRL_RPN_TRAIN.USE_BL:
      bl_means_done = np.empty(cfg.DRL_RPN.MAX_ITER_TRAJ)
      bl_stds_done = np.empty(cfg.DRL_RPN.MAX_ITER_TRAJ)
      bl_means_fix = np.empty(cfg.DRL_RPN.MAX_ITER_TRAJ)
      bl_stds_fix = np.empty(cfg.DRL_RPN.MAX_ITER_TRAJ)
      for idx in range(cfg.DRL_RPN.MAX_ITER_TRAJ):
        if len(self._bl_done[idx]) > 0:
          bl_means_done[idx] = np.mean(self._bl_done[idx])
          bl_stds_done[idx] = np.std(self._bl_done[idx])
          bl_means_fix[idx] = np.mean(self._bl_fix[idx])
          bl_stds_fix[idx] = np.std(self._bl_fix[idx])
      bl_stds_done[bl_stds_done == 0] = 1
      bl_stds_fix[bl_stds_fix == 0] = 1

    curr_batch_avg_loss = 0
    for idx in range(len(self._ep_batch['x'])):

      # Potentially normalize to mean 0, std 1
      ep_rew_done = self._ep_batch['rew_done'][idx]
      ep_rew_fix = self._ep_batch['rew_fix'][idx]
      if cfg.DRL_RPN_TRAIN.USE_BL:
        ep_rew_done -= bl_means_done[: len(ep_rew_done)]
        ep_rew_fix -= bl_means_fix[: len(ep_rew_fix)]
        ep_rew_done /= bl_stds_done[: len(ep_rew_done)]
        ep_rew_fix /= bl_stds_fix[: len(ep_rew_fix)]

      # Make sure that the final entry of fixate rews is identically equal to
      # zero (done action does not affect fixate quality)
      ep_rew_fix[-1] = 0

      # Update grad buffer
      feed_dict_grad_comp \
        = {self._rl_in: self._ep_batch['x'][idx],
           self._rl_hid: self._ep_batch['h'][idx],
           self._aux_done_info: self._ep_batch['aux'][idx],
           self._done_labels: self._ep_batch['done'][idx],
           self._fix_labels: self._ep_batch['fix'][idx],
           self._advs_done: self._ep_batch['rew_done'][idx],
           self._advs_fix: self._ep_batch['rew_fix'][idx],
           self._cond_switch_fix: self._ep_batch['cond'][idx]}
      ce_done, ce_fix, ce_done_rew_prod, ce_fix_rew_prod, loss_rl, new_grads\
        = sess.run([self._predictions['ce_done'],
                    self._predictions['ce_fix'],
                    self._predictions['ce_done_rew_prod'],
                    self._predictions['ce_fix_rew_prod'],
                    self._predictions['loss_rl'],
                    self._predictions['new_grads']],
                   feed_dict=feed_dict_grad_comp)
      curr_batch_avg_loss += loss_rl

      # Accumulate gradients to buffer
      for ix, grad in enumerate(new_grads):
        self._grad_buffer[ix] += grad
    
    curr_batch_avg_loss /= cfg.DRL_RPN_TRAIN.BATCH_SIZE
    sc.update(curr_batch_avg_loss, stats)

    # Update policy parameters
    feed_dict_upd_grads \
      = {self._batch_grad[grad_idx]: self._grad_buffer[grad_idx] \
         for grad_idx in range(len(self._batch_grad))}
    feed_dict_upd_grads.update({self._lr_rl_in: lr_rl})
    sess.run(self._update_grads, feed_dict=feed_dict_upd_grads)

    # Reset gradient buffer etc.
    self.reset_after_gradient()


  def _collect_traj(self, t, free_will, nbr_gts):
    # Stack inputs, hidden states, action grads, and rewards for this episode
    epx = np.vstack(self._ep['x'])
    eph = np.vstack(self._ep['h'])
    ep_aux = np.vstack(self._ep['aux'])
    ep_done = np.vstack([0] * t + [free_will])
    ep_fix = np.hstack(self._ep['fix'])
    ep_rew_done = np.hstack(self._ep['rew_done'])
    ep_rew_fix = np.hstack(self._ep['rew_fix'])
    ep_rew_done = np.flipud(np.cumsum(np.flipud(ep_rew_done)))
    ep_rew_fix = np.flipud(np.cumsum(np.flipud(ep_rew_fix)))

    # This ensures that images with many gt's are not "more valuable" than
    # images with few gts
    if nbr_gts > 0:
      ep_rew_done /= nbr_gts
      ep_rew_fix /= nbr_gts

    # Add to collections
    self._ep_batch['x'].append(epx)
    self._ep_batch['h'].append(eph)
    self._ep_batch['aux'].append(ep_aux)
    self._ep_batch['done'].append(ep_done)
    self._ep_batch['fix'].append(ep_fix)
    self._ep_batch['rew_done'].append(ep_rew_done)
    self._ep_batch['rew_fix'].append(ep_rew_fix)
    self._ep_batch['cond'].append(int(free_will))

    # Add to baselines
    if cfg.DRL_RPN_TRAIN.USE_BL:
      for len_ctr in range(len(ep_rew_done)):
        self._bl_done[len_ctr].append(ep_rew_done[len_ctr])
        self._bl_fix[len_ctr].append(ep_rew_fix[len_ctr])


  # Done reward
  def reward_done(self, fix_prob, t, gt_max_ious, free_will=True):
    if free_will:
      # The fixate labels need to be handled with care,
      # depending on whether or not we terminated by an
      # action (done-action) or by running out of iterations!
      self._ep['fix'].append(0)
      rew_done = reward_done(gt_max_ious)
      self._ep['rew_done'].append(rew_done)
      # Fixate rewards should never be held "accountable"
      # due to stopping condition 
      self._ep['rew_fix'].append(0.0)
    else:
      _, _, fix_one_hot = sample_fix_loc(fix_prob, 'train')
      self._ep['fix'].append(fix_one_hot)
      rew_fixate = 0.0
      rew_done = -0.5 # so agent may learn stop before forced
      self._ep['rew_fix'].append(rew_fixate)
      self._ep['rew_done'].append(rew_done) 

    # Now we also collect all throughout this trajectory
    self._collect_traj(t, free_will, gt_max_ious.shape[0])

    # Return reward
    return rew_done


  # Fixate reward
  def reward_fixate(self, pred_bboxes, gt_boxes, gt_max_ious, t, beta):

    # Separation of rewards (described in paper) does not appear useful when 
    # training for various beta exploration penalties, so not used here
    self._ep['rew_done'].append(-beta)

    # Fixate reward
    rew_fixate, gt_max_ious = reward_fixate(pred_bboxes, gt_boxes, gt_max_ious)
    self._ep['rew_fix'].append(rew_fixate)

    return rew_fixate, gt_max_ious


  # Below are two helper function used, depending on whether agent
  # terminates by done action or by exceeding max-length trajectory
  def ce_fix_terminate_via_max_it(self, fix_logits):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=fix_logits, labels=self._fix_labels, name="ce_fix1")


  # In this case, fix_labels' final entry is wrong and we need to get
  # rid of that part (setting cross-entropy manually to zero)
  def ce_fix_terminate_via_done(self, fix_logits):
    ce_fix = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=fix_logits, labels=self._fix_labels, name="ce_fix2")
    return tf.concat([tf.slice(ce_fix, [0], [tf.shape(ce_fix)[0] - 1]),
                      tf.zeros([1])], 0)


  def reset_after_gradient(self):
    for ix, grad in enumerate(self._grad_buffer):
      self._grad_buffer[ix] = grad * 0
    self._ep_batch = {'x': [], 'h': [], 'aux': [], 'done': [], 'fix': [],
                      'rew_done': [], 'rew_fix': [], 'cond': []}
    self._bl_done = [[] for _ in range(cfg.DRL_RPN.MAX_ITER_TRAJ)]
    self._bl_fix = [[] for _ in range(cfg.DRL_RPN.MAX_ITER_TRAJ)]


  def reset_pre_traj(self):
    # xs = collected observations, hs, is corresponding hidden,
    # ys = collected "fake labels", rews = collected rewards
    self._ep = {'x': [], 'h': [], 'aux': [], 'fix': [], 'rew_done': [],
                'rew_fix': [], 'rew_done_gt': [], 'rew_fix_gt': []}


  def init_rl_train(self, sess):

    # Return RL-trainable variables (thus skip detector parameters here;
    # they are treated separately).
    tvars = tf.trainable_variables()[-17:]
    self._batch_grad = [tf.placeholder(tf.float32,
                                       name='drl_rpn_grad_' + str(idx))\
                                       for idx in range(len(tvars))]

    # Optimizer
    temp = set(tf.global_variables())
    self._lr_rl_in = tf.placeholder(tf.float32)
    adam = tf.train.AdamOptimizer(learning_rate=cfg.DRL_RPN_TRAIN.LEARNING_RATE)
    self._update_grads = adam.apply_gradients(zip(self._batch_grad, tvars))
    sess.run(tf.variables_initializer(set(tf.global_variables()) - temp),
             feed_dict={self._lr_rl_in: cfg.DRL_RPN_TRAIN.LEARNING_RATE})

    # RL loss
    self._done_labels = tf.placeholder(tf.float32, [None, 1], name="done_labels")
    self._fix_labels = tf.placeholder(tf.int32, [None], name="fix_labels")
    self._advs_done = tf.placeholder(tf.float32, [None], name="reward_done")
    self._advs_fix = tf.placeholder(tf.float32, [None], name="reward_fix")
    self._cond_switch_fix = tf.placeholder(tf.int32)
    done_logits = self._predictions['done_logits']
    fix_logits = self._predictions['fix_logits']
    ce_done = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(
                  labels=self._done_labels, logits=done_logits,
                  name="ce_done_logits"))
    ce_fix = tf.cond(tf.equal(self._cond_switch_fix, 0),
                     lambda: self.ce_fix_terminate_via_max_it(fix_logits),
                     lambda: self.ce_fix_terminate_via_done(fix_logits))
    ce_done_rew_prod = ce_done * self._advs_done
    ce_fix_rew_prod = ce_fix * self._advs_fix
    ce_rew_prod = ce_done_rew_prod + ce_fix_rew_prod
    loss_rl = tf.reduce_sum(ce_rew_prod)
    new_grads = tf.gradients(loss_rl, tvars)

    # Add to predictions container
    self._predictions['ce_done'] = ce_done
    self._predictions['ce_fix'] = ce_fix
    self._predictions['ce_done_rew_prod'] = ce_done_rew_prod
    self._predictions['ce_fix_rew_prod'] = ce_fix_rew_prod
    self._predictions['loss_rl'] = loss_rl
    self._predictions['new_grads'] = new_grads

    # Initialize gradient buffer
    self._grad_buffer = sess.run(tvars)

    # Reset the gradient placeholder and other parts
    self.reset_after_gradient()


  def assign_post_hist_weights(self, sess):
    print("Assigning pre-trained fc-weights to post-hist module")

    # Get FROM drl-RPN pretrained detector
    with tf.variable_scope('vgg_16/cls_score', reuse=True):
      cls_score_weights = tf.get_variable('weights')
      cls_score_biases = tf.get_variable('biases')

    # Apply TO post-hist adjustment module
    with tf.variable_scope('post_hist/cls_score_hist', reuse=True):
      cls_score_weights_hist = tf.get_variable('weights')
      cls_score_biases_hist = tf.get_variable('biases')

    # Perform the weight copy
    sess.run(cls_score_weights_hist.assign(cls_score_weights.eval(session=sess)))
    sess.run(cls_score_biases_hist.assign(cls_score_biases.eval(session=sess)))

    # Also assign fc-weights leading up to the respective detector cls and box
    # heads! ('fc7')
    with tf.variable_scope('vgg_16/fc6', reuse=True):
      fc6_weights = tf.get_variable('weights')
      fc6_biases = tf.get_variable('biases')
    with tf.variable_scope('vgg_16/fc7', reuse=True):
      fc7_weights = tf.get_variable('weights')
      fc7_biases = tf.get_variable('biases')
    with tf.variable_scope('post_hist/vgg_16/fc6', reuse=True):
      fc6_weights_hist = tf.get_variable('weights')
      fc6_biases_hist = tf.get_variable('biases')
    with tf.variable_scope('post_hist/vgg_16/fc7', reuse=True):
      fc7_weights_hist = tf.get_variable('weights')
      fc7_biases_hist = tf.get_variable('biases')
    sess.run(fc6_weights_hist.assign(fc6_weights.eval(session=sess)))
    sess.run(fc6_biases_hist.assign(fc6_biases.eval(session=sess)))
    sess.run(fc7_weights_hist.assign(fc7_weights.eval(session=sess)))
    sess.run(fc7_biases_hist.assign(fc7_biases.eval(session=sess)))
    print("Done assigning pre-trained fc-weights to post-hist module\n")


  def _net_conv_from_im(self, is_training=True):
    return self._predictions['net_conv']


  def _net_conv_given(self):
    return self._net_conv_in


  def _net_rois_batched(self):
    return self._rois_seq_batched


  def _net_rois_seq(self):
    return self._rois_seq


  def _build_network(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0,stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # Note: We create both net_conv given from input image (image_to_head)
    # and as a placeholder, since we also want to be able to immediately
    # send in the precomputed feature map at some stages
    net_conv = self._image_to_head(is_training)
    self._predictions['net_conv'] = net_conv
    self._net_conv_in \
      = tf.placeholder(tf.float32, shape=[None, None, None, cfg.DIMS_BASE])
      
    # Below conditional is used as follows: The first time an image is sent
    # through the network, net_conv above is produced. But if we later want
    # to use that conv-map again (e.g. when training detector in drl-RPN
    # training), we don't want to have to send the full image again, and instead
    # directly use the pre-computed feature map.

    # TODO: Note that due to the tensorflow graph construction,
    # both branches will be executed despite using cond -- potentially could
    # gain speedup by having separate graph construction during inferences

    self._cond_switch = tf.placeholder(tf.int32)
    reg_prop_in = tf.cond(tf.equal(self._cond_switch, 0),
                          lambda: self._net_conv_from_im(is_training),
                          lambda: self._net_conv_given())
    reg_prop_in = self._net_conv_from_im(is_training)

    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network (we use the bbox and cls rpn outputs in drl-RPN)
      self._region_proposal(reg_prop_in, False, initializer)

    # Similar story for the below conditional
    self._cond_switch_roi = tf.placeholder(tf.int32)
    self._rois_seq = tf.placeholder(tf.float32, shape=[None, 5])
    self._rois_seq_batched \
      = self._proposal_target_layer_wo_scores(self._rois_seq,'rois_seq_batched')
    rois_in = tf.cond(tf.equal(self._cond_switch_roi, 0),
                      lambda: self._net_rois_batched(),
                      lambda: self._net_rois_seq())

    # Sequential class-specific processing
    pool5_drl_rpn = self._crop_pool_layer(self._net_conv_in, rois_in,
                                          'pool5_drl_rpn')
    fc7_seq = self._head_to_tail(pool5_drl_rpn, is_training)
    with tf.variable_scope(self._scope, self._scope):
      self._region_classification(fc7_seq, is_training, initializer,
                                  initializer_bbox)

    # If desired, also build module corresponding to posterior class
    # probability adjustments
    if cfg.DRL_RPN.USE_POST:
      self._post_hist(is_training,
                      tf.random_normal_initializer(mean=0.0, stddev=0.01))


  def _post_hist(self, is_training=True, initializer=None):

    rois_in = tf.cond(tf.equal(self._cond_switch_roi, 0),
                      lambda: self._net_rois_batched(),
                      lambda: self._net_rois_seq())
    pool5_post_hist = self._crop_pool_layer(self._net_conv_in, rois_in,
                                            'pool5_post_hist')
    with tf.variable_scope('post_hist'):
      fc7_post_hist = self._head_to_tail(pool5_post_hist, is_training, False, 0.75)
      hist_dim = cfg.NBR_CLASSES * cfg.DRL_RPN.H_HIST * cfg.DRL_RPN.W_HIST
      self._cls_hist = tf.placeholder(tf.float32, shape=[None, hist_dim])
      hist_tanh_cls = slim.fully_connected(self._cls_hist, cfg.NBR_CLASSES,
                                           weights_initializer=initializer,
                                           trainable=is_training, reuse=False,
                                           activation_fn=tf.nn.tanh,
                                           scope='hist_tanh_cls')
      self._region_classification_hist(fc7_post_hist, hist_tanh_cls,
                                       is_training)


  def _region_classification_hist(self, fc7, hist_tanh_cls, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    cls_score_hist = slim.fully_connected(fc7, self._num_classes, 
                                          weights_initializer=initializer,
                                          trainable=is_training, reuse=False,
                                          activation_fn=None,
                                          scope='cls_score_hist')
    cls_score_hist += hist_tanh_cls
    cls_prob_hist = self._softmax_layer(cls_score_hist, 'cls_prob_hist')
    cls_pred_hist = tf.argmax(cls_score_hist, axis=1, name='cls_pred_hist')
    self._predictions['cls_score_hist'] = cls_score_hist
    self._predictions['cls_pred_hist'] = cls_pred_hist
    self._predictions['cls_prob_hist'] = cls_prob_hist


  def build_drl_rpn_network(self, is_training=True):

    # Initial processing
    net_conv = self._predictions['net_conv']
    rpn_cls_prob = self._predictions['rpn_cls_prob']
    rpn_bbox_pred = self._predictions['rpn_bbox_pred']
    self._initial_rl_input(net_conv, rpn_cls_prob, rpn_bbox_pred)

    # Convolutional GRU
    self._conv_gru(is_training, tf.contrib.layers.xavier_initializer())


  # Shorter way of writing tf.get_variable(...)
  def _make_var(self, name, shape, initializer=None, is_training=True):    
    return tf.get_variable(name, shape, dtype=None, initializer=initializer,
                           regularizer=None, trainable=is_training)


  # The Conv-GRU processor
  def _conv_gru(self, is_training, initializer, name='conv_gru'):

    # Extract some relevant config keys for convenience
    dims_base = cfg.DIMS_BASE
    dims_aux = cfg.DIMS_AUX
    dims_tot = cfg.DIMS_TOT

    # Input placeholders
    # dims: batch-time-height-width-channel
    self._rl_in = tf.placeholder(tf.float32, shape=[None, None, None, dims_tot])
    self._rl_hid = tf.placeholder(tf.float32, shape=[None, None, None, 300])

    # Define convenience operator
    self.conv = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
      
    # Create conv-GRU kernels
    self.xr_kernel_base = self._make_var('xr_weights_base', [3, 3, dims_base, 240],
                                    initializer, is_training)
    self.xr_kernel_aux = self._make_var('xr_weights_aux', [9, 9, dims_aux, 60],
                                   initializer, is_training)
    self.xh_kernel_base = self._make_var('xh_weights_base', [3, 3, dims_base, 240],
                                    initializer, is_training)
    self.xh_kernel_aux = self._make_var('xh_weights_aux', [9, 9, dims_aux, 60],
                                   initializer, is_training)
    self.xz_kernel_base = self._make_var('xz_weights_base', [3, 3, dims_base, 240],
                                    initializer, is_training)
    self.xz_kernel_aux = self._make_var('xz_weights_aux', [9, 9, dims_aux, 60],
                                   initializer, is_training)
    self.hr_kernel = self._make_var('hr_weights', [3, 3, 300, 300],
                               initializer, is_training)
    self.hh_kernel = self._make_var('hh_weights', [3, 3, 300, 300],
                               initializer, is_training)
    self.hz_kernel = self._make_var('hz_weights', [3, 3, 300, 300],
                               initializer, is_training)
    self.h_relu_kernel = self._make_var('h_relu_weights', [3, 3, 300, 128],
                                   initializer, is_training)

    # Create Conv-GRU biases
    bias_init = initializer
    self.r_bias = self._make_var('r_bias', [300], bias_init, is_training)
    self.h_bias = self._make_var('h_bias', [300], bias_init, is_training)
    self.z_bias = self._make_var('z_bias', [300], bias_init, is_training)
    self.relu_bias = self._make_var('relu_bias', [128], bias_init, is_training) 

    # Used for some aux info (e.g. exploration penalty when used as feature)
    add_dim = 130
    self.additional_kernel = self._make_var('additional_weights', [3, 3, add_dim, 2],
                                            initializer, is_training)
    self.additional_bias = self._make_var('additional_bias', [2], bias_init,
                                          is_training)
    self._aux_done_info = tf.placeholder(tf.float32,
                                         shape=[None, None, None, 2])

    # Define weights for stopping condition (no bias here)
    self.done_weights = self._make_var('done_weights', [625, 1], initializer,
                                       is_training)

    # We need to make a TensorFlow dynamic graph-style while-loop, as our
    # conv-GRU will be unrolled a different number of steps depending on the
    # termination decisions of the agent
    
    # First we need to set some init / dummy variables
    in_shape = tf.shape(self._rl_in)
    done_logits_all = tf.zeros([0, 1])
    fix_logits_all = tf.zeros([0, in_shape[1] * in_shape[2]])
    done_prob = tf.zeros([0, 1])
    fix_prob_map = tf.zeros([0, 0, 0, 0])
    h = tf.slice(self._rl_hid, [0, 0, 0, 0], [1, -1, -1, -1])
    
    # Looping termination condition (TF syntax demands also the other variables
    # are sent as input, although not used for the condition check)
    nbr_steps = in_shape[0]
    i = tf.constant(0)
    while_cond = lambda i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map: tf.less(i, nbr_steps)

    # Unroll current step (if forward pass) and if in training unroll
    # a full rollout
    i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map \
      = tf.while_loop(while_cond, self.rollout, [i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map],
                             shape_invariants=[i.get_shape(), tf.TensorShape([None, 1]),
                                               tf.TensorShape([None, None]), tf.TensorShape([None, 1]),
                                               h.get_shape(), tf.TensorShape([None, None, None, None])])

    # Insert to containers
    self._predictions['done_prob'] = done_prob
    self._predictions['fix_prob'] = fix_prob_map
    self._predictions['done_logits'] = done_logits_all
    self._predictions['fix_logits'] = fix_logits_all
    self._predictions['rl_hid'] = h


  # Unroll the GRU
  def rollout(self, i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map):

    # Extract some relevant config keys for convenience
    dims_base = cfg.DIMS_BASE

    # Split into base feature map and auxiliary input
    rl_base = tf.slice(self._rl_in, [i, 0, 0, 0], [1, -1, -1, dims_base])
    rl_aux = tf.slice(self._rl_in, [i, 0, 0, dims_base], [1, -1, -1, -1])

    # eq. (1)
    xr_conv = tf.concat([self.conv(rl_base, self.xr_kernel_base),
                         self.conv(rl_aux, self.xr_kernel_aux)], 3)
    hr_conv = self.conv(h, self.hr_kernel)
    r = tf.sigmoid(xr_conv + hr_conv + self.r_bias)

    # eq. (2)
    xh_conv = tf.concat([self.conv(rl_base, self.xh_kernel_base),
                         self.conv(rl_aux, self.xh_kernel_aux)], 3)
    hh_conv = self.conv(r * h, self.hh_kernel)
    hbar = tf.tanh(xh_conv + hh_conv + self.h_bias)

    # eq. (3)
    xz_conv = tf.concat([self.conv(rl_base, self.xz_kernel_base),
                         self.conv(rl_aux, self.xz_kernel_aux)], 3)
    hz_conv = self.conv(h, self.hz_kernel)
    z = tf.sigmoid(xz_conv + hz_conv + self.z_bias) 

    # eq. (4)
    h = (1 - z) * h + z * hbar

    # eq. (5)
    conv_gru = tf.nn.relu(self.conv(h, self.h_relu_kernel) + self.relu_bias)

    # HERE BEGINS THE CODE FOR TRANSFORMING INTO ACTION PROBABILITIES
    aux_done_info = tf.slice(self._aux_done_info, [i, 0, 0, 0], [1, -1, -1, -1])

    # Extract relevant stuff
    input_shape = tf.shape(conv_gru)
    batch_sz = 1 # must be 1
    height = input_shape[1]
    width = input_shape[2]

    # Append beta and time-info (auxiliary info)
    conv_gru \
      = tf.concat([conv_gru,
                   tf.ones((batch_sz, height, width, 2)) * aux_done_info], 3)

    # eq. (6)
    conv_gru_processed = tf.nn.tanh(self.conv(conv_gru, self.additional_kernel) \
                                    + self.additional_bias)

    done_slice = tf.slice(conv_gru_processed, [0, 0, 0, 0], [1, -1, -1, 1])
    fix_slice = tf.slice(conv_gru_processed, [0, 0, 0, 1], [1, -1, -1, 1])
    done_slice_reshaped = tf.image.resize_images(done_slice, [25, 25])
    done_slice_vecd = tf.reshape(done_slice_reshaped, [batch_sz, 625])
    done_logits = tf.matmul(done_slice_vecd, self.done_weights) 
    done_prob = tf.sigmoid(done_logits)

    # Probability of where to fix next (need some rearrangement in
    # between to get proper dimensions over which softmax is performed)
    reshape_layer = tf.reshape(tf.transpose(fix_slice, [0, 3, 1, 2]),
                               [1, 1, height * width])
    smax_layer = tf.nn.softmax(reshape_layer)
    fix_prob_map = tf.transpose(tf.reshape(smax_layer,
                              [1, 1, height, width]), [0, 2, 3, 1])
    fix_slice_logits = tf.reshape(fix_slice, [batch_sz, -1]) 

    # Append
    done_logits_all = tf.concat([done_logits_all, done_logits], 0)
    fix_logits_all = tf.concat([fix_logits_all, fix_slice_logits], 0)

    # Return
    return tf.add(i, 1), done_logits_all, fix_logits_all, done_prob, h, fix_prob_map


  # Computes (1 minus [0,1]-normalized) 2-norms of the rpn bbox pred entries
  # of form [x1, y1, x2, y2]. It is hypothesized that it may be useful for the
  # agent to know how large offsets the underlying network deems to be necessary
  # for the correspondingly picked voxels. Intuitively it would seem like boxes
  # requring very little adjustment are often correlated to also being good
  # candidates overall, and if so, the agent can learn to recognize this.
  def _compute_rpn_bbox_norm(self, rpn_bbox_pred, name='rpn_bbox_norms'):

    # Get input shape
    input_shape = tf.shape(rpn_bbox_pred)

    # Format into shape [1 x H x W x A, 4]
    rpn_bbox_format = tf.reshape(rpn_bbox_pred, [-1, 4]) 

    # Compute norm of each box coordinate
    rpn_bbox_norm = tf.sqrt(tf.reduce_sum(tf.square(rpn_bbox_format), 1))

    # Go back to original shape (note: input_shape[3] = 4 * nbr_anchors,
    # which explains the dividing by 4)
    rpn_bbox_norm_orig_shape = tf.reshape(rpn_bbox_norm,
                                          [input_shape[0], input_shape[1],
                                           input_shape[2],
                                           tf.cast(input_shape[3] / 4,tf.int32)])

    # Normalize to range [0,1], and also return "1 - norm" instead of "norm"
    # (because will use -1 to represent unacceptably small voxels which won't
    # be used and so it seems reasonable to have "good candidates" with small
    # offset-norms to be close to 1, which is as far as possible from -1)
    rpn_bbox_norm_out = 1 - rpn_bbox_norm_orig_shape \
                        / tf.reduce_max(rpn_bbox_norm_orig_shape)
    return rpn_bbox_norm_out


  def _initial_rl_input(self, net_conv, rpn_cls_prob, rpn_bbox_pred,
                        name='rl_in_init'):

    # Compute [0,1]-normalized bbox pred norms
    rpn_bbox_norm = self._compute_rpn_bbox_norm(rpn_bbox_pred)

    # Form initial input
    shape_info = tf.shape(rpn_bbox_norm)
    batch_sz = shape_info[0]
    height = shape_info[1]
    width = shape_info[2]
    rpn_cls_objness = tf.slice(rpn_cls_prob, [0, 0, 0, cfg.NBR_ANCHORS],
                               [-1, -1, -1, -1])
    self._predictions['rpn_cls_objness'] = rpn_cls_objness
    cls_probs_rl_input = tf.zeros((batch_sz, height, width, cfg.NBR_CLASSES))
    rl_in_init = tf.concat([net_conv / tf.reduce_max(net_conv), rpn_cls_objness,
                            rpn_bbox_norm, cls_probs_rl_input], 3)

    # Potentially resize for increased speed
    h_scale = cfg.DRL_RPN.H_SCALE
    w_scale = cfg.DRL_RPN.W_SCALE
    new_sz = [tf.cast(tf.round(h_scale * tf.cast(height, tf.float32)), tf.int32),
              tf.cast(tf.round(w_scale * tf.cast(width, tf.float32)), tf.int32)]
    rl_in_init = tf.image.resize_images(rl_in_init, new_sz)
    self._predictions['rl_in_init'] = rl_in_init

    # Also setup all RoIs and RoI observation volume
    self._proposal_layer_all(rpn_bbox_pred, rpn_cls_prob)


  def get_init_rl(self, sess, image, im_info):
    feed_dict = {self._image: image, self._im_info: im_info, self._cond_switch:0,
                 self._net_conv_in: np.zeros((1, 1, 1, cfg.DIMS_BASE))}
    net_conv, rl_in, rois_all, roi_obs_vol, rpn_cls_objness, not_keep_ids \
      = sess.run([self._predictions['net_conv'], self._predictions['rl_in_init'],
                  self._predictions['rois_all'],self._predictions['roi_obs_vol'],
                  self._predictions['rpn_cls_objness'],
                  self._predictions['not_keep_ids']], feed_dict=feed_dict)
    if not cfg.DRL_RPN.USE_HIST:
      rl_in = rl_in[:, :, :, :cfg.DIMS_NONHIST]

    # Create drl-RPN hidden state (conv-GRU hidden state)
    batch_sz, height, width = rl_in.shape[:3]
    rl_hid = np.zeros((batch_sz, height, width, 300))

    # Potentially we will want to use top-K within third axis (anchor dim)
    # when selecting RoIs at observation rectangles (using fewer per channel
    # increases speed)
    if cfg.DRL_RPN.TOPK_OBJNESS > 0:
      rpn_cls_topK_objness_vals \
        = -np.sort(-rpn_cls_objness, axis=3)[:,:,:, cfg.DRL_RPN.TOPK_OBJNESS-1]
      rpn_cls_topK_objness = np.zeros(rpn_cls_objness.shape, dtype=np.bool)
      rpn_cls_topK_objness[\
        rpn_cls_objness >= rpn_cls_topK_objness_vals[:, :, :, np.newaxis]] = 1
    else:
      rpn_cls_topK_objness = None

    # Get height, width of downsized feature map and orig. feature map, and also
    # calculate the fixation rectangle size used etcetera
    height, width = rl_in.shape[1:3]
    height_orig, width_orig = roi_obs_vol.shape[1:3]
    fix_rect_h = int(round(cfg.DRL_RPN.H_FIXRECT * height))
    fix_rect_w = int(round(cfg.DRL_RPN.W_FIXRECT * width))
    h_ratio_orig = float(height_orig) / height
    w_ratio_orig = float(width_orig) / width
    fix_rect_h_orig = int(round(fix_rect_h * h_ratio_orig))
    fix_rect_w_orig = int(round(fix_rect_w * w_ratio_orig))

    # Return
    return net_conv, rl_in, rl_hid, rois_all, roi_obs_vol, \
            rpn_cls_topK_objness, rpn_cls_objness.reshape(-1), \
            height, width, height_orig, width_orig, \
            fix_rect_h, fix_rect_w, h_ratio_orig, w_ratio_orig, \
            fix_rect_h_orig, fix_rect_w_orig, not_keep_ids


  def action_pass(self, sess, rl_in, rl_hid, t, beta, is_training=True):
    """ This is the "forward pass" of the drl-RPN action selection """

    # Mean-normalize incoming beta
    # NOTE: We need to make beta 10 times larger for the learning to
    # "take effect")
    if len(cfg.DRL_RPN_TRAIN.BETAS) > 1:
      beta /= (0.05 * max(cfg.DRL_RPN_TRAIN.BETAS))
    else:
      beta /= max(cfg.DRL_RPN_TRAIN.BETAS)

    aux_done = np.empty((1, 1, 1, 2))
    aux_done[0, 0, 0, 0] = t / (cfg.DRL_RPN.MAX_ITER_TRAJ_FLT - 1)
    aux_done[0, 0, 0, 1] = beta

    if is_training:
      # Store in containers (used backpropagating gradients)

      # must copy rl_in -- otherwise rl-udpates in future affect past states
      # due to 'pass-by-reference' in python numpy-array-updates
      self._ep['x'].append(np.copy(rl_in))
      self._ep['h'].append(rl_hid)
      self._ep['aux'].append(aux_done)
    feed_dict_action = {self._rl_in: rl_in, self._rl_hid: rl_hid,
                        self._aux_done_info: aux_done}
    rl_hid, done_prob, fix_prob \
      = sess.run([self._predictions['rl_hid'],
                  self._predictions['done_prob'],
                  self._predictions['fix_prob']],
                 feed_dict=feed_dict_action)
    fix_prob = fix_prob[0, :, :, 0]
    return rl_hid, done_prob, fix_prob


  def seq_rois_pass(self, sess, net_conv, rois_seq, is_train_det=False):
    """
    This function handles the per-fixation sequential forwarding of RoIs
    for class-specific predictions
    """
    feed_dict_seq = {self._net_conv_in: net_conv, self._rois_seq: rois_seq,
                     self._cond_switch_roi: 1, self._gt_boxes: np.zeros((1, 5))}
    cls_prob_seq, bbox_preds_seq = sess.run([self._predictions['cls_prob_seq'],
                                             self._predictions['bbox_pred_seq']],
                                            feed_dict=feed_dict_seq)
    # If test-time (or any time, e.g. drl-RPN training, where we are NOT
    # specifically training the detector component), need to "undo"
    # mean-std normalization
    if not is_train_det:
      bbox_preds_seq *= cfg.STDS_BBOX
      bbox_preds_seq += cfg.MEANS_BBOX
    return cls_prob_seq, bbox_preds_seq


  # Only used at test time
  def post_hist_nudge(self, sess, net_conv, rois_seq, cls_hist):
    """
    This function performs posterior class-specifc cls- and bbox-adjustments
    based on detections in trajectory 
    """
    feed_dict_seq = {self._net_conv_in: net_conv, self._rois_seq: rois_seq,
                     self._cls_hist: cls_hist, self._cond_switch_roi: 1,
                     self._gt_boxes: np.zeros((1, 5))}
    cls_prob_hist = sess.run(self._predictions['cls_prob_hist'],
                             feed_dict=feed_dict_seq)
    return cls_prob_hist


  ############# DRL-RPN ADDITIONAL COMPONENTS -- END ##########################


  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError


  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError


  def create_architecture(self, mode, tag=None, anchor_scales=(8, 16, 32),
                          anchor_ratios=(0.5, 1, 2)):
    assert tag != None

    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag

    self._num_classes = cfg.NBR_CLASSES
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    # handle most of the regularizers here
    weights_regularizer \
      = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                    slim.conv2d_transpose, slim.separable_conv2d,
                    slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      self._build_network(training)

    layers_to_output = {}
    if training:
      self._add_losses()
      if cfg.DRL_RPN.USE_POST:
        self._add_losses(True)
      layers_to_output.update(self._losses)
    layers_to_output.update(self._predictions)
    return layers_to_output


  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError


  def fix_variables(self, sess, pretrained_model, do_reverse):
    raise NotImplementedError


  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat


  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image, self._im_info: im_info}
    cls_score, cls_prob, bbox_pred, rois \
      = sess.run([self._predictions["cls_score_seq"],
                  self._predictions['cls_prob_seq'],
                  self._predictions['bbox_pred_seq'], self._predictions['rois']],
                 feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois


  def train_step_det(self, sess, train_op, net_conv, rois_seq, gt_boxes,
                     im_info):
    feed_dict = {self._net_conv_in: net_conv, self._rois_seq: rois_seq,
                 self._im_info: im_info, self._gt_boxes: gt_boxes,
                 self._cond_switch: 1,
                 self._image: np.zeros((1, 1, 1, 3)), self._cond_switch_roi: 0}
    loss_cls, loss_box, loss, _ \
      = sess.run([self._losses['cross_entropy'], self._losses['loss_box'],
                  self._losses['total_loss'], train_op], feed_dict=feed_dict)
    return loss_cls, loss_box, loss


  def train_step_post(self, sess, train_op, net_conv, rois_seq, gt_boxes,
                      im_info, cls_hist):
    feed_dict = {self._net_conv_in: net_conv, self._rois_seq: rois_seq,
                 self._im_info: im_info, self._gt_boxes: gt_boxes,
                 self._cls_hist: cls_hist, self._cond_switch: 1,
                 self._image: np.zeros((1, 1, 1, 3)), self._cond_switch_roi: 0}
    loss, _ = sess.run([self._losses['total_loss_hist'], train_op],
                       feed_dict=feed_dict)
    return loss


  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)
