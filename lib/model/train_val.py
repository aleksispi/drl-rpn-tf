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

try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time
from time import sleep

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from model.config import cfg, cfg_from_list
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
from utils.statcoll import StatCollector
from model.factory import run_drl_rpn


class SolverWrapper(object):
  """ A wrapper class for the training process """

  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir,
               pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.pretrained_model = pretrained_model


  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename


  def from_snapshot(self, sess, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.saver.restore(sess, sfile)
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter


  def get_variables_in_checkpoint_file(self, file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")


  def construct_graph(self, sess):
    # Set the random seed for tensorflow
    tf.set_random_seed(cfg.RNG_SEED)
    with sess.graph.as_default():
      # Build the main computation graph
      layers = self.net.create_architecture('TRAIN', tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      # Define the loss
      loss = layers['total_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

      # Compute the gradients with regard to the loss
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)

      # Initialize post-hist module of drl-RPN
      if cfg.DRL_RPN.USE_POST:
        loss_post = layers['total_loss_hist']
        lr_post = tf.Variable(cfg.DRL_RPN_TRAIN.POST_LR, trainable=False)
        self.optimizer_post = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM) 
        gvs_post = self.optimizer_post.compute_gradients(loss_post)
        train_op_post = self.optimizer_post.apply_gradients(gvs_post)
      else:
        lr_post = None
        train_op_post = None

      # Initialize main drl-RPN network
      self.net.build_drl_rpn_network()

    return lr, train_op, lr_post, train_op_post


  def initialize(self, sess):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    variables = tf.global_variables()
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = self.net.get_variables_to_restore(variables,
                                                             var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are 
    # hanged to BGR For VGG16 it also changes the convolutional weights fc6
    # and fc7 to fully connected weights
    #
    # OBS: IF YOU WANT TO TRAIN FROM EXISTING FASTER
    # R-CNN WEIGHTS, AND NOT FROM IMAGENET WEIGHTS, SET BELOW FLAG TO FALSE!!!!
    self.net.fix_variables(sess, self.pretrained_model, False)
    print('Fixed.')
    last_snapshot_iter = 0
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)
    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths


  def restore(self, sess, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
    # Set the learning rate
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        rate *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)
    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths


  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)
    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      if os.path.exists(str(sfile)):
        os.remove(str(sfile))
      else:
        os.remove(str(sfile + '.data-00000-of-00001'))
        os.remove(str(sfile + '.index'))
      sfile_meta = sfile + '.meta'
      os.remove(str(sfile_meta))
      ss_paths.remove(sfile)


  def _print_det_loss(self, iter, max_iters, tot_loss, loss_cls, loss_box,
                      lr, timer, in_string='detector'):
    if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
      if loss_box is not None:
        print('iter: %d / %d, total loss: %.6f\n '
              '>>> loss_cls (%s): %.6f\n '
              '>>> loss_box (%s): %.6f\n >>> lr: %f' % \
              (iter + 1, max_iters, tot_loss, in_string, loss_cls, in_string,
               loss_box, lr))
      else:
        print('iter: %d / %d, total loss (%s): %.6f\n >>> lr: %f' % \
              (iter + 1, max_iters, in_string, tot_loss, lr))
      print('speed: {:.3f}s / iter'.format(timer.average_time))


  def _check_if_continue(self, iter, max_iters, snapshot_add):
    img_start_idx = cfg.DRL_RPN_TRAIN.IMG_START_IDX
    if iter > img_start_idx:
      return iter, max_iters, snapshot_add, False
    if iter < img_start_idx:
      print("iter %d < img_start_idx %d -- continuing" % (iter, img_start_idx))
      iter += 1
      return iter, max_iters, snapshot_add, True
    if iter == img_start_idx:
      print("Adjusting stepsize, train-det-start etcetera")
      snapshot_add = img_start_idx
      max_iters -= img_start_idx
      iter = 0
      cfg_from_list(['DRL_RPN_TRAIN.IMG_START_IDX', -1])
      cfg_from_list(['DRL_RPN_TRAIN.DET_START',
                     cfg.DRL_RPN_TRAIN.DET_START - img_start_idx])
      cfg_from_list(['DRL_RPN_TRAIN.STEPSIZE',
                     cfg.DRL_RPN_TRAIN.STEPSIZE - img_start_idx])
      cfg_from_list(['TRAIN.STEPSIZE', [cfg.TRAIN.STEPSIZE[0] - img_start_idx]])
      cfg_from_list(['DRL_RPN_TRAIN.POST_SS',
                    [cfg.DRL_RPN_TRAIN.POST_SS[0] - img_start_idx]])
      print("Done adjusting stepsize, train-det-start etcetera")
      return iter, max_iters, snapshot_add, False


  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, cfg.NBR_CLASSES)
    self.data_layer_val = RoIDataLayer(self.valroidb, cfg.NBR_CLASSES, True)

    # Construct the computation graph corresponding to the original Faster R-CNN
    # architecture first (and potentially the post-hist module of drl-RPN)
    lr_det_op, train_op, lr_post_op, train_op_post \
      = self.construct_graph(sess)

    # Initialize the variables or restore them from the last snapshot
    rate, last_snapshot_iter, stepsizes, np_paths, ss_paths \
      = self.initialize(sess)

    # We will handle the snapshots ourselves
    self.saver = tf.train.Saver(max_to_keep=100000)

    # Initialize
    self.net.init_rl_train(sess)

    # Setup initial learning rates
    lr_rl = cfg.DRL_RPN_TRAIN.LEARNING_RATE
    lr_det = cfg.TRAIN.LEARNING_RATE
    sess.run(tf.assign(lr_det_op, lr_det))
    if cfg.DRL_RPN.USE_POST:
      lr_post = cfg.DRL_RPN_TRAIN.POST_LR
      sess.run(tf.assign(lr_post_op, lr_post))

    # Sample first beta
    if cfg.DRL_RPN_TRAIN.USE_POST:
      betas = cfg.DRL_RPN_TRAIN.POST_BETAS
    else:
      betas = cfg.DRL_RPN_TRAIN.BETAS
    beta_idx = 0
    beta = betas[beta_idx]

    # Setup drl-RPN timers
    timers = {'init': Timer(), 'fulltraj': Timer(), 'upd-obs-vol': Timer(),
                  'upd-seq': Timer(), 'upd-rl': Timer(), 'action-rl': Timer(),
                  'coll-traj': Timer(), 'run-drl-rpn': Timer(),
                  'train-drl-rpn': Timer()}

    # Create StatCollector (tracks various RL training statistics)
    stat_strings = ['reward', 'rew-done', 'traj-len', 'frac-area',
                    'gt >= 0.5 frac', 'gt-IoU-frac']
    sc = StatCollector(max_iters, stat_strings)

    timer = Timer()
    iter = 0
    snapshot_add = 0
    while iter < max_iters:

      # Get training data, one batch at a time (assumes batch size 1)
      blobs = self.data_layer.forward()

      # Allows the possibility to start at arbitrary image, rather
      # than always starting from first image in dataset. Useful if
      # want to load parameters and keep going from there, rather
      # than having those and encountering visited images again.
      iter, max_iters, snapshot_add, do_continue \
        = self._check_if_continue(iter, max_iters, snapshot_add)
      if do_continue:
        continue

      if not cfg.DRL_RPN_TRAIN.USE_POST:

        # Potentially update drl-RPN learning rate
        if (iter + 1) % cfg.DRL_RPN_TRAIN.STEPSIZE == 0:
          lr_rl *= cfg.DRL_RPN_TRAIN.GAMMA

        # Run drl-RPN in training mode
        timers['run-drl-rpn'].tic()
        stats = run_drl_rpn(sess, self.net, blobs, timers, mode='train',
                            beta=beta, im_idx=None, extra_args=lr_rl)
        timers['run-drl-rpn'].toc()

        if (iter + 1) % cfg.DRL_RPN_TRAIN.BATCH_SIZE == 0:
          print("\n##### DRL-RPN BATCH GRADIENT UPDATE - START ##### \n")
          print('iter: %d / %d' % (iter + 1, max_iters))
          print('lr-rl: %f' % lr_rl)
          timers['train-drl-rpn'].tic()
          self.net.train_drl_rpn(sess, lr_rl, sc, stats)
          timers['train-drl-rpn'].toc()
          sc.print_stats()
          print('TIMINGS:')
          print('runnn-drl-rpn: %.4f' % timers['run-drl-rpn'].get_avg())
          print('train-drl-rpn: %.4f' % timers['train-drl-rpn'].get_avg())
          print("\n##### DRL-RPN BATCH GRADIENT UPDATE - DONE ###### \n")

          # Also sample new beta for next batch
          beta_idx += 1
          beta_idx %= len(betas)
          beta = betas[beta_idx]
        else:
	        sc.update(0, stats)

        # At this point we assume that an RL-trajectory has been performed.
        # We next train detector with drl-RPN running in deterministic mode.
        # Potentially train detector component of network
        if cfg.DRL_RPN_TRAIN.DET_START >= 0 and \
          iter >= cfg.DRL_RPN_TRAIN.DET_START:

          # Run drl-RPN in deterministic mode
          net_conv, rois_drl_rpn, gt_boxes, im_info, timers, _ \
            = run_drl_rpn(sess, self.net, blobs, timers, mode='train_det',
                          beta=beta, im_idx=None)

          # Learning rate
          if (iter + 1) % cfg.TRAIN.STEPSIZE[0] == 0:
            lr_det *= cfg.TRAIN.GAMMA
            sess.run(tf.assign(lr_det_op, lr_det))

          timer.tic()
          # Train detector part
          loss_cls, loss_box, tot_loss \
            = self.net.train_step_det(sess, train_op, net_conv, rois_drl_rpn,
                                      gt_boxes, im_info)
          timer.toc()

          # Display training information
          self._print_det_loss(iter, max_iters, tot_loss, loss_cls, loss_box,
                               lr_det, timer)
   
      # Train post-hist module AFTER we have trained rest of drl-RPN! Specifically
      # once rest of drl-RPN has been trained already, copy those weights into
      # the folder of pretrained weights and rerun training with those as initial
      # weights, which will then train only the posterior-history module
      else:

        # The very first time we need to assign the ordinary detector weights
        # as starting point
        if iter == 0:
          self.net.assign_post_hist_weights(sess)

        # Sample beta
        beta = betas[beta_idx]
        beta_idx += 1
        beta_idx %= len(betas)

        # Run drl-RPN in deterministic mode
        net_conv, rois_drl_rpn, gt_boxes, im_info, timers, cls_hist \
          = run_drl_rpn(sess, self.net, blobs, timers, mode='train_det',
                        beta=beta, im_idx=None)

        # Learning rate (assume only one learning rate iter for now!)
        if (iter + 1) % cfg.DRL_RPN_TRAIN.POST_SS[0] == 0:
          lr_post *= cfg.TRAIN.GAMMA
          sess.run(tf.assign(lr_post_op, lr_post))

        # Train post-hist detector part
        tot_loss = self.net.train_step_post(sess, train_op_post, net_conv,
                                            rois_drl_rpn, gt_boxes, im_info,
                                            cls_hist)

        # Display training information
        self._print_det_loss(iter, max_iters, tot_loss, None, None,
                             lr_post, timer, 'post-hist')

      # Snapshotting
      if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter + 1
        ss_path, np_path = self.snapshot(sess, iter + 1 + snapshot_add)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      # Increase iteration counter
      iter += 1

    # Potentially save one last time
    if last_snapshot_iter != iter:
      self.snapshot(sess, iter + snapshot_add)


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')
  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')
  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir,
              pretrained_model=None, max_iters=40000):
  """Train drl-RPN for a Faster R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir,
                       pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters)
    print('done solving')
