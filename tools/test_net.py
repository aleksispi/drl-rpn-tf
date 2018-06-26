# --------------------------------------------------------
# Tensorflow drl-RPN
# Licensed under The MIT License [see LICENSE for details]
# Partially written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
from time import sleep

import tensorflow as tf
from nets.vgg16 import vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a drl-RPN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
                      help='model to test',
                      default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to test',
                      default='voc_2007_test', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                      action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
                      help='max number of detections per image',
                      default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16',
                      default='vgg16', type=str)
  parser.add_argument('--use_hist', dest='use_hist',
                      help='1 = use class-specific history, 0 = do not',
                      default=1, type=int)
  parser.add_argument('--use_post', dest='use_post',
                      help='1 = use post-hist class-prob adjustments, 0 = do not',
                      default=0, type=int)
  parser.add_argument('--nbr_fix', dest='nbr_fix',
                      help='0: auto-stop, > 0 run drl-RPN exactly nbr_fix steps',
                      default=0, type=int)
  parser.add_argument('--set_idx', dest='set_idx',
                      help='1-4 = frege machine, 5-6 = desktop machine',
                      default=5, type=int)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the initialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  # This extra_string used by me (Aleksis) when running code on two
  # different machines, for convenience
  extra_string = ''
  if args.imdb_name == 'voc_2012_test':
    extra_string += '_test'
  if args.set_idx in [1, 2, 3, 4]:
    extra_string += '_frege'
  imdb = get_imdb(args.imdb_name + extra_string)
  imdb.competition_mode(args.comp_mode)

  # Set class names in config file based on IMDB
  class_names = imdb.classes
  cfg_from_list(['CLASS_NAMES', [class_names]])

  # Update config depending on if class-specific history used or not
  if not args.use_hist:
    cfg_from_list(['DRL_RPN.USE_HIST', False])
    cfg_from_list(['DIMS_TOT', cfg.DIMS_NONHIST])
    cfg_from_list(['DIMS_AUX', 2 * cfg.NBR_ANCHORS])
  elif args.use_post > 0:
    cfg_from_list(['DRL_RPN.USE_POST', True])

  # Specify if run drl-RPN in auto mode or a fix number of iterations
  cfg_from_list(['DRL_RPN_TEST.NBR_FIX', args.nbr_fix])

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # Set the random seed for tensorflow
  tf.set_random_seed(cfg.RNG_SEED)

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16()
  else:
    raise NotImplementedError

  # load model
  net.create_architecture("TEST", tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)
  net.build_drl_rpn_network(False)

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    tf.train.Saver().restore(sess, args.model)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)
  sess.close()