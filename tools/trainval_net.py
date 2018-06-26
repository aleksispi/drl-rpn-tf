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

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
from time import sleep

import tensorflow as tf
from nets.vgg16 import vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a drl-RPN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--save', dest='save_path',
                      help='path for saving model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16',
                      default='vgg16', type=str)
  parser.add_argument('--use_hist', dest='use_hist',
                      help='1 = use class-specific history, 0 = do not',
                      default=1, type=int)
  parser.add_argument('--det_start', dest='det_start',
                      help='-1: dont train detector; >=0: train detector onwards',
                      default=-1, type=int)
  parser.add_argument('--use_post', dest='use_post',
                      help='-1 = dont train post-hist nudge, >=0: train onwards',
                      default=-1, type=int)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


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
  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(args.imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  # Set class names in config file based on IMDB
  class_names = imdb.classes
  cfg_from_list(['CLASS_NAMES', [class_names]])

  # Update config depending on if class-specific history used or not
  if not args.use_hist:
    cfg_from_list(['DRL_RPN.USE_HIST', False])
    cfg_from_list(['DIMS_TOT', cfg.DIMS_NONHIST])
    cfg_from_list(['DIMS_AUX', 2 * cfg.NBR_ANCHORS])
  elif args.use_post >= 0:
    cfg_from_list(['DRL_RPN.USE_POST', True])
    cfg_from_list(['DRL_RPN_TRAIN.USE_POST', args.use_post])

  # Update config to match start of training detector
  cfg_from_list(['DRL_RPN_TRAIN.DET_START', args.det_start])

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag, args.save_path)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = vgg16()
  else:
    raise NotImplementedError

  train_net(net, imdb, roidb, valroidb, output_dir, pretrained_model=args.weight,
            max_iters=args.max_iters)