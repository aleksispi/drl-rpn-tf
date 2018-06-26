import numpy as np

from model.config import cfg


class StatCollector(object):

  def __init__(self, nbr_ep, stat_strings, is_training=True):
    
    # Set mode
    self.is_training = is_training

    # What to display
    self.stat_strings = stat_strings
    self.nbr_stats = len(stat_strings)
    max_string_len = 0
    for i in range(self.nbr_stats):
      max_string_len = max(max_string_len, len(self.stat_strings[i]))
    self.spaces = []#cell(self.nbr_stats, 1)
    for i in range(self.nbr_stats):#= 1 : self.nbr_stats
      self.spaces.append((max_string_len - len(self.stat_strings[i])) * ' ')
    
    # Initialize total averages
    self.mean_loss = 0.0
    self.means = np.zeros(self.nbr_stats, dtype=np.float32)
    
    # Initialize exponential moving averages
    self.ma_loss = 0.0
    self.mas = np.zeros(self.nbr_stats, dtype=np.float32)
    
    self.ma_weight = cfg.DRL_RPN_TRAIN.MA_WEIGHT
    self.nbr_ep = nbr_ep
    self.ep = 0
    self.bz = cfg.DRL_RPN_TRAIN.BATCH_SIZE
    
    # Initialize arrays of total averages
    self.mean_losses = np.zeros(nbr_ep / self.bz, dtype=np.float32)
    self.means_all = np.zeros((self.nbr_stats, nbr_ep), dtype=np.float32)
    
    # Initialize arrays of moving averages
    self.ma_losses = np.zeros(nbr_ep / self.bz, dtype=np.float32)
    self.mas_all = np.zeros((self.nbr_stats, nbr_ep), dtype=np.float32)

    # Initialize custom stuff (in this case, avg. traj length vs. #gt-instances,
    # and avg. traj length vs. beta)
    self.means_traj_vs_gts = np.zeros(10, dtype=np.float32)
    self.mas_traj_vs_gts = np.zeros(10, dtype=np.float32)
    self.ep_gts = [0 for _ in range(self.means_traj_vs_gts.shape[0])]
    self.means_traj_vs_betas = {beta: 0.0 for beta in cfg.DRL_RPN_TRAIN.BETAS}
    try:
      self.ep_betas = {beta: 0 for beta in cfg.DRL_RPN_TRAIN.BETAS}
      self.mas_traj_vs_betas = {beta: 0.0 for beta in cfg.DRL_RPN_TRAIN.BETAS}
    except:
      pass


  def update(self, loss, other):
    # Updates averages
    self.update_loss(loss) 
    self.update_means_mas(other)
    self.ep += 1
    

  def update_loss(self, loss):
    # Tracks the loss
    if (self.ep + 1) % self.bz != 0:
        return
    batch_idx = (self.ep + 1) / self.bz - 1
    self.mean_loss = (batch_idx * self.mean_loss + loss) / (batch_idx + 1) 
    self.mean_losses[batch_idx] = self.mean_loss 
    self.ma_loss = (1 - self.ma_weight) * self.ma_loss + self.ma_weight * loss 
    self.ma_losses[batch_idx] = self.ma_loss 


  def update_means_mas(self, data):
    # Tracks various statistics
    for i in range(len(data)):
      if not isinstance(data[i], list):
        self.means[i] = (self.ep * self.means[i] + data[i]) / (self.ep + 1) 
        self.means_all[i, self.ep] = self.means[i] 
        self.mas[i] = (1 - self.ma_weight) * self.mas[i] + self.ma_weight*data[i] 
        self.mas_all[i, self.ep] = self.mas[i] 
      else:
        nbr_gts = min(data[i][0], 9)
        beta = data[i][1]
        traj_len = data[i][2]
        self.means_traj_vs_gts[nbr_gts] \
          = (self.ep_gts[nbr_gts] * self.means_traj_vs_gts[nbr_gts] + traj_len) \
              / (self.ep_gts[nbr_gts] + 1)
        self.ep_gts[nbr_gts] += 1 
        self.mas_traj_vs_gts[nbr_gts] \
          = (1 - 50*self.ma_weight) * self.mas_traj_vs_gts[nbr_gts] \
              + 50*self.ma_weight * traj_len
        try:
          self.means_traj_vs_betas[beta] \
            = (self.ep_betas[beta] * self.means_traj_vs_betas[beta] + traj_len) \
                / (self.ep_betas[beta] + 1)
          self.ep_betas[beta] += 1 
          self.mas_traj_vs_betas[beta] \
            = (1 - 50*self.ma_weight) * self.mas_traj_vs_betas[beta] \
                + 50*self.ma_weight * traj_len
        except:
          pass

  def print_stats(self):
    if self.is_training:
      print('Mean loss (tot, MA):      (%f, %f)' % (self.mean_loss, self.ma_loss))
    for i in range(self.nbr_stats):
      print('Mean %s (tot, MA): %s(%f, %f)' \
        % (self.stat_strings[i], self.spaces[i], self.means[i], self.mas[i]))
    print('Traj-len vs. # gt-instances:')
    print(['%.2f' % g for g in self.means_traj_vs_gts])
    print(['%.2f' % g for g in self.mas_traj_vs_gts])
    try:
      if len(self.means_traj_vs_betas) > 1:
        print('Traj-len vs. betas:')
        print(['%.3f: %.2f' % (b, self.means_traj_vs_betas[b]) \
                for b in self.means_traj_vs_betas])
        print(['%.3f: %.2f' % (b, self.mas_traj_vs_betas[b]) \
                for b in self.mas_traj_vs_betas])
    except:
      pass