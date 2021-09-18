import ml_collections
import torch
import os

def get_default_configs_wolf():
  config = ml_collections.ConfigDict()

  # flow
  config.wolf = wolf = ml_collections.ConfigDict()
  wolf.model_config = 'configs/cifar10/glow-gaussian-uni.json'
  wolf.rank = 1
  wolf.local_rank = 0
  wolf.batch_size = 512
  wolf.eval_batch_size = 4
  wolf.batch_steps = 1
  wolf.init_batch_size = 1024
  wolf.epochs = 500
  wolf.valid_epochs = 1
  wolf.seed = 65537
  wolf.train_k = 1
  wolf.log_interval = 10
  wolf.lr = 0.001
  wolf.warmup_steps = 500
  wolf.lr_decay = 0.999997
  wolf.beta1 = 0.9
  wolf.beta2 = 0.999
  wolf.eps = 1e-8
  wolf.weight_decay = 0
  wolf.amsgrad = True
  wolf.grad_clip = 0
  wolf.dataset = 'cifar10'
  wolf.category = None
  wolf.image_size = 32
  wolf.workers = 4
  wolf.n_bits = 8
  wolf.recover = -1

  return config