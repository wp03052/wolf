import ml_collections
import torch
import os

def get_default_configs_flow():
  config = ml_collections.ConfigDict()

  # flow
  config.flow = flow = ml_collections.ConfigDict()
  flow.model = 'resflow'
  #flow.model = 'identity'
  flow.nblocks = '8'
  flow.ema_rate = 0.999
  flow.lr = 5e-5
  flow.intermediate_dim = 512
  flow.resblock_type = 'resflow' # ['biggan', 'fc', 'resflow']
  flow.beta = 1.
  flow.logit_transform = False

  return config