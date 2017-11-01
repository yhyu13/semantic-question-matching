#-*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=64, help='batch size')
data_arg.add_argument('--padlen', type=int, default=20, help='input sequence length')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_size', type=int, default=50, help='hidden dimension')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--training_epochs', type=int, default=10, help='nb epoch')
train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')

train_arg.add_argument('--reformulate_proba', type=float, default=0.5, help='probability of reformulating. if 0, model = simple autoencoder')
train_arg.add_argument('--n_sample', type=int, default=1, help='number of samples per instance')

# Misc
misc_arg = add_argument_group('User options') #####################################################

train_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode when model is trained')
misc_arg.add_argument('--restore_model', type=str2bool, default=False, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='autoencoder', help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='autoencoder', help='loader sub directory')
misc_arg.add_argument('--log_dir', type=str, default='autoencoder', help='summary writer log directory') 



def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


def print_config():
  config, _ = get_config()
  print('\n')
  print('Data Config:')
  print('* Batch size:',config.batch_size)
  print('* Pad length:',config.padlen)
  print('\n')
  print('Network Config:')
  print('* Restored model:',config.restore_model)
  print('* Hidden dimension:',config.hidden_size)
  print('\n')
  if config.is_training==True:
    print('Training Config:')
    print('* Nb epoch:',config.training_epochs)
    print('* Learning rate (init,decay_step,decay_rate):',config.lr_start,config.lr_decay_step,config.lr_decay_rate)
    print('* Saving model to:',' save/'+config.save_to)
  else:
    print('Testing Config:')
  print('* Summary writer log dir:',' summary/'+config.log_dir)
  print('\n')