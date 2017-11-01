import tensorflow as tf
import random
import numpy as np


def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    varprefix = varprefix.split(':')[0]
    if varprefix in prefix:
      ret_list.append(var)
  return ret_list

def linear(inputs, output_size, no_bias=False, scope=None):
  """Define a linear connection."""

  with tf.variable_scope(scope or 'Linear'):
    input_size = inputs.get_shape()[1].value
    matrix = tf.get_variable('Matrix', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
    bias_term = tf.get_variable('Bias', [output_size], initializer=tf.constant_initializer(0))
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      output = output + bias_term
  return output

def mlp(inputs,  mlp_hidden=[],  mlp_nonlinearity=tf.nn.tanh, scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res