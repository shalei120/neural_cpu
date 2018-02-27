import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

def linear(args, output_size, bias, bias_start=0.0, stddev=1.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = []
    for a in args:
        try:
            shapes.append(a.get_shape().as_list())
        except Exception as e:
            shapes.append(a.shape)

    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
            total_arg_size += shape[0]
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size], tf.float32, tf.random_uniform_initializer(-stddev, stddev))
        tf.add_to_collection('l2', tf.nn.l2_loss(matrix)) # Add L2 Loss
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable("Bias", [output_size], initializer=init_ops.constant_initializer(bias_start))
        tf.add_to_collection('l2', tf.nn.l2_loss(bias_term)) # Add L2 Loss
    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term

def Linear(input_, output_size, bias_start=0.0,stddev=1.0, is_range=False, squeeze=False,
           name=None, reuse=None):
    """Applies a linear transformation to the incoming data.

    Args:
        input: a 2-D or 1-D data (`Tensor` or `ndarray`)
        output_size: the size of output matrix or vector
    """
    with tf.variable_scope("Linear", reuse=reuse):
        if type(input_) == np.ndarray:
            shape = input_.shape
        else:
            shape = input_.get_shape().as_list()

        is_vector = False
        if len(shape) == 1:
            is_vector = True
            input_ = tf.reshape(input_, [1, -1])
            input_size = shape[0]
        elif len(shape) == 2:
            input_size = shape[1]
        else:
            raise ValueError("Linear expects shape[1] of inputuments: %s" % str(shape))

        w_name = "%s_w" % name if name else None
        b_name = "%s_b" % name if name else None

        w = tf.get_variable(w_name, [input_size, output_size], tf.float32,
                            tf.random_uniform_initializer(-stddev, stddev))
        
        mul = tf.matmul(input_, w)

        if is_range:
            def identity_initializer(tensor):
                def _initializer(shape, dtype=tf.float32, partition_info=None):
                    return tf.identity(tensor)
                return _initializer

            range_ = tf.reverse(tf.range(1, output_size+1, 1), [True])
            b = tf.get_variable(b_name, [output_size], tf.float32,
                                identity_initializer(tf.cast(range_, tf.float32)))
        else:
            b = tf.get_variable(b_name, [output_size], tf.float32, 
                                init_ops.constant_initializer(bias_start))

        tf.add_to_collection('l2', tf.nn.l2_loss(w))
        tf.add_to_collection('l2', tf.nn.l2_loss(b))
        if squeeze:
            output = tf.squeeze(tf.nn.bias_add(mul, b))
        else:
            output = tf.nn.bias_add(mul, b)

        if is_vector:
            return tf.reshape(output, [-1])
        else:
            return output

def binary_cross_entropy_with_logits(logits, targets, name=''):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.name_scope("bce_loss" + name) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return -math_ops.reduce_mean(logits * math_ops.log(targets + eps) +
                                     (1 - logits) * math_ops.log(1 - targets + eps))

def l2_loss(logits=None, labels=None, name=''):
    with ops.name_scope("l2_loss" + name) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(labels, name="targets")
        return math_ops.reduce_mean(math_ops.square(logits-targets))

def weight(name, shape, init='he', value = 0.0, range=None):
    """ Initializes weight.
    :param name: Variable name
    :param shape: Tensor shape
    :param init: Init mode. xavier / normal / uniform / he (default is 'he')
    :param range:
    :return: Variable
    """
    initializer = tf.constant_initializer()

    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]

    if init == 'xavier':
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)

    elif init == 'he':
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif init == 'uniform':
        if range is None:
            raise ValueError("range must not be None if uniform init is used.")
        initializer = tf.random_uniform_initializer(-range, range)

    elif init == 'constant':
        initializer = init_ops.constant_initializer(value)

    var = tf.get_variable(name, shape, initializer=initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))
    return var
