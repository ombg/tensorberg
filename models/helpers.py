import functools
import tensorflow as tf

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


#def variable_summaries(var, name='summaries'):
#    """
#    Attach a lot of summaries to a Tensor
#    (for TensorBoard visualization).
#    """
#    with tf.name_scope(name):
#        mean = tf.reduce_mean(var)
#        tf.summary.scalar('mean', mean)
#        with tf.name_scope('stddev'):
#            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#        tf.summary.scalar('stddev', stddev)
#        tf.summary.scalar('max', tf.reduce_max(var))
#        tf.summary.scalar('min', tf.reduce_min(var))
#        tf.summary.scalar(name, var)
#        tf.summary.histogram(name, var)

