"""
Implementation of the two-layer net from the cs231n course
in Tensorflow.
Visualization with summeries using Tensorboard
"""

import numpy as np
import tensorflow as tf

run_id = 0
logs_path = '/tmp/tensorflow/cs231n/assignment1/run' + str(run_id)

def init_toy_data(M, N):
  """
  Initialize toy data.

  M: Number of training samples.
  N: Dimension of the training samples. 
  X: Randomly generated training samples; has shape (M, N) 
  y: Target labels; must have shape (M, 1)
  """
  np.random.seed(0)
  X = 10 * np.random.randn(M, N)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

def inference():
  #TODO More code here.  
  tf.nn.relu()
  return y_score

def loss(X, y_target):
  loss = None
  y_score = inference(weights, X, bias) 
  return cross_entropy_loss, y_score

def main(): 

  # Define the network's size
  input_size = 4
  hidden_size = 10
  num_classes = 3
  num_inputs = 5

  X, y = init_toy_data(num_inputs, input_size)
  # Evaluate the computed loss
  scores = loss(X)
  print( 'Your scores:')
  print( scores)
  print()
  print( 'correct scores:')
  correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215 ],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])
  print( correct_scores)
  print()

  # The difference should be very small. We get < 1e-7
  print( 'Difference between your scores and correct scores:')
  print( np.sum(np.abs(scores - correct_scores)))

if __name__ == "__main__":
  main()

#
# Old stuff
#
training_epochs = 5000
display_step = 100

train_x = np.linspace(-1, 1, 101)
train_t = 3 * train_x + np.random.randn(*train_x.shape) * 0.33 + 0.5

n_samples = train_x.shape[0]

# Create the graph
x = tf.placeholder(tf.float32, shape=None, name="train_samples")
t = tf.placeholder(tf.float32, shape=None, name="targets")

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name='bias')

# Score function
y = tf.add(tf.multiply(x, w), b)

# loss function
# tf.name_scope creates a large gray box including everything
# in the "with block"
# This is for the "Graph" section in Tensorboard.
with tf.name_scope('MSEloss'):
  loss = tf.reduce_sum(tf.pow(t - y, 2)) / (2*n_samples)
# This let you monitor the change of the value "loss"
# but you need to add the output with writer.add_summary()
# in the session block below.
tf.summary.scalar('mse_loss', loss)

# Built-in optimizer function
# While monitoring the change of the loss function
# with Tensorboard, you can play around with different 
# optimizers.
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# plot graph in Tensorboard
summary_op = tf.summary.merge_all()
the_graph=tf.get_default_graph()
log_writer = tf.summary.FileWriter(logs_path, graph=the_graph)

init_op = tf.global_variables_initializer()

# Start the session
with tf.Session() as sess:
  sess.run(init_op) # Init op MUST be called by a run() function.
  for epoch in range(training_epochs):
    summary, _ = sess.run([summary_op, optimizer], feed_dict={x:train_x, t:train_t})
    log_writer.add_summary(summary, epoch)
    # Slow implementation
    # for(xx,tt) in zip(train_x, train_t):
    #   sess.run(optimizer, feed_dict={x:tt, t:tt})

  print("Optimization Finished!")
  training_loss = sess.run(loss, feed_dict={x:train_x, t:train_t})
  print("Training loss=", training_loss, "W=", sess.run(w), "b=", sess.run(b), '\n')

