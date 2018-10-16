import tensorflow as tf

from tqdm import tqdm
import numpy as np
#from utils.logger import DefinedSummarizer

class Trainer:
    def __init__(self, sess, model, data, config):
        """
        Constructing the trainer
        :param sess: TF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all the configurations you have specified in the json
        :param logger: logger class which will summarize and write the values to the tensorboard
        :param data: The data loader if specified. 
        """
        # Assign all class attributes
        self.model = model
        #self.logger = logger
        self.config = config
        self.sess = sess
        if data is not None:
            self.data = data

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary

        :param epoch: take the number of epoch if you are interested
        :return:
        """
        self.data.initialize(self.sess)
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        #TODO DEBUG
        print(loss, acc)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        #self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy])
        return loss, acc
