import tensorflow as tf

from tqdm import tqdm
import numpy as np
from utils.logger import Logger

class Trainer:
    def __init__(self, sess, model, data_loader, config, logger):
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
        self.logger = logger
        self.config = config
        self.sess = sess
        if data_loader is not None:
            self.data_loader = data_loader

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
        # Load the training dataset
        self.data_loader.initialize_train(self.sess)
        loop = tqdm(range(self.data_loader.num_batches), ascii=True)
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        train_epoch_loss = np.mean(losses)
        train_epoch_acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it,
                              summarizer="train",
                              scope="subset_1000",
                              summaries_dict={'train/loss_per_epoch': train_epoch_loss,
                                              'train/acc_per_epoch': train_epoch_acc})
        # Check the validation accuracy once per epoch
        self.data_loader.initialize_val(self.sess)
        val_epoch_loss, val_epoch_acc = self.validation_step()

        self.logger.summarize(cur_it,
                              summarizer="test",
                              scope="subset_1000",
                              summaries_dict={'val/loss_per_epoch': val_epoch_loss,
                                              'val/acc_per_epoch': val_epoch_acc})
        self.model.save(self.sess)

    def train_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        fetches = [self.model.train_step,
                   self.model.loss,
                   self.model.accuracy]
        _, loss, acc = self.sess.run(fetches)
        return loss, acc

    def validation_step(self):
        """
        Runs all graph ops which need validation
        """
        fetches = [self.model.loss,
                   self.model.accuracy]
        loss, acc = self.sess.run(fetches)
        return loss, acc
