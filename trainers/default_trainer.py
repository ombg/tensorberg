#from data_loader.cifar_imgdb import ImgdbLoader
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.logger import DefinedSummarizer

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__init__(sess, model, config,logger, data_loader=data)

    def train_epoch(self):
        self.data_loader.initialize(self.sess)
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
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy])
        return loss, acc
