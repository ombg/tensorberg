from shutil import rmtree
import os.path
import unittest
import tensorflow as tf
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from utils import config
from models.regression import VggMod
import default_trainer

class TestSaveBatch(unittest.TestCase):
    def test_fileio(self):
        path_name = tempfile.mkdtemp()
        try:
            data = np.ones([3,56,56,3], dtype=np.float32)
            data[0] *= 0.
            data[2] *= 2.
            default_trainer.save_batch(data, path_name)
            file_content_0 = np.load(os.path.join(path_name,'pred_0.npy'))
            file_content_1 = np.load(os.path.join(path_name,'pred_1.npy'))
            file_content_2 = np.load(os.path.join(path_name,'pred_2.npy'))
        finally:
            rmtree(path_name)
        assert_allclose(np.zeros([56,56,3]), file_content_0)
        assert_allclose(np.ones([56,56,3]), file_content_1)
        assert_allclose(2. * np.ones([56,56,3]), file_content_2)

class TestTrainer(tf.test.TestCase):
    def setUp(self):
        import json
        testcase_config_filename = '/tmp/testcase_config.json'
        testcase_config = {
            "input_path": "testing.txt",
            "testing_percentage": "10",
            "validation_percentage": "10",
            "weights_file": "/tmp/vgg16_weights.npz",
            "dataset_name": "imgdb",
            "exp_name": "imgdb_vgg16_fc2_4096_cl4_dr_0.1",
            "work_dir": "dummy",
            "checkpoint_to_restore": "dummy",
            "num_epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 1,
            "max_to_keep":5
        }
        with open(testcase_config_filename, 'w') as f:
            json.dump(testcase_config, f)
        testcase_config_reloaded = config.process_config(testcase_config_filename)
        model = VggMod(testcase_config_reloaded)
        with self.test_session() as sess:
            self.trainer = default_trainer.Trainer(sess, model, testcase_config_reloaded) 

    def test_predict(self):
        """Test if an OSError is handled by predict()
        """
        file_list = ['/tmp/file1.txt','/tmp/file2.txt']
        try:
            self.trainer.predict(file_list)
        except OSError as err:
            self.fail('predict() should catch an OSError.')

    def test_imread(self):
        """Test if a FileNotFoundError is handled by predict()
        """
        file_name = '/tmp/file1jflskajfsldaf.txt'
        try:
            self.trainer.predict(file_name)
        except FileNotFoundError as err:
            self.fail('predict() should catch a FileNotFoundError.')
if __name__ == '__main__':
    unittest.main()

        
