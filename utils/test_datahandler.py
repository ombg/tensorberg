import unittest
import tensorflow as tf
import datahandler
import config
import testdata

import numpy as np

class FileListDatasetLoaderTestCase(unittest.TestCase):
    def setUp(self):
        import json
        testcase_config_filename = '/tmp/testcase_config.json'
        testcase_config = {
            "data_path":"utils/testing.txt",
            "data_path_samples":"/tmp/in_dir/images",
            "data_path_gt":"/tmp/in_dir/maps",
            "data_path_pred":"/tmp/in_dir/pred_maps",
            "is_training": "False",
            "validation_percentage": "10",
            "testing_percentage": "10",
            "weights_file": "/tmp/vgg16_weights.npz",
            "dataset_name": "crowd_maps",
            "work_dir": "/tmp/work_dir/",
            "exp_name": "crowdnetreg_debug",
            "checkpoint_to_restore": "no_restore",
            "num_epochs": 20,
            "learning_rate": 0.001,
            "batch_size": 10,
            "max_to_keep":5
        }
        with open(testcase_config_filename, 'w') as f:
            json.dump(testcase_config, f)
        testcase_config_reloaded = config.process_config(testcase_config_filename)
        self.dsetsloader = datahandler.FileListDatasetLoader(testcase_config_reloaded)
        
    def test_create_file_lists(self):
        #self.maxDiff=None
        self.assertDictEqual(
            self.dsetsloader.image_lists,
            testdata.gold_result)

    def tearDown(self):
        with open('/tmp/test_dump.txt', 'w') as f:
            f.write('{}'.format(self.dsetsloader.image_lists))
            f.write('\n\nGoldresult\n\n')
            f.write('{}'.format(testdata.gold_result))

class TFRecordDatasetLoaderTestCase(tf.test.TestCase):

    def setUp(self):
        self.dset = datahandler.dset_from_tfrecord(
                        '/tmp/cifar_tfrecord/train.tfrecords',
                        do_shuffle=True,
                        use_distortion=True,
                        repetitions=-1)

        self.iterator = self.dset.make_one_shot_iterator()
        self.image_batch, self.label_batch = self.iterator.get_next()

    def test_shape(self):
        self.assertEqual(self.dset.output_types, (tf.float32, tf.float32))
        img_batch_shape = self.dset.output_shapes[0].as_list()
        label_batch_shape = self.dset.output_shapes[1].as_list()
        self.assertEqual(img_batch_shape, [None,32,32,3])
        self.assertEqual(label_batch_shape, [None, 10])

    def test_dset_from_tfrecord(self):
        """Test if `tf.data.Dataset` is loaded.
        """
        with self.test_session() as sess:
            self.assertLess(tf.reduce_min(self.image_batch).eval(), 20.0)
            self.assertGreater(tf.reduce_max(self.image_batch).eval(), 230.0)
            self.assertAlmostEqual(tf.reduce_mean(self.image_batch).eval(),
                                   127.0, delta=30.0)

if __name__ == '__main__':
    unittest.main()
