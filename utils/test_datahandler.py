import unittest
from datahandler import FileListDatasetLoader
import config
import testdata

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

if __name__ == '__main__':
    unittest.main()
