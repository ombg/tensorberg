import unittest
from datahandler import FileListDatasetLoader
import config
import testdata

class FileListDatasetLoaderTestCase(unittest.TestCase):
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
        self.dsetsloader = FileListDatasetLoader(testcase_config_reloaded)
        
    def test_create_file_lists(self):
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
