import unittest
import datahandler
import testdata

class DataHandlerTestCase(unittest.TestCase):
    def setUp(self):
        self.txt_file = 'testing.txt'
        self.class_names = { 0: 'no_crowd',
                             1: 'sparse',
                             2: 'medium_dense',
                             3: 'dense' }
        
    def test_create_file_lists_from_list(self):
        self.assertDictEqual(
            datahandler.create_file_lists_from_list(
                self.txt_file,
                10,
                10,
                class_names=self.class_names), 
            testdata.gold_result)

if __name__ == '__main__':
    unittest.main()
