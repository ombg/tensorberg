from shutil import rmtree
import os.path
import unittest
import tempfile
import numpy as np
from numpy.testing import assert_allclose

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

if __name__ == '__main__':
    unittest.main()

        
