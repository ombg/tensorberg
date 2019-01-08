import tensorflow as tf
import numpy as np
import imageio
from skimage import filters
from skimage.transform import resize

import data_utils

class DataUtilsTest(tf.test.TestCase):

    def test_blur_png(self):
        with self.test_session() as sess:
            img_in = imageio.imread("test_img.png")
            expected_blurred = filters.gaussian(img_in, sigma=0.1)
            expected_blurred = resize(expected_blurred, [56,56,1])
            filename = tf.constant("test_img.png", tf.string)
            actual_blurred_op = data_utils.blur_png(filename)
            actual_blurred = actual_blurred_op.eval()
            imageio.imwrite("/tmp/test_img_blurred.png", expected_blurred)
            imageio.imwrite("/tmp/test_img_tf_blurred.png", actual_blurred)
            self.assertAlmostEqual(np.all(actual_blurred), np.all(expected_blurred),places=10,delta=1)

if __name__ == '__main__':
    tf.test.main()
