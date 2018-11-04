# Tensorberg - Image-based machine learning with TensorFlow's low-level API

The best way to really understand what is going on is to code it yourself, rather than depending on high-level APIs. Here, I share some of the code - based on TensorFlow 1.10 - which I use for classification tasks.
Feel free to open an issue or pull request.
Some features are:

* Usage of TensorFlow's low-level Python API. Nothing is hidden. It is more lines of code, but you see what is going on.
* Usage of `tf.data.Dataset` whereever possible. No need to use `feed_dict`.
* Usage of Tensorboard and multiple tf.summary.FileWriter`s to compare training and validation accuracy in one graph.
* Usage of function decorators (TODO) to structure the code nicely.

## Installation

* Clone the repository
* Inside the repository's root directory, type
{{{
pipenv install
}}}

This installs some required packages. 
It is still a work in progress but might help to get people started with data handling adn simple classification models in TensorFlow.

This work is inspired by the following repositories and blog posts:
* ["TensorFlow Project Template"](https://github.com/MrGemy95/Tensorflow-Project-Template)
* ["Structuring Your TensorFlow Models"](https://danijar.com/structuring-your-tensorflow-models/) by Danijar Hafner
* ["Finetune AlexNet with Tensorflow"](https://github.com/kratzert/finetune_alexnet_with_tensorflow)
