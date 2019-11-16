# Repo for the Rare Disease Detection paper work

We implemented the code of CONAN.

## Files

The "data" folder contains the data preprocessing files __{ipf,nash,ibd}\_data\_process.py__, the training data files __{ipf,nash,ibd}\_train\_data\_{x}.py__, and the testing data files __{ipf,nash,ibd}_test_data.py__.

{x} as {10, 100} indicates the ratio of the negative vs positive samples in the training data.

The "baseline" folder contains the codes of compared methods __models.py__.

The root folder contains our method files __our.py__ and __cgan.py__, and original GAN codes __gan.py__.

__models.py__ and __our.py__ both contain two options. Comment the codes between "### Option x: ..." "### End build ..." before running the other option.

## Requirements

- Python 3.6x

- Keras 1.0.x

- Numpy 1.14.x

- Tensorflow 1.13.x

- TensorFlow GPU 1.10.x

## Parameters

- gamma=2., alpha=.25, followed "Focal loss for dense object detection"

- MAX_SENT_LENGTH = 10, 241, 798 for IPF, NASH and IBD respectively, which are the ave. # of visits

- MAX_SENTS = 300, 1, 1 for IPF, NASH and IBD respectively

- EMBEDDING_DIM = 64, 128, or 256, where 128 achieves the best performance

- epochs=10, set # of epochs through validation (no valiadation now)

- batch_size <= 1024 to achieve the best performance

- n_samples = 200, 20, 20 for IPF, NASH and IBD respectively

## How to run

Note: This code is written in Python3.
```
python data/ipf_data_process.py
python models.py # Running compared methods
python our.py # Running our method
```