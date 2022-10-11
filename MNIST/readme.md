MNIST - Classification of Handwritten Digits

I've created two models that can be run by simply cloning this repo and running the files "cnn.py"" or "ffnn_train_test.py".

They are both pretty simple and are meant to achieve great accuracy (<95%) with the least amount of time for training.

It uses the standard 60k/10k sample split for training and testing respectively.

I have enabled training on M1 Mac GPU though the standard CPU device can also be used.

How to call on terminal:

cd where your file in is your computer

python3 /.../cnn.py --epochs 5 --device mps


This runs the CNN model with 5 epochs and on GPU. The following parameters can be changed wheen the file is executed:

--batch-size - 64 default \
--test-batch-size - 100 default \
--epochs - 1 default \
--lr - 1.0 default (initial learning rate) \
--gamma - 0.1 default (lr decay scalar) \
--device - "cpu" or "mps" (mps is for GPU) \
--dry-run - False default (do single pass for speed) \
--seed - 1 default (random seed) \
--log-interval - 100 (how many batches to wait before logging training status) \
--save-model - False default

