# DeepSequence Pytorch implementation

This repository contains an implementation of [DeepSequence](https://github.com/debbiemarkslab/DeepSequence) in Pytorch, following the same structure of the repository as the original project: <br>
    - helper.py: same exact file from DeepSequence, used only to inherit the DataHelper class in the deepsequence_eperiment.ipynb notebook. <br>
    - model.py: contains the model. The model allows to make a Vanilla VAE or to use the blitz library to make the decoder Bayesian. <br>
    - train.py: contains the main training loop that is called in deepsequence_eperiment.ipynb <br>
    - model_performance_tests.py: contains accuracy test to evaluate the trained model on <br>
