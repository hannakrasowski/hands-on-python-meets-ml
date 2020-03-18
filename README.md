# hands-on-python-meets-ml

This git provides a short introduction to python, jupyter notebooks and supervised training of neural networks. There are four minitasks in the main notebook *hands-on-python-meets-machine-learning*. The solution is provided by the notebook *hands-on-python-meets-machine-learning-solutions*

You can either run everthing on kaggle:

TUTORIAL: 
SOLUTIONS:	https://www.kaggle.com/hannakrasowski/hands-on-python-meets-machine-learning-solutions

But the public resources are limited. So if you really want to play around and understand all of the code you should install a jupyter notebook distribution on your device. Therefore we provide a short describtion of how to install Anaconda and how to get this repository running.

## Installation of Anaconda

There are several ways to install jupyter notebook. The most convenient one is using Anaconda, because you also have a graphical interface and can have multiple environments with different packages installed for different tasks.

For a stepwise manual for installation please refer to: https://docs.anaconda.com/anaconda/install/

## Setup for this git

Open the Anaconda Prompt and go to the root folder of this git repository

`cd C\User\...\hands-on-python-meets-ml`

Run the following command in order to install the required packages in your base version of Anaconda

`conda install --yes --file requirements.txt`

Then run the next command to start jupyter notebook

`jupyter notebook`
