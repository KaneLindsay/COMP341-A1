from multiprocessing import freeze_support
from Train import TrainNetwork
from Test import TestNetwork


def run(train=False, test=True):
    if train:
        TrainNetwork(num_epochs=100)
    if test:
        TestNetwork()


if __name__ == '__main__':
    freeze_support()
    run(train=False)
