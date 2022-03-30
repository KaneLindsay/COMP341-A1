from Train import TrainNetwork
from Test import TestNetwork


def run(train=False, test=True):
    if train:
        TrainNetwork()
    if test:
        TestNetwork()


run()
