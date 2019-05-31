from torchvision.models import AlexNet
from dataset_distillation.networks import AlexCifarNet


def alexnet_wrapper(c_out):
    return AlexNet(num_classes=c_out)


class FakeDistillState:
    num_classes = 10
    nc = 3


def alex_cifar_net_wrapper(c_out):
    state = FakeDistillState
    state.num_classes = c_out
    return AlexCifarNet(state)
