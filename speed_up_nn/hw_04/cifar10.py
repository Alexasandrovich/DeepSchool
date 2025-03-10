import torch
import torchvision.transforms as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class Cutout(torch.nn.Module):
    def __init__(self, p=1.0, size=(8, 8)):
        super().__init__()
        self.p = p
        self.size = size

    def forward(self, img):
        if torch.rand(1) < self.p:
            h, w = img.shape[1:]
            start_y = torch.randint(size=(1,), low=0, high=(h - self.size[0])).item()
            start_x = torch.randint(size=(1,), low=0, high=(w - self.size[1])).item()
            img[start_y:start_y + self.size[0], start_x:start_x + self.size[1]] = 0.0
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        Cutout(size=(8, 8)),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
