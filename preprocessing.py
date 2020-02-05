from torchvision import transforms
from data import SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH


def create_transformer_v1(input_size=None):
    if input_size is None:
        input_size = (SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH)
        transformer = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomRotation(5),
            transforms.ToTensor()
        ])

    return transformer


def create_testing_transformer_v1(input_size=None):
    if input_size is None:
        input_size = (SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH)
        transformer = transforms.ToTensor()
    else:
        transformer = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])

    return transformer