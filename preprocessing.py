import cv2
import numpy as np
import albumentations as alb
from PIL import Image
from torchvision import transforms
from data import SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH


def create_transformer_v1(input_size=None, augmentor=None):
    if input_size is None:
        input_size = (SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH)
        if augmentor:
            transformer = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                TransformWithAlbumentations(augmentor),
                transforms.ToTensor()
            ])
        else:
            transformer = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(input_size),
                transforms.ToTensor()
            ])
    else:
        if augmentor:
            transformer = transforms.Compose([
                transforms.Resize(input_size),
                TransformWithAlbumentations(augmentor),
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


def create_augmentor_v1(enable_random_morph=False):
    if enable_random_morph:
        ts = [
            alb.OneOf([
                alb.GridDistortion(p=1),
                alb.ShiftScaleRotate(rotate_limit=10, p=1)
            ], p=0.5),
            alb.RandomBrightness(p=0.5),
            RandomMorph(p=0.5)
        ]
    else:
        ts = [
            alb.OneOf([
                alb.GridDistortion(p=1),
                alb.ShiftScaleRotate(rotate_limit=10, p=1)
            ], p=0.5),
            alb.RandomBrightness(p=0.5),
            alb.Blur(p=0.5)
        ]
    augmentor = alb.Compose(ts)
    return augmentor


class TransformWithAlbumentations:
    def __init__(self, augmentor,
                 force_to_input_ndarray=True,
                 force_to_output_pil=True):
        self.augmentor = augmentor
        self.force_to_input_ndarray = force_to_input_ndarray
        self.force_to_output_pil = force_to_output_pil

    def __call__(self, img):
        if self.force_to_input_ndarray:
            img = np.array(img)
        aug = self.augmentor(image=img)
        img = aug['image']
        if self.force_to_output_pil:
            img = Image.fromarray(img.astype(np.uint8))
        return img


class RandomMorph(alb.ImageOnlyTransform):
    """https://www.kaggle.com/c/bengaliai-cv19/discussion/128198
    """
    def __init__(self, _min=2, _max=6, element_shape=cv2.MORPH_ELLIPSE, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self._min = _min
        self._max = _max
        self.element_shape = element_shape

    def apply(self, image, **params):
        arr = np.random.randint(self._min, self._max, 2)
        kernel = cv2.getStructuringElement(self.element_shape, tuple(arr))

        if random.random() > 0.5:
            # make it thinner
            image = cv2.erode(image, kernel, iterations=1)
        else:
            # make it thicker
            image = cv2.dilate(image, kernel, iterations=1)

        return image