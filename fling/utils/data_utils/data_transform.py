import torchvision.transforms
from torchvision import transforms


class ToDevice:
    r"""
    Overview:
        Convert an image with type ``torch.Tensor`` to a specific device.
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, pic):
        """
        Overview:
            Put image to the desired device.
        Arguments:
            pic: Image to be moved.
        Returns:
            Tensor: Converted image.
        """
        return pic.to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + f'({self.device})'


def get_data_transform(cfg: dict, train: bool) -> torchvision.transforms.Compose:
    r"""
    Overview:
        Given the configurations for data augmentation, return the data-transformations.
    Arguments:
        cfg: the input configurations for data augmentation.
        train: whether the returned transformations are for train-dataset or test-dataset.
    Returns:
        Compose: the composed data-transformations.
    """
    transform_names = cfg.keys()
    results = []
    # Note that: the ``ToTensor()`` will be automatically added into the results.
    # Here is a trick for improve efficiency:
    # If ``Normalize()`` is in the defined data-augmentations, the ``ToTensor()`` will be added just before it.
    # Otherwise, ``ToTensor()`` will be added at last.
    has_norm = False

    # Add data-augmentations one-by-one.
    for k in transform_names:
        # Get the name of data-augmentation.
        transform_args = cfg[k]
        name = k
        if name == 'resize':
            results.append(transforms.Resize(transform_args.size))
        elif name == 'random_resized_crop':
            # The augmentations added for train dataset and test dataset are different.
            if train:
                results.append(
                    transforms.RandomResizedCrop(transform_args.size, transform_args.scale, transform_args.ratio)
                )
            else:
                results.append(transforms.Resize(transform_args.size))
        elif name == 'color_jitter':
            if train:
                results.append(
                    transforms.ColorJitter(
                        transform_args.brightness, transform_args.contrast, transform_args.saturation
                    )
                )
        elif name == 'horizontal_flip':
            if train:
                results.append(transforms.RandomHorizontalFlip(transform_args.p))
        elif name == 'vertical_flip':
            if train:
                results.append(transforms.RandomVerticalFlip(transform_args.p))
        elif name == 'random_rotation':
            if train:
                results.append(transforms.RandomRotation(transform_args.degree))
        elif name == 'Normalize':
            results.append(transforms.ToTensor())
            results.append(transforms.Normalize(transform_args.mean, transform_args.std))
            has_norm = True
        elif name == 'random_crop':
            if train:
                results.append(transforms.RandomCrop(transform_args.size, transform_args.padding))
        elif name == 'to_device':
            results.append(ToDevice(transform_args.device))
        else:
            raise ValueError(f'Unrecognized data transform method: {name}')

    # Add ``ToTensor()`` if ``Normalize()`` is not defined.
    if not has_norm:
        results.append(transforms.ToTensor())
    return transforms.Compose(results)
