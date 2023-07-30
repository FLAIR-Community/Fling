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


def get_data_transform(cfg, train):
    transforms.ToTensor()
    transform_names = cfg.keys()
    results = []
    results.append(transforms.ToTensor())
    for k in transform_names:
        transform_args = cfg[k]
        name = k
        if name == 'resize':
            results.append(transforms.Resize(transform_args.size))
        elif name == 'random_resized_crop':
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
            results.append(transforms.Normalize(transform_args.mean, transform_args.std))
        elif name == 'random_crop':
            if train:
                results.append(transforms.RandomCrop(transform_args.size, transform_args.padding))
        elif name == 'to_device':
            results.append(ToDevice(transform_args.device))
        else:
            raise ValueError(f'Unrecognized data transform method: {name}')
    return transforms.Compose(results)
