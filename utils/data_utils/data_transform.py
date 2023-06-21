from torchvision import transforms


def get_data_transform(cfg, train):
    transform_names = cfg.keys()
    results = []
    results.append(transforms.ToTensor)
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
                        transform_args.brightness,
                        transform_args.contrast,
                        transform_args.saturation
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
        else:
            raise ValueError(f'Unrecognized data transform method: {name}')
    return results
