import os
from tqdm import tqdm
from typing import List, Dict, Callable

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter

from fling.utils import TVLoss


def _to_image(img: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor image into the image form.
    """
    im_copy = img.clone().detach().cpu().squeeze(0).numpy()
    im_copy = im_copy.transpose(1, 2, 0)
    im_copy = (255 * im_copy.clip(0, 1)).astype('uint8')
    return im_copy


def _abs_contrib_crop(img: torch.Tensor, threshold: int = 10) -> torch.Tensor:
    """
    Crop the generated image according to its abs value.
    """
    img = img.cpu()
    abs_img = torch.abs(img)
    smalls = abs_img < np.percentile(abs_img, threshold)

    return img - img * smalls


def _norm_crop(img: torch.Tensor, threshold: int = 10) -> torch.Tensor:
    """
    Crop the generated image according to its norm.
    """
    norm = torch.norm(img, dim=0)
    norm = norm.cpu().numpy()

    # Create a binary matrix, with 1's wherever the pixel falls below threshold
    smalls = norm < np.percentile(norm, threshold)
    smalls = np.tile(smalls, (3, 1, 1))

    # Crop pixels from image
    crop = img.cpu() - img.cpu() * smalls
    return crop


def _get_layer_hook(act_dict: Dict, layer_id: str, channel_id: int) -> Callable:
    """
    Return a proper hook function, using the given layer name and channel id. The activation captured by this \
    hook function will be saved into ``act_dict``.
    """

    def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if len(output.shape) == 2:
            channel_out = output[:, channel_id]
        elif len(output.shape) == 4:
            channel_out = output[:, channel_id, ...]
        else:
            raise ValueError(f'Can not process feature with shape: {output.shape}.')
        act_dict[layer_id] = torch.mean(channel_out)

    return hook


class ActivationMaximizer:
    """
    Overview:
        This class implement activation maximization method introduced in "Visualizing Higher-Layer Features of \
        a Deep Network" (https://docplayer.net/17848924-Visualizing-higher-layer-features-of-a-deep-network.html). \
        The concrete implementation is based on: https://github.com/Nguyen-Hoa/Activation-Maximization.
    """

    def __init__(
        self,
        iteration: int,
        working_dir: str,
        iteration_per_save: int = 10,
        tv_weight: float = 0,
        enable_gaussian_blur_normalizer: bool = False,
        enable_contrib_crop_normalizer: bool = False,
        enable_norm_crop_normalizer: bool = False
    ):
        """
        Overview:
            Initialize the class using given arguments.
        Arguments:
            iteration: The number of iterations for maximizing the input image.
            working_dir: The working dir of this visualizer. The results will be saved to this path.
            iteration_per_save: The interation interval between saving two images. Default to be 10.
            tv_weight: The weight of total variance loss (TV-loss). A larger weight can result in more "smooth" images.
            enable_gaussian_blur_normalizer: Whether to enable gaussian blur regularizer, default to be ``False``.
            enable_contrib_crop_normalizer: Whether to enable abs value regularizer, default to be ``False``.
            enable_norm_crop_normalizer: Whether to enable norm crop regularizer, default to be ``False``.
        """
        self.iteration = iteration
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        self.base_index = 0
        self.iteration_per_save = iteration_per_save
        self.tv_weight = tv_weight

        self.enable_gaussian_blur_normalizer = enable_gaussian_blur_normalizer
        self.enable_contrib_crop_normalizer = enable_contrib_crop_normalizer
        self.enable_norm_crop_normalizer = enable_norm_crop_normalizer

    def activation_maximization(
            self,
            model: nn.Module,
            layer_id: str,
            channel_id: int,
            image_shape: List[int],
            learning_rate: float,
            device: str,
            save_img: bool = True,
            weight_decay: float = 0.
    ) -> None:
        """
        Overview:
            Maximize activation of certain neurons and find the corresponding input image.
        Arguments:
            model: The model to be visualized.
            layer_id: The layer name for visualization.
            channel_id: The index of channel in the corresponding layer.
            image_shape: The image shape of the result image.
            learning_rate: The learning rate for updating input image.
            device: Device to run this attack algorithm, such as ``"cpu"`` and ``"cuda"``.
            save_img: Whether to save the reconstructed images into working dir. Default to be ``True``.
            weight_decay: The weight decay value for updating input image.
        """
        dummy_image = torch.randn(image_shape).to(device).unsqueeze(0)
        dummy_image.requires_grad = True
        model = model.to(device)
        model = model.eval()

        optimizer = torch.optim.Adam([dummy_image], lr=learning_rate, weight_decay=weight_decay)

        # Initialize activation diction. Add hook function to the corresponding layer.
        activation_dict = {}
        dict(model.named_modules())[layer_id].register_forward_hook(
            _get_layer_hook(act_dict=activation_dict, layer_id=layer_id, channel_id=channel_id)
        )

        tv_criterion = TVLoss()
        total_imgs = []

        for iters in tqdm(range(self.iteration)):
            _ = model(dummy_image)
            # Calculate loss function.
            activation_loss = -activation_dict[layer_id]
            tv_loss = tv_criterion(dummy_image)
            loss = activation_loss + self.tv_weight * tv_loss

            # Update the input image.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Regularize the generated image using different methods.
            if self.enable_gaussian_blur_normalizer and iters % 4 == 0:
                temp = dummy_image.squeeze(0)
                temp = temp.detach().cpu().numpy()
                for channel in range(3):
                    temp[channel] = gaussian_filter(temp[channel], sigma=0.1)
                temp = torch.from_numpy(temp)
                dummy_image.data = temp.unsqueeze(0).to(device)

            if self.enable_norm_crop_normalizer:
                dummy_image = _norm_crop(dummy_image.detach().squeeze(0)).to(device)
                dummy_image = dummy_image.unsqueeze(0)

            if self.enable_contrib_crop_normalizer:
                dummy_image = _abs_contrib_crop(dummy_image.detach().squeeze(0)).to(device)
                dummy_image = dummy_image.unsqueeze(0)

            # Save the image for later gif generation.
            if iters % self.iteration_per_save == 0:
                img = _to_image(dummy_image)
                total_imgs.append(img)

        if save_img:
            import imageio
            imageio.mimsave(os.path.join(self.working_dir, f'{layer_id}_{channel_id}.gif'), total_imgs, duration=0.25)
            last_image = _to_image(dummy_image)
            imageio.imwrite(os.path.join(self.working_dir, f'{layer_id}_{channel_id}_final.png'), last_image)
