import math
import os
from tqdm import tqdm
from typing import Union, Dict, Tuple
from easydict import EasyDict

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

from fling.utils import Logger
from fling.utils import TVLoss
from fling.utils import get_weights


def _l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Calculate the l2 distance between two tensors.
    return torch.sum((x - y) ** 2).sum()


def _cos_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Calculate the cos distance between two tensors.
    assert x.shape == y.shape
    x = torch.flatten(x)
    y = torch.flatten(y)
    return 1 - F.cosine_similarity(x, y, dim=0, eps=1e-10)


def _reconstruction_psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Given a batch of real images and reconstructed images, calculate the reconstruction loss.
    assert x.shape == y.shape
    x = torch.flatten(x, start_dim=1)
    y = torch.flatten(y, start_dim=1)
    psnr = -10 * torch.log10(torch.mean((x - y) ** 2, dim=1))
    return psnr


class DLGAttacker:
    """
    Overview:
        A series of attack methods that recover input images using calculated gradients, i.e. leakage from gradients.
    Reference paper: 1) Deep Leakage from Gradients <link https://arxiv.org/pdf/1906.08935.pdf link>, known as DLG. \
    2) Inverting Gradients - How easy is it to break privacy in federated learning? \
    <link https://arxiv.org/pdf/2003.14053.pdf link>, known as iDLG.
    """

    def __init__(
        self,
        iteration: int,
        working_dir: str,
        iteration_per_save: int = 10,
        distance_measure: str = 'euclid',
        tv_weight: float = 0
    ):
        """
        Overview:
            Initialize the attacker object.
        Arguments:
            iteration: The number of iterations for recovering the input image.
            working_dir: The working dir of this attacker. The attack results will be saved to this path.
            iteration_per_save: The interation interval between saving two reconstructed images. Default to be 10.
            distance_measure: The metric for measuring the distance between recovered gradient and ground truth \
                gradient. If set to "euclid", the algorithm is equivalent to DLG, and if set to "cos", the algorithm \
                is equivalent to iDLG.
            tv_weight: The weight of total variance loss (TV-loss). A larger weight can result in more "smooth" \
                recovered images.
        """
        self.iteration = iteration
        self.working_dir = working_dir
        self.logger = Logger(self.working_dir)
        self.base_index = 0
        self.iteration_per_save = iteration_per_save
        self.tv_weight = tv_weight

        if distance_measure == 'cos':
            self.metric = _cos_distance
        elif distance_measure == 'euclid':
            self.metric = _l2_distance
        else:
            raise ValueError(f'Unrecognized distance measure: {distance_measure}')

    def attack(
            self,
            model: nn.Module,
            dataset: Dataset,
            device: str,
            class_number: int,
            parameter_args: Union[Dict, EasyDict] = {"name": "all"},
            batch_size: bool = 1,
            use_gt_labels: bool = True,
            save_img: bool = False,
            optim_backend: str = 'lbfgs',
    ) -> Tuple[float, float]:
        """
        Overview:
            The main attack function.
        Arguments:
            model: The model to be attacked.
            dataset: The dataset tried to be recovered by the attacker.
            batch_size: Batch-size of the input images. A larger batch size will be harder to attack. Default to be 1.
            parameter_args: Parameters whose gradients are visible to attackers, which typically equals to \
                ``aggregation_parameters`` in the config file.
            device: Device to run this attack algorithm, such as ``"cpu"`` and ``"cuda"``.
            class_number: The number of classes for the classification task.
            use_gt_labels: Whether the ground truth label is able to be acquired by the attacker. Default to be \
                ``True``. If set to be ``False``, it will be harder to attack.
            save_img: Whether to save the reconstructed images into working dir. Default to be ``False``.
            optim_backend: Backend for optimizing the recovered image. Supports "lbfgs" and "adam".
        Returns:
            final_loss: The reconstruction loss after the final iterations across all samples in the dataset. \
                A smaller value indicates a better performance for the attacker.
            min_loss: The min reconstruction loss of all iterations across all samples in the dataset. \
                A smaller value indicates a better performance for the attacker.
        """
        start, end = 0, batch_size
        batch_idx = 0
        total_loss = []
        total_min_loss = []

        model = model.to(device)
        model = model.eval()
        tv_criterion = TVLoss()

        # Get the parameters whose gradients are visible to the attacker.
        if not isinstance(parameter_args, EasyDict):
            parameter_args = EasyDict(parameter_args)
        visible_parameters = get_weights(model, parameter_args)

        # Iterate for each batch in the dataset.
        while end <= len(dataset):
            if start == end:
                break
            batch = dataset[start:end]
            start = end
            end = min(len(dataset), end + batch_size)
            self.logger.logging(f'Star batch: {batch_idx}...')
            batch_idx += 1

            # Prepare input data and labels.
            batch_x = torch.stack([batch[i]['input'] for i in range(len(batch))]).to(device)
            batch_y = torch.tensor([batch[i]['class_id'] for i in range(len(batch))]).to(device).unsqueeze(1)
            gt_label = torch.zeros(batch_y.size(0), class_number, device=device)
            gt_label.scatter_(1, batch_y, 1)

            # Prepare dummy data and label. Set the corresponding optimizer.
            dummy_data = torch.rand_like(batch_x, requires_grad=True)
            if use_gt_labels:
                dummy_label = gt_label
                if optim_backend == 'lbfgs':
                    optimizer = torch.optim.LBFGS([dummy_data])
                elif optim_backend == 'adam':
                    optimizer = torch.optim.Adam([dummy_data])
                else:
                    raise ValueError(f'Unrecognized optimizer: {optim_backend}.')
            else:
                dummy_label = torch.randn(batch_y.size(0), class_number, device=device, requires_grad=True)
                if optim_backend == 'lbfgs':
                    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
                elif optim_backend == 'adam':
                    optimizer = torch.optim.Adam([dummy_data, dummy_label])
                else:
                    raise ValueError(f'Unrecognized optimizer: {optim_backend}.')

            # Calculate the ground truth gradient.
            gt_pred = model(batch_x)
            gt_onehot_label = torch.softmax(gt_label, dim=-1)
            gt_loss = torch.mean(torch.sum(-gt_onehot_label * F.log_softmax(gt_pred, dim=-1), 1))
            gt_dy_dx = torch.autograd.grad(gt_loss, visible_parameters, create_graph=True, retain_graph=True)

            # This is for saving reconstructed images.
            batch_img_history = {self.base_index + start + i: [] for i in range(dummy_data.shape[0])}

            # Save the min reconstruction loss and the corresponding images.
            best_losses = torch.zeros(dummy_data.shape[0], dtype=torch.float).to(device)
            torch.fill_(best_losses, -1e6)
            best_images = torch.zeros_like(batch_x).to('cpu')

            # Optimize the reconstructed images iteratively.
            for iters in tqdm(range(self.iteration)):

                def closure():
                    optimizer.zero_grad()
                    # Calculate gradients using dummy data and label.
                    dummy_pred = model(dummy_data)
                    dummy_onehot_label = torch.softmax(dummy_label, dim=-1)
                    dummy_loss = torch.mean(torch.sum(-dummy_onehot_label * F.log_softmax(dummy_pred, dim=-1), 1))
                    dummy_dy_dx = torch.autograd.grad(
                        dummy_loss, visible_parameters, create_graph=True, retain_graph=True
                    )

                    # Main loss function is the difference between gradients calculated by dummy data and real data.
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, gt_dy_dx):
                        grad_diff += self.metric(gx, gy)
                    # Calculate tv-loss.
                    tv_loss = tv_criterion(dummy_data)

                    # Final loss is the combination of tv-loss and grad-diff.
                    total_loss = grad_diff + self.tv_weight * tv_loss
                    total_loss.backward(retain_graph=True)

                    return total_loss

                optimizer.step(closure)
                dummy_data.clamp(min=0, max=1)

                # Calculate the reconstruction loss and save the best loss of all iterations.
                # Save the corresponding images.
                reconstruction_losses = _reconstruction_psnr(dummy_data, batch_x)
                best_losses = torch.maximum(best_losses, reconstruction_losses)
                best_images[best_losses == reconstruction_losses] = \
                    dummy_data[best_losses == reconstruction_losses].detach().cpu()

                # Save the reconstructed data.
                if iters % self.iteration_per_save == 0:
                    for i in range(dummy_data.shape[0]):
                        img_idx = self.base_index + start + i
                        recovered_image = dummy_data[i].detach().cpu().numpy()
                        recovered_image = (
                            255 * np.concatenate([recovered_image, batch_x[i].detach().cpu().numpy()], axis=2)
                        ).astype('uint8')
                        recovered_image = np.swapaxes(recovered_image, 0, 1)
                        recovered_image = np.swapaxes(recovered_image, 1, 2)
                        batch_img_history[img_idx].append(recovered_image)

            # Record final reconstruction loss for this batch.
            total_loss.append(torch.mean(_reconstruction_psnr(dummy_data, batch_x)).item())
            total_min_loss.append(torch.mean(best_losses).item())
            save_dict = {'last_psnr': total_loss[-1], 'max_psnr': total_min_loss[-1]}
            self.logger.add_scalars_dict('reconstruction', save_dict, rnd=math.ceil(start // batch_size))

            # Save the images in .gif format.
            if save_img:
                import imageio
                for idx, (img_idx, images) in enumerate(batch_img_history.items()):
                    imageio.mimsave(os.path.join(self.working_dir, f'{img_idx}.gif'), images, duration=0.25)
                    # Save the best image in .png format.
                    recovered_image = (
                        255 * np.concatenate([best_images[idx].numpy(), batch_x[idx].detach().cpu().numpy()], axis=2)
                    ).astype('uint8')
                    recovered_image = np.swapaxes(recovered_image, 0, 1)
                    recovered_image = np.swapaxes(recovered_image, 1, 2)
                    imageio.imwrite(os.path.join(self.working_dir, f'{img_idx}.png'), recovered_image)

        self.base_index += len(dataset)
        final_loss = sum(total_loss) / len(total_loss)
        min_loss = sum(total_min_loss) / len(total_min_loss)
        self.logger.logging(f'Final reconstruction PSNR: {final_loss}.\t Max reconstruction PSNR: {min_loss}.')
        return final_loss, min_loss
