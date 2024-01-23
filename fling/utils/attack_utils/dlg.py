import os

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

from fling.utils import Logger


class DLGAttacker:
    """
    Overview:
        The attack method introduced in: Deep Leakage from Gradients. Given the gradients of input images, recover the
    original input images.
    """

    def __init__(self, iteration: int, working_dir: str, iteration_per_save: int = 10):
        """
        Overview:
            Initialize the attacker object.
        Arguments:
            iteration: The number of iterations for recovering the input image.
            working_dir: The working dir of this attacker. The attack results will be saved to this path.
            iteration_per_save: The interation interval between saving two reconstructed images. Default to be 10.
        """
        self.iteration = iteration
        self.working_dir = working_dir
        self.logger = Logger(self.working_dir)
        self.base_index = 0
        self.iteration_per_save = iteration_per_save

    def attack(
            self,
            model: nn.Module,
            dataset: Dataset,
            device: str,
            class_number: int,
            batch_size: bool = 1,
            use_gt_labels: bool = True,
            save_img: bool = False
    ) -> float:
        """
        Overview:
            The main attack function.
        Arguments:
            model: The model to be attacked.
            dataset: The dataset tried to be recovered by the attacker.
            batch_size: Batch-size of the input images. A larger batch size will be harder to attack. Default to be 1.
            device: Device to run this attack algorithm, such as ``"cpu"`` and ``"cuda"``.
            class_number: The number of classes for the classification task.
            use_gt_labels: Whether the ground truth label is able to be acquired by the attacker. Default to be \
                ``True``. If set to be ``False``, it will be harder to attack.
            save_img: Whether to save the reconstructed images into working dir. Default to be ``False``.
        Returns:
            mean_loss: The mean reconstruction loss across all samples in the dataset. A smaller value indicates a \
                better performance for the attacker.
        """
        start, end = 0, batch_size
        batch_idx = 0
        total_loss = []

        model = model.to(device=device)
        model = model.eval()

        # Iterate for each batch in the dataset.
        while end <= len(dataset):
            batch = dataset[start:end]
            start = end
            end = min(len(dataset), end + batch_size)
            self.logger.logging(f'Star batch: {batch_idx}...')
            batch_idx += 1

            # Prepare input data and labels.
            batch_x = torch.stack([batch[i][0] for i in range(len(batch))]).to(device)
            batch_y = torch.tensor([batch[i][1] for i in range(len(batch))]).to(device).unsqueeze(1)
            gt_label = torch.zeros(batch_y.size(0), class_number, device=device)
            gt_label.scatter_(1, batch_y, 1)

            # Prepare dummy data and label. Set the corresponding optimizer.
            dummy_data = torch.rand_like(batch_x, requires_grad=True)
            if use_gt_labels:
                dummy_label = gt_label
                optimizer = torch.optim.LBFGS([dummy_data])
            else:
                dummy_label = torch.randn(batch_y.size(0), class_number, device=device, requires_grad=True)
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

            # Calculate the ground truth gradient.
            gt_pred = model(batch_x)
            gt_onehot_label = torch.softmax(gt_label, dim=-1)
            gt_loss = torch.mean(torch.sum(-gt_onehot_label * F.log_softmax(gt_pred, dim=-1), 1))
            gt_dy_dx = torch.autograd.grad(gt_loss, model.parameters(), create_graph=True, retain_graph=True)

            # This is for saving reconstructed images.
            batch_img_history = {self.base_index + start + i: [] for i in range(dummy_data.shape[0])}

            # Optimize the reconstructed images iteratively.
            for iters in tqdm(range(self.iteration)):

                def closure():
                    optimizer.zero_grad()
                    # Calculate gradients using dummy data and label.
                    dummy_pred = model(dummy_data)
                    dummy_onehot_label = torch.softmax(dummy_label, dim=-1)
                    dummy_loss = torch.mean(torch.sum(-dummy_onehot_label * F.log_softmax(dummy_pred, dim=-1), 1))
                    dummy_dy_dx = torch.autograd.grad(
                        dummy_loss, model.parameters(), create_graph=True, retain_graph=True
                    )

                    # Final loss function is the difference between gradients calculated by dummy data and real data.
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, gt_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward(retain_graph=True)

                    return grad_diff

                optimizer.step(closure)

                # Save the reconstructed data.
                if iters % self.iteration_per_save == 0:
                    for i in range(dummy_data.shape[0]):
                        img_idx = self.base_index + start + i
                        recovered_image = 255 * dummy_data[i].detach().cpu().numpy().astype('uint8')
                        recovered_image = np.swapaxes(recovered_image, 0, 1)
                        recovered_image = np.swapaxes(recovered_image, 1, 2)
                        batch_img_history[img_idx].append(recovered_image)

            # Record reconstruction loss for this batch.
            total_loss.append(torch.sqrt(torch.sum(dummy_data - batch_x) ** 2).item())
            self.logger.logging(f'Mean reconstruction loss: {total_loss[-1]}.')

            # Save the images in .gif format.
            if save_img:
                import imageio
                for img_idx, images in batch_img_history.items():
                    imageio.mimsave(os.path.join(self.working_dir, f'{img_idx}.gif'), images, duration=0.25)

        self.base_index += len(dataset)
        mean_loss = sum(total_loss) / len(total_loss)
        self.logger.logging(f'Mean reconstruction loss: {mean_loss}.')
        return mean_loss
