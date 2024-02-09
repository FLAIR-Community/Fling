from torchvision.models import resnet18
from fling.utils.visualize_utils import ActivationMaximizer


if __name__ == '__main__':
    model = resnet18(pretrained=True)
    activation_maximizer = ActivationMaximizer(iteration=1000, working_dir='./resnet18_am', tv_weight=10)
    activation_maximizer.activation_maximization(model, "fc", channel_id=0,
                                                 image_shape=[3, 224, 224], device='cuda', learning_rate=1e-1)
