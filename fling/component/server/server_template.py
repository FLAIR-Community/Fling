class ServerTemplate:

    def __init__(self, args, test_dataset):
        self.args = args
        self.glob_dict = None

        device = args.learn.device
        self.device = device

    def apply_grad(self, grad, lr=1.):
        state_dict = self.glob_dict
        for k in grad:
            state_dict[k] = state_dict[k] + lr * grad[k]

    def test_step(self, model, batch_data, criterion, monitor):
        raise NotImplementedError

    def preprocess_data(self, data):
        raise NotImplementedError

    def test(self, model, test_loader=None):
        raise NotImplementedError
