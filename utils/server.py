import copy
from utils.client import get_loss
import torch


class Server:

    def __init__(self, args, device, test_loader):
        self.args = args
        self.attrs = {}

        self.device = device if device >= 0 else 'cpu'
        self.test_loader = test_loader

    def add_attr(self, name, item):
        self.attrs[name] = copy.deepcopy(item)

    def __getitem__(self, item):
        return self.attrs[item]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def apply_grad(self, grad, lr=1.):
        state_dict = self.attrs['glob_dict']
        for k in grad:
            state_dict[k] = state_dict[k] + lr * grad[k]

    def check_bias(self, client_pool, save_ram=False, max_client=float('inf')):
        # used for checking difference for features calculated on different clients.
        # return a diction consisting: abs_value, cos_distance, l2_distance.
        # All returned keys will be later added to tb. Only support model having: compute_feature.
        model_list = [copy.deepcopy(client.model) for client in client_pool]
        max_len = min(max_client, len(model_list))
        model_list = model_list[:max_len]

        # to device
        for model in model_list:
            model.eval()
            if not save_ram:
                model.to(self.device)

        abs_vals = []
        l2_dists = []
        cos_dists = []

        for _, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_feature = []
            for model in model_list:
                if save_ram:
                    model = model.to(self.device)
                    batch_feature.append(model.compute_feature(batch_x).detach())
                    model = model.to('cpu')
                else:
                    batch_feature.append(model.compute_feature(batch_x).detach())
            abs_val = torch.cat([torch.abs(batch_feature[ii]) for ii in range(len(batch_feature))], dim=0)
            l2_dist = torch.cat(
                [
                    torch.norm(batch_feature[i] - batch_feature[i + 1], dim=-1, keepdim=False)
                    for i in range(len(batch_feature) - 1)
                ],
                dim=0
            )
            batch_feature = torch.nn.functional.normalize(torch.stack(batch_feature, dim=0), dim=-1)
            cos_dist = torch.cat(
                [torch.diag(batch_feature[i] @ batch_feature[i + 1].T) for i in range(len(batch_feature) - 1)]
            )

            abs_vals.append(abs_val.cpu())
            l2_dists.append(l2_dist.cpu())
            cos_dists.append(cos_dist.cpu())
        abs_vals = torch.cat(abs_vals, dim=0)
        l2_dists = torch.cat(l2_dists, dim=0)
        cos_dists = torch.cat(cos_dists, dim=0)

        res_dict = {}
        res_dict['abs_mean'] = torch.mean(abs_vals).item()
        res_dict['abs_std'] = torch.std(abs_vals).item()
        res_dict['abs_max'] = torch.max(abs_vals).item()

        res_dict['l2_mean'] = torch.mean(l2_dists).item()
        res_dict['l2_std'] = torch.std(l2_dists).item()
        res_dict['l2_max'] = torch.max(l2_dists).item()

        res_dict['cos_mean'] = torch.mean(cos_dists).item()
        res_dict['cos_std'] = torch.std(cos_dists).item()
        res_dict['cos_min'] = torch.min(cos_dists).item()

        if not save_ram:
            for model in model_list:
                model.to('cpu')
        return res_dict

    def test(self, model, loss, test_loader=None):
        if test_loader:
            old_loader = self.test_loader
            self.test_loader = test_loader
        model = copy.deepcopy(model)

        correct = 0
        total = 0
        tot_loss = 0
        model.eval()
        model.to(self.device)

        criterion = get_loss(loss)

        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if 'dataloader_type' in self.args.__dict__.keys() and self.args.dataloader_type == 'nlp':
                o = model(batch_x, batch_y)
                loss = criterion(o, batch_y)
                tot_loss += loss.item()
                _, y_pred = o[0][0].data.max(1, keepdim=True)
                correct += 1
                total += batch_y.shape[0]
            else:
                o = model(batch_x)
                tot_loss += criterion(o, batch_y).item()
                y_pred = o.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(batch_y.data.view_as(y_pred)).long().sum().item()
                total += batch_y.shape[0]
        model.to('cpu')

        avg_acc = correct / total
        avg_loss = tot_loss / total

        if test_loader:
            self.test_loader = old_loader

        res_dict = {'acc': avg_acc, 'loss': avg_loss}
        return res_dict
