import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import no_grad_context, set_grad_enabled
from .text import *


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):

        #with torch.no_grad(): # what is predecessor?
        l_x, raw_outputs, outputs = model(x)
        with no_grad_context():
            pred = F.softmax(l_x, dim=1)

        # prepare random unit tensor
        d = to_gpu(torch.rand(x.shape).sub(0.5))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            with set_grad_enabled():
            # calc adversarial direction
                for _ in range(self.ip):
                    pred_hat, raw_out, out = model(x + self.xi * d)
                    logp_hat = F.log_softmax(pred_hat, dim=1)
                    adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                    adv_distance.backward()
                    d = _l2_normalize(d.grad)
                    model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
