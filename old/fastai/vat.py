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
    #d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    nrm = torch.norm(d, p=2, dim=1) + 1e-8
    return d.div(nrm.view(nrm.shape[0], 1).expand_as(d) + 1e-8)


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
            pred = F.softmax(l_x, dim=1).detach()

        # prepare random unit tensor
        rnn = model[0]
        emb_shape = (32, rnn.emb_size)

        emb = model[0].encoder_with_dropout(x, dropout=rnn.dropoute if model[0].training else 0)
        emb = model[0].dropouti(emb).detach()
        print(f'emb: {emb.shape}')


        attack = V_(to_gpu(torch.rand(emb_shape).sub(0.5)), requires_grad=True)

        attack = _l2_normalize(attack)
        attack.requires_grad_(True)
        start = attack[0][0]
        #print(f'attack[o]: {start}')

        with _disable_tracking_bn_stats(model):
            with set_grad_enabled(model.training):
            # calc adversarial direction
                for _ in range(self.ip):
                    #attac

                    logp_hat = self.seq_rnn_emb2logits(model, emb, attack * self.xi)

                    adv_distance = F.kl_div(logp_hat, pred, #detach(),
                                            # reduction='batchmean'
                                            )
                    #assert attack.grad is not None, '1. dgrad None'
                    adv_distance.backward() # does this change attack?
                    assert attack[0][0] != start
                    #assert attack.grad is not None, '2. dgrad Nonep'
                    attack = _l2_normalize(attack.grad)
                    model.zero_grad()

            # calc LDS
            r_adv = attack * self.eps
            logp_hat = self.seq_rnn_emb2logits(model, emb, r_adv)
            lds = F.kl_div(logp_hat, pred)

        return lds

    def seq_rnn_emb2logits(self, model, emb, attack):
        rnn_out = model[0].forward_from_embedding(emb + attack)
        pred_hat, _, __ = model[1].forward(rnn_out)
        return F.log_softmax(pred_hat, dim=1)
