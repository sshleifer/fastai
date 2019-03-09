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


def require_nonleaf_grad(v):
    def hook(g):
        v.grad_nonleaf = g

    v.register_hook(hook)

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
        '''We want this code to work, and to allow effect gradient computations of weights indirectly.
        We do not need attack connected to the graph after.
        '''

        #with torch.no_grad(): # what is predecessor?

        with no_grad_context():
            l_x, _, __ = model(x)
            # 1) this context manager doesnt do much idt
            # 2) does adversarial_text use logits?
            pred = F.softmax(l_x, dim=1)#.detach()
        pred = pred.detach()
        l_x = l_x.detach()

        # prepare random unit tensor
        rnn = model[0]
        emb_shape = (32, rnn.emb_size)

        emb = model[0].encoder_with_dropout(x, dropout=rnn.dropoute if model[0].training else 0)
        emb = model[0].dropouti(emb)#.detach()
        attack = V_(to_gpu(torch.rand(emb_shape).sub(0.5)), requires_grad=True,
                    volatile=False)
        attack = _l2_normalize(attack)
        assert not attack.volatile, 'attack volatile before power iteration'

        with _disable_tracking_bn_stats(model):
            with set_grad_enabled(model.training):
            # calc adversarial direction
                for i in range(self.ip):
                    #attack.requires_grad_(True)
                    #attack = attack * self.xi
                    print(f'attack.requires_grad: {attack.requires_grad}')
                    logp_hat = self.seq_rnn_emb2logits(model, emb, attack)
                    assert not attack.volatile, 'attack volatile before adv_dist.backward()'
                    adv_distance = F.kl_div(logp_hat, pred,)  # EOS Weights?
                    attack.retain_grad()  # needed to make attack.grad not None
                    assert not attack.volatile, 'attack volatile before adv_dist.backward(), after retain_grad'
                    adv_distance.backward() # does this change attack?

                    print('grad: attack.grad')
                    #print
                    assert attack.grad is not None
                    assert not attack.volatile, 'attack volatile after adv_dist.backward()'
                    attack = _l2_normalize(attack.grad)  # breaks cause grad is None
                    # nans in attack?
                    model.zero_grad()
                    #attack.data.grad.zero_()

            # calc LDS
            # attack.zero_grad()

            if attack.volatile:
                attack = attack.detach()
                attack = V_(attack.data, requires_grad=False)
            assert not attack.volatile
            r_adv = attack * self.eps
            logp_hat = self.seq_rnn_emb2logits(model, emb, r_adv)
            lds = F.kl_div(logp_hat, pred)
        assert not lds.volatile
        return lds

    def seq_rnn_emb2logits(self, model, emb, attack):
        rnn_out = model[0].forward_from_embedding(emb + attack)
        pred_hat, _, __ = model[1].forward(rnn_out)
        return F.log_softmax(pred_hat, dim=1)
