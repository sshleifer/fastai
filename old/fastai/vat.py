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
    # TODO(SS): they use a stabler l2_norm in the adversarial text code
    nrm = torch.norm(d, p=2, dim=1) + 1e-8
    return d.div(nrm.view(nrm.shape[0], 1, nrm.shape[1]).expand_as(d))

def require_nonleaf_grad(v):
    """Unused"""
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
        sl, bs = x.shape
        # 2) does adversarial_text use logits? Yes but they call sigmoid inside their KL helper
        original_logits, _, __ = model(x)
        original_logits = original_logits.detach()


        # prepare random unit tensor
        rnn = model[0]
        #emb_shape = (bs, rnn.emb_size)

        embedded = model[0].encoder_with_dropout(x, dropout=rnn.dropoute if model[0].training else 0)
        embedded = model[0].dropouti(embedded).detach() # 3d [sl, bs, esize] (I think)
        attack = V_(to_gpu(torch.rand(embedded.shape).sub(0.5)), requires_grad=True, volatile=False)
        attack = _l2_normalize(attack)
        assert not attack.volatile, 'attack volatile before power iteration'

        with _disable_tracking_bn_stats(model):
            with set_grad_enabled(model.training):
            # calc adversarial direction
                for i in range(self.ip):
                    #attack.requires_grad_(True)
                    #attack = attack * self.xi

                    assert attack.requires_grad
                    logp_hat = self.seq_rnn_emb2logits(model, embedded, attack)
                    mask = torch.zeros_like(logp_hat)
                    mask[-1] = 1
                    raise ValueError(f'mask shape: {mask.shape}')
                    logp_hat *= mask
                    assert not attack.volatile, 'attack volatile before adv_dist.backward()'
                    adv_distance = F.kl_div(logp_hat, original_logits,)  # EOS Weights?
                    assert not attack.volatile, 'attack volatile before adv_dist.backward(), after retain_grad'
                    # the backpropagation algorithm should not be used to propagate
                    # gradients through the adversarial example construction process.
                    attack_grad, = torch.autograd.grad(adv_distance, attack)
                    attack = attack_grad
                    model.zero_grad()


            if attack.volatile:
                attack = attack.detach()
                attack = V_(attack.data, requires_grad=False)
            assert not attack.volatile
            r_adv = attack * self.eps
            logp_hat = self.seq_rnn_emb2logits(model, embedded, r_adv)
            lds = F.kl_div(logp_hat, original_logits)
        assert not lds.volatile
        return lds

    def seq_rnn_emb2logits(self, model, emb, attack):
        rnn_out: tuple = model[0].forward_from_embedding(emb + attack)
        #print(f'rnn_out shape: {rnn_out.shape}')
        import pdb; pdb.set_trace()
        raise ValueError(f'rnn_out shape: {rnn_out[0].shape, rnn_out[1].shape}')
        pred_hat, _, __ = model[1].forward(rnn_out)
        return pred_hat
        # return F.log_softmax(pred_hat, dim=1)
