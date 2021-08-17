import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

real_min = torch.tensor(1e-30)

def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    eulergamma = torch.tensor(0.5772, dtype=torch.float32)

    part1 = eulergamma * (1 - 1 / Wei_shape) + log_max(
        Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)

    part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma / Wei_shape)

    part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))

    KL = part1 + part2 + part3
    return KL


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class DeepConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(DeepConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf)
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf))

            w2 = torch.empty(nf, nf)
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf))

        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            rx = x
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = x + rx
        else:
            raise NotImplementedError
        return x


class ResConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(ResConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf)
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf))

            w2 = torch.empty(nx, nf)
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            rx = x
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = rx + x
        else:
            raise NotImplementedError
        return x


class Conv1DSoftmax(nn.Module):
    def __init__(self, voc_size, topic_size):
        super(Conv1DSoftmax, self).__init__()

        w = torch.empty(voc_size, topic_size)
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)

    def forward(self, x):
        w = torch.softmax(self.w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

class Conv1DSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, last_layer=None):
        super(Conv1DSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        if last_layer is None:
            w1 = torch.empty(self.voc_size, self.emb_size)
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)
        else:
            w1 = torch.empty(self.voc_size, self.emb_size)
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)

        w2 = torch.empty(self.topic_size, self.emb_size)
        nn.init.normal_(w2, std=0.02)
        self.alphas = Parameter(w2)

    def forward(self, x, t):
        if t == 0:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        else:
            w = torch.mm(self.rho.detach(), torch.transpose(self.alphas, 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x


def variable_para(shape, device='cuda'):
    w = torch.empty(shape, device=device)
    nn.init.normal_(w, std=0.02)
    return torch.tensor(w, requires_grad=True)


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


