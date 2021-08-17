import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from utils import *
import numpy as np
import os
import torch.nn.functional as F

class GBN_model(nn.Module):
    def __init__(self, args):
        super(GBN_model, self).__init__()
        self.args = args
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(10.0).float()

        self.wei_shape = torch.tensor(1e-1).float()

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.topic_size = args.topic_size
        self.topic_size = [self.vocab_size] + self.topic_size
        self.layer_num = len(self.topic_size) - 1
        self.embed_size = args.embed_size

        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(self.hidden_size[i]) for i in range(self.layer_num)])

        h_encoder = [DeepConv1D(self.hidden_size[0], 1, self.vocab_size)]
        for i in range(self.layer_num - 1):
            h_encoder.append(ResConv1D(self.hidden_size[i + 1], 1, self.hidden_size[i]))
            #h_encoder.append(Conv1D(self.hidden_size[i+1], 1, self.vocab_size))

        self.h_encoder = nn.ModuleList(h_encoder)

        shape_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size[i]) for i in
                         range(self.layer_num - 1)]
        shape_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size[self.layer_num - 1]))
        self.shape_encoder = nn.ModuleList(shape_encoder)

        scale_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size[i]) for i in
                         range(self.layer_num - 1)]
        scale_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size[self.layer_num - 1]))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        decoder = [Conv1DSoftmaxEtm(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in
                   range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)

        for t in range(self.layer_num - 1):
            self.decoder[t + 1].rho = self.decoder[t].alphas

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.to(self.args.device)))

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num = 50):
        # sample one
        eps = torch.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1).to(self.args.device)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-self.log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        return torch.mean(theta, dim=0, keepdim=False)

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))
        return - likelihood / (x.shape[1])

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.to(self.args.device) * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.to(self.args.device) + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / (Wei_scale.shape[1])

    def _ppl(self, x, theta):
        # x: K1 * N
        X1 = self.decoder[0](theta, 0)  # V * N
        X2 = X1 / (X1.sum(0) + real_min)
        ppl = x * torch.log(X2.T + real_min) / -x.sum()
        # ppl = tf.reduce_sum(x * tf.math.log(X2 + real_min)) / tf.reduce_sum(x)
        return ppl.sum().exp()

    def test_ppl(self, x, y):
        _, theta, _, _ = self.forward(x)

        # _, theta_y, _, _ = self.forward_heart(y)
        ppl = self._ppl(y, theta[0])
        # ret_dict.update({"ppl": ppl})
        return ppl

    def forward(self, x):

        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        gam_scale = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num
        loss = [0] * (self.layer_num + 1)
        likelihood = [0] * (self.layer_num + 1)

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t-1])))

            hidden_list[t] = hidden

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_list[t])),
                                       self.real_min.to(self.args.device))      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.to(self.args.device))

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_list[t])), self.real_min.to(self.args.device))

                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                phi_theta[t] = self.decoder[t](theta[t], t)

            else:
                temp = phi_theta[t+1].permute(1, 0)
                hidden_phitheta = torch.cat((hidden_list[t], temp), 1)

                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)),
                                       self.real_min.to(self.args.device))  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.to(self.args.device))

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.to(self.args.device))
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                phi_theta[t] = self.decoder[t](theta[t], t)

        for t in range(self.layer_num + 1):
            if t == 0:
                loss[t] = self.compute_loss(x.permute(1, 0), phi_theta[t])
                likelihood[t] = loss[t]

            elif t == self.layer_num:
                loss[t] = self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).to(self.args.device), torch.tensor(1.0, dtype=torch.float32).to(self.args.device),
                                             k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
                likelihood[t] = loss[t]

            else:
                loss[t] = self.KL_GamWei(phi_theta[t], torch.tensor(1.0, dtype=torch.float32).to(self.args.device),
                                         k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
                likelihood[t] = self.compute_loss(theta[t - 1], phi_theta[t])

        return phi_theta, theta, loss, likelihood
