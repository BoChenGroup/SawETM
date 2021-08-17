import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from model import *
import matplotlib.pyplot as plt
import pickle

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)

        self.model = GBN_model(args)
        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data_loader):

        for epoch in range(self.epochs):

            for t in range(self.layer_num - 1):
                self.model.decoder[t + 1].rho = self.model.decoder[t].alphas

            self.model.to(self.args.device)

            loss_t = [0] * (self.layer_num + 1)
            likelihood_t = [0] * (self.layer_num + 1)
            num_data = len(train_data_loader)

            for i, (train_data, _) in enumerate(train_data_loader):
                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                train_data = torch.tensor(train_data, dtype=torch.float).to(self.args.device)
                # train_label = torch.tensor(train_label, dtype=torch.long).cuda()

                re_x, theta, loss_list, likelihood = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                    elif t < self.layer_num:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                        (0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                re_x, theta, loss_list, likelihood = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                    elif t < self.layer_num:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                        (0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()

            if epoch % 1 == 0:
                for t in range(self.layer_num + 1):
                    print('epoch {}|{}, layer {}|{}, loss: {}, likelihood: {}, lb: {}'.format(epoch, self.epochs, t,
                                                                                              self.layer_num,
                                                                                              loss_t[t]/2,
                                                                                              (likelihood_t[t]*0.5),
                                                                                              loss_t[t]/2))
                

            self.model.eval()

            if epoch % 10 == 0:
                test_likelihood, test_ppl = self.test(train_data_loader)
                save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch}, self.save_path, True)
                print('epoch {}|{}, test_ikelihood,{}, ppl: {}'.format(epoch, self.epochs, test_likelihood, test_ppl))


    def test(self, data_loader):
        self.model.eval()

        likelihood_t = 0
        num_data = len(data_loader)
        ppl_total = 0

        for i, (train_data, test_data) in enumerate(data_loader):
            train_data = torch.tensor(train_data, dtype = torch.float).to(self.args.device)
            test_data = torch.tensor(test_data, dtype=torch.float).to(self.args.device)
            # test_label = torch.tensor(test_label, dtype=torch.long).cuda()

            with torch.no_grad():
                ppl = self.model.test_ppl(train_data, test_data)
                # likelihood_total += ret_dict["likelihood"][0].item() / num_data
                ppl_total += ppl.item() / num_data

            # re_x, theta, loss_list, likelihood = self.model(test_data)
            # likelihood_t += likelihood[0].item() / num_data

        # save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch}, self.save_path, True)

        return likelihood_t, ppl_total


    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    def vis(self):
        # layer1
        w_1 = torch.mm(self.GBN_models[0].decoder.rho, torch.transpose(self.GBN_models[0].decoder.alphas, 0, 1))
        phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

        index1 = range(100)
        dic1 = phi_1[:, index1[0:49]]
        # dic1 = phi_1[:, :]
        fig7 = plt.figure(figsize=(10, 10))
        for i in range(dic1.shape[1]):
            tmp = dic1[:, i].reshape(28, 28)
            ax = fig7.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index1[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_2 = torch.mm(self.GBN_models[1].decoder.rho, torch.transpose(self.GBN_models[1].decoder.alphas, 0, 1))
        phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()
        index2 = range(49)
        dic2 = np.matmul(phi_1, phi_2[:, index2[0:49]])
        #dic2 = np.matmul(phi_1, phi_2[:, :])
        fig8 = plt.figure(figsize=(10, 10))
        for i in range(dic2.shape[1]):
            tmp = dic2[:, i].reshape(28, 28)
            ax = fig8.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index2[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_3 = torch.mm(self.GBN_models[2].decoder.rho, torch.transpose(self.GBN_models[2].decoder.alphas, 0, 1))
        phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()
        index3 = range(32)

        dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, index3[0:32]])
        #dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, :])

        fig9 = plt.figure(figsize=(10, 10))
        for i in range(dic3.shape[1]):
            tmp = dic3[:, i].reshape(28, 28)
            ax = fig9.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index3[i] + 1))
            ax.imshow(tmp)

        plt.show()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self, outpath='phi_output'):

        phi = []
        for t in range(self.layer_num):
            w_t = torch.mm(self.model.decoder[t].rho, torch.transpose(self.model.decoder[t].alphas, 0, 1))
            phi_t = torch.softmax(w_t, dim=0).cpu().detach().numpy()
            phi.append(phi_t)

        self.vision_phi(phi, outpath=outpath)