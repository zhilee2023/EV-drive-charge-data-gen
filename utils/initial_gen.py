import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily
import torch
import torch.utils
import torch.utils.data
import numpy as np
import json

class Initial_Gen(nn.Module):
    def __init__(self, n_loc, h_dim, embed_vector_len, latent_dim=6, device='cpu'):
        super(Initial_Gen, self).__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(n_loc, embed_vector_len)
        self.input_dim = embed_vector_len + 5  # 更新输入维度：嵌入向量长度加上其他四个变量
        self.soc_comp = 6  # 组件数用于SOC的GMM
        self.mon_comp = 12
        self.fc1 = nn.Linear(self.input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim//2)
        self.fc3 = nn.Linear(h_dim//2, 2 * latent_dim)  # 输出均值和对数方差
        self.fc4 = nn.Linear(latent_dim, h_dim//2)
        self.fc5 = nn.Linear(h_dim//2, h_dim)
        self.loc_decoder = nn.Linear(h_dim, n_loc)
        self.soc_decoder = nn.Linear(h_dim, self.soc_comp*3)
        self.month_decoder = nn.Linear(h_dim, self.mon_comp*3)
        self.user_decoder = nn.Linear(h_dim, 2)
        self.battery_decoder = nn.Linear(h_dim, 5)
        self.weekday_decoder = nn.Linear(h_dim, 7)
        self.decoders = [self.loc_decoder, self.soc_decoder,self.user_decoder, self.battery_decoder, self.weekday_decoder,self.month_decoder]
        self.discrete_loss = nn.CrossEntropyLoss()
        self.device=device
        file_path="battery_diction.json"
        with open(file_path, 'r') as file:
            diction=json.load(file)
        self.diction=diction
        self.to(device)

    def encode(self, x):
        loc_emb = self.embed(x[:, 0].long())  # 假设位置ID在第一列
        x = torch.cat([loc_emb, x[:, 1:]], dim=1)
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        return self.fc3(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.leaky_relu(self.fc4(z))
        h4 = F.leaky_relu(self.fc5(h3))
        outputs= [self.decoders[i](h4) for i in range(len(self.decoders))]  # 除了SOC以外的都取argmax
        return outputs

    def forward(self, x):
        output = self.encode(x)
        mu, logvar = output[:, :self.latent_dim], output[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        outputs = self.decode(z)
        loss = self.compute_loss(outputs, x, mu, logvar)
        return loss
    
    def compute_gmm(self,output,ncomp):
        mus = output[:, :ncomp]
        sigs = torch.exp(output[:, ncomp:2 * ncomp]*0.5)
        weights = F.softmax(output[:, 2 * ncomp:], dim=1)
        mix = Categorical(weights)
        comp = Normal(mus, sigs)
        gmm = MixtureSameFamily(mix, comp)
        return gmm
    
    def compute_loss(self, output, target, mu, logvar):
        # 位置、用户、电池、星期几的离散损失
        #print(output[0].shape,target[:,0])
        loss_discrete = torch.mean(torch.stack([self.discrete_loss(output[i], target[:, i].long()) for i in [0,2,3,4]]))
        # SOC的高斯混合模型损失
        soc_model=self.compute_gmm(output[1],self.soc_comp)
        loss_soc = -soc_model.log_prob(target[:, 1].float()).mean() 

        month_model=self.compute_gmm(output[-1],self.mon_comp)
        loss_month = -month_model.log_prob(target[:, -1].float()).mean()  # 假设SOC目标在第五列
        # KL散度
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return loss_discrete + loss_soc + kld+loss_month
    
    def generate_samples(self,batch_size):
        """
        生成给定数量的样本。
        
        参数:
        model: 训练好的VAE模型。
        num_samples: 要生成的样本数量。
        device: 设备类型，'cuda'或'cpu'。

        返回:
        生成的样本。
        """
        with torch.no_grad():  # 关闭梯度计算
            # 从标准正态分布中采样潜在变量
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            # 解码得到输出
            outputs = self.decode(z)
            #print(torch.argmax(outputs[0], dim=1))

            samples = []
            for i in range(len(self.decoders)):  # 只对离散型输出使用 Categorical 抽样
                if i==1:
                    soc_model=self.compute_gmm(outputs[i],self.soc_comp)
                    samples.append(torch.clamp(soc_model.sample().unsqueeze(1),0,1))
                elif i==5:
                    mon_model=self.compute_gmm(outputs[i],self.mon_comp)
                    samples.append(torch.clamp(mon_model.sample().unsqueeze(1),0,1))
                else:
                    probabilities = F.softmax(outputs[i], dim=1)
                    cat_distribution =torch.distributions.Categorical(probabilities)
                    samples.append(cat_distribution.sample().unsqueeze(1))
            samples=torch.cat(samples,dim=1)
            inverted_dict = {v: round(float(k), 1) for k, v in self.diction.items()}
            new_col=torch.from_numpy(np.array([inverted_dict[i] for i in samples[:,3].cpu().numpy()])).to(self.device)
            samples[:,3]=new_col
        return samples