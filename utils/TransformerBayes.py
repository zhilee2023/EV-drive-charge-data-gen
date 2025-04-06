import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
from .util import PositionalEncoding,GMM_bayes, Discrete_bayes,update_month
from .initial_gen import Initial_Gen

class TransformerBayes(nn.Module):
    def __init__(self, x_dim, time_step,n_head,n_layers,d_model,n_loc,embed_vector_len,device='cpu',tau=0.01,rnn_backbone=False):
        super(TransformerBayes,self).__init__()
        self.x_dim = x_dim
        self.n_head=n_head
        self.time_step=time_step
        self.n_loc=n_loc
        self.eps=torch.finfo(torch.float).eps
        self.device=device
        self.embed=torch.nn.Embedding(n_loc, embed_vector_len).to(self.device)
        self.embed_len=embed_vector_len
        self.time_step=time_step
        self.x_enclen=x_dim-2+2*embed_vector_len
        self.d_model=d_model
        self.tau=tau
        self.mask=torch.triu(torch.ones((self.time_step,self.time_step))==1,diagonal=1).to(self.device)
        self.rnn_backbone=rnn_backbone

        if self.rnn_backbone==False:
            self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,dim_feedforward=self.d_model*4,batch_first=True,device=self.device)
            self.encoder_mod = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        else:
            self.encoder_rnn = nn.LSTM(input_size=d_model,hidden_size=d_model,num_layers=n_layers,batch_first=True,dropout=0.1).to(self.device)

        self.transform=torch.FloatTensor(np.load("min_scale.npy")).to(self.device)

        self.pos= PositionalEncoding(d_model=self.d_model).to(self.device)
        #self.output_linear=nn.Linear(self.d_model,self.output_len).to(self.device)
        self.local=nn.Linear(self.x_enclen,self.d_model).to(self.device)
        #self.augment_mul=5
        self.decoder_type=Discrete_bayes(D_in=self.d_model,D_out=2,device=self.device,tau=self.tau).to(self.device)
        self.decoder_loc=Discrete_bayes(D_in=self.d_model+1,D_out=self.n_loc,device=self.device,tau=self.tau).to(self.device)
        self.decoder_start_hour=GMM_bayes(D_in=self.d_model+2,n_comp=7,device=self.device,tau=self.tau).to(self.device)
        self.decoder_distance=GMM_bayes(D_in=self.d_model+3, n_comp=3,device=self.device,tau=self.tau).to(self.device)
        self.decoder_duration=GMM_bayes(D_in=self.d_model+4, n_comp=5,device=self.device,tau=self.tau).to(self.device)
        self.decoder_soc_change=GMM_bayes(D_in=self.d_model+5, n_comp=6,device=self.device,tau=self.tau).to(self.device)
        self.decoder_stay=GMM_bayes(D_in=self.d_model+6, n_comp=3,device=self.device,tau=self.tau).to(self.device)
        self.decoder=[self.decoder_type,self.decoder_loc,self.decoder_start_hour,self.decoder_distance,
                      self.decoder_duration, self.decoder_soc_change,self.decoder_stay]
        self.p0=Initial_Gen(n_loc=n_loc, h_dim=64, embed_vector_len=self.embed_len, latent_dim=4,device=self.device)
        self.permutation = {
            'trip_kind': 0,    
            'loc': 1,    
            'start_hour': 2,      
            'distance': 3,      
            'duration': 4,      
            'end_soc': 5,    
            'stay': 6            
            }
    
    def backbone_net(self,x,eval_=False):
        if eval_==False:
            if self.rnn_backbone==False:
                x = self.pos(x)
                x = self.encoder_mod(x, mask=self.mask)
            else:
                x,_=self.encoder_rnn(x)
        else:
            if self.rnn_backbone==False:
                x = self.pos.eval()(x)
                x = self.encoder_mod.eval()(x)
            else:
                x,_=self.encoder_rnn.eval()(x)
        return x
    
    def forward(self, batch_x,init):
        # type, loc, start_hour, distance, duration, end_soc, stay,start_index,start_soc, uer_label, battery_capacity,week_day,month
        p0_loss=self.p0(init)
        y=batch_x.clone().detach()
        seq=torch.cat([batch_x[:,:,:-1].clone(),(batch_x[:, :, -1:] * 12).floor().clone()],axis=-1)
        x=seq.clone().detach()
        cond_x=seq.clone().detach()
        x_init=torch.cat([x[:,:,0:1],self.embed(x[:,:,1].to(torch.int)),x[:,:,2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
        #生成x0 start_index,start_soc,uer_label,battery_capacity,week_day,month基于此数据t0生成t1
        x_0=torch.cat([torch.ones((x_init.shape[0],1,x_init.shape[-1]-5-self.embed_len)).to(self.device)*(-0.1),x_init[:,0,-(5+self.embed_len):].unsqueeze(1)],dim=-1)
        x_init=torch.cat([x_0,x_init[:,0:-1,:]],dim=1)
        ##decoder inputprocess
        x=self.local(x_init)
        x=self.backbone_net(x)
        # x=self.pos(x)
        # x=self.encoder_mod(x,mask=self.mask)
        x=torch.cat([x,cond_x],dim=-1)
        results=list(map(lambda i: self.decoder[i](x,i),range(len(self.decoder)))) 
        distributions,sample=tuple(zip(*results))
        sample=torch.cat(sample,dim=-1)
        cond_loss= self.constraint_loss(sample,cond_x)
        recon_loss=self.recon_loss(distributions,y)
        return p0_loss,recon_loss,cond_loss

    def recon_loss(self, distributions,y): 
        # type, loc, start_hour, distance, duration, soc_change, stay,week_day, uer_label, battery_capacity
        session_loss=-distributions[0].log_prob(y[:,:,0].to(torch.int)).mean()
        loc_loss=-distributions[1].log_prob(y[:,:,1].to(torch.int)).mean()
        start_hour_loss=-distributions[2].log_prob(y[:,:,2].unsqueeze(-1)).mean()
        distance_loss=-distributions[3].log_prob(y[:,:,3].unsqueeze(-1)).mean()
        duration_loss=-distributions[4].log_prob(y[:,:,4].unsqueeze(-1)).mean()
        soc_change_loss=-distributions[5].log_prob(y[:,:,5].unsqueeze(-1)).mean()
        stay_loss=-distributions[6].log_prob(y[:,:,6].unsqueeze(-1)).mean()
        loss=loc_loss+session_loss+soc_change_loss+distance_loss+\
             stay_loss+duration_loss+start_hour_loss
        return loss
    
    
    def constraint_loss(self,sample,cond_x):
        # type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,week_day
        trip_kind=cond_x[:,:,0]
        start_soc=cond_x[:,:,8]
        ##end SOC constraint
        flag=(start_soc-sample[:,:,5])
        soc_out_0=torch.relu((1-trip_kind)*flag)##如果是充电的话那么start_soc-end_soc为正的被惩罚
        soc_out_1=torch.relu(-trip_kind*flag)##如果是驾驶的话那么start_soc-end_soc为负的被惩罚
        loss_end_soc=torch.mean(soc_out_0**2+soc_out_1**2)
        ##0<x<1constraint
        out_bound=torch.relu(sample[:,:,[2,3,4,5,6]]-1)
        in_bound=torch.relu(0-sample[:,:,[2,3,4,5,6]])
        loss_con=torch.mean(out_bound**2+in_bound**2)
        loss=loss_con+loss_end_soc
        return loss


    def sample(self,batch_num,time_steps):
        #type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,weekday,month
        sample = np.zeros((batch_num, self.time_step,self.x_dim))
        x_last=torch.ones((batch_num,self.x_dim), device=self.device)#*(-0.1)
        #flag1=x_last[:,0]==0
        min,scale=self.transform
        c0=self.p0.generate_samples(batch_num).squeeze()
        m0=c0[:,-1].cpu().numpy()
        c0[:,-1]=(c0[:,-1]*12//1)%12#月份取整
        x_last[:,7:]=c0
        x=x_last.unsqueeze(1)
        for t in range(time_steps):
            xp=torch.cat([x[:,:,0:1],self.embed(x[:,:,1].to(torch.int)),x[:,:,2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
            xp[:,0,:-(5+self.embed_len)]=torch.ones(xp[:,0,:-(5+self.embed_len)].shape)*(-0.1)
            x_new=self.evaluate(xp,c0)
            day_filp=(x_last[:,2]+x_last[:,4]/24)>1#if sample_hour< last_start_hour+duration
            #start_soc 还原 
                   #更新weekday
                #weekday 更新 0,1, 
            flag1=(x_new[:,0]==1)
            #如果是此条是充电start_index（x[:,-5]）=end_index(x_new[:,1])
            x_last[:,1]=x_new[:,-6]*(~flag1)+x_new[:,1]*flag1
            #还原x_new 3,4,6
            x_last[:,[3,4,6]]=(torch.exp(x_new[:,[3,4,6]]*scale+min)-1)
            #还原x_new 记录trinp_kind和start_hour
            x_last[:,[0,2,5]]=x_new[:,[0,2,5]]
            #充电则距离设置为0
            x_last[~flag1,3]=0
            #其他变量保持不变
            x_last[:,7:]=c0
            #x_last[:,5]=x_new[:,5]
            #更新c0
            x_last[:,6]=x_last[:,6]//1
            c0[:,0]=x_new[:,1]#start_index更新
            c0[:,1]=x_new[:,5]#start_soc更新
            past_day= past_day=day_filp+x_last[:,6]
            c0[:,-2]=(c0[:,-2]+past_day)%7#weekday更新

            m0=update_month(m0,past_day.cpu().numpy())
            c0[:,-1]=torch.from_numpy(m0//1).float().to(self.device)#weekday更新
            #纠正错误sample
            #充电的错误sample loc
            x_new[:,1]=x_last[:,1]
            #充电了错误sample了distance
            x_new[:,3]=x_new[:,0]*x_new[:,3]
            x=torch.cat([x,x_new.unsqueeze(1)],dim=1)
            
            sample[:,t,:]=x_last.detach().cpu().numpy()
        return sample
    
    
    def sample_truncated(self,gmm, constraint):

        # 获取 GMM 参数
        means = gmm.component_distribution.mean.squeeze()
        stds = gmm.component_distribution.stddev.squeeze()
        mixture_weights = gmm.mixture_distribution.probs.squeeze()
        # 随机选择组件
        categorical = D.Categorical(mixture_weights)
        components_idx = categorical.sample().unsqueeze(-1)
        # 选取相应的均值和标准差
        mu = torch.gather(means,-1, components_idx).squeeze(-1)
        sigma = torch.gather(stds, -1, components_idx).squeeze(-1)
        # 创建一个截断正态分布并进行采样
        #sigma=np.clamp(sigma,min=1e-5)
        #sample=self.truncated_normal_rejection_sampling(mu,sigma,constraint[0],constraint[1])
        #sample=torch.from_numpy(sample)
        distribution = D.Normal(mu, sigma)
        # 将截断范围转换为与 mu 和 sigma 相同形状的张量
        min_x_tensor,max_x_tensor =constraint
        # 计算截断范围并采样
        a = distribution.cdf(min_x_tensor)
        b = distribution.cdf(max_x_tensor)
        p = a + (b - a) * torch.rand_like(a)
        p=torch.clip(p,min=1e-6,max=1-1e-6)
        samples = distribution.icdf(p.to(self.device))
        return samples.unsqueeze(1)
    
    def evaluate(self,x,c0):
        with torch.no_grad():
            ##decoder inputprocess
            x=self.local.eval()(x)
            x=self.backbone_net(x,eval_=True)
            x=x[:,-1,:]
            session_sample=self.decoder[0](x.unsqueeze(1))[0].sample()
            x=torch.cat([x,session_sample],dim=-1)
            loc_sample=self.decoder[1](x.unsqueeze(1))[0].sample()
            x=torch.cat([x,loc_sample],dim=-1)
            min_x= torch.zeros(len(x)).to(self.device)
            max_x=torch.ones(len(x)).to(self.device)
            constraints_1=(min_x,max_x)
            start_hour_sample=self.sample_truncated(self.decoder[2](x.unsqueeze(1))[0],constraints_1)
            x=torch.cat([x,start_hour_sample],dim=-1)
            distance_sample=self.sample_truncated(self.decoder[3](x.unsqueeze(1))[0],constraints_1)*(session_sample)
            x=torch.cat([x,distance_sample],dim=-1)
            duration_sample=self.sample_truncated(self.decoder[4](x.unsqueeze(1))[0],constraints_1)
            x=torch.cat([x,duration_sample],dim=-1)
            start_soc = c0[:, -5]
            trip_kind = session_sample.squeeze()
            # 当 trip_kind 为 0 时，min_x 为 start_soc，否则为 0
            min_x = torch.where(trip_kind == 0, start_soc, torch.tensor(0.0, device=self.device))
            max_x = torch.where(trip_kind == 1, start_soc, torch.tensor(1.0, device=self.device))
            constraints_2=  (min_x, max_x)
            end_soc_sample=self.sample_truncated(self.decoder[5](x.unsqueeze(1))[0],constraints_2)
            x=torch.cat([x,end_soc_sample],dim=-1)
            stay=self.sample_truncated(self.decoder[6](x.unsqueeze(1))[0],constraints_1)
            sample=torch.cat([x[:,-6:],stay,c0],dim=-1)
        return sample
    

    






