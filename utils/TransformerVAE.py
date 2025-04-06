import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributions as D
from .util import PositionalEncoding,GMM, Discrete,update_month
from .initial_gen import Initial_Gen

class TransformerVAE(nn.Module):
    def __init__(self, x_dim, time_step,n_head,n_layers,d_model,n_loc,embed_vector_len,device='cpu',z_dim=4,tau=0.01,rnn_backbone=False):
        super(TransformerVAE,self).__init__()
        self.device=device
        self.x_dim = x_dim
        self.n_head=n_head
        self.time_step=time_step
        self.n_loc=n_loc
        self.eps=torch.finfo(torch.float).eps
        self.embed=torch.nn.Embedding(n_loc+1, embed_vector_len).to(self.device)
        self.embed_len=embed_vector_len
        self.time_step=time_step
        self.x_enclen=x_dim-2+2*embed_vector_len
        self.d_model=d_model
        self.mask=torch.triu(torch.ones((self.time_step,self.time_step))==1,diagonal=1).to(self.device)
        self.transform=torch.FloatTensor(np.load("min_scale.npy")).to(self.device)
        self.tau=tau
        self.rnn_backbone=rnn_backbone
        self.z_dim=z_dim
        self.z_output=nn.Linear(self.d_model,(z_dim)*2).to(self.device)
        self.z_decoder=nn.Linear(self.z_dim,self.d_model).to(self.device)
        if self.rnn_backbone==False:
            self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,dim_feedforward=self.d_model*4,batch_first=True,device=self.device)
            self.encoder_mod = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)#.to(self.device)
        else:
            self.encoder_rnn = nn.LSTM(input_size=d_model,hidden_size=d_model,num_layers=n_layers,batch_first=True,dropout=0.1).to(self.device)
        

        
        if self.rnn_backbone==False:
            self.decoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,dim_feedforward=self.d_model*4,batch_first=True,device=self.device)
            self.decoder_mod = nn.TransformerEncoder(self.decoder_layer, num_layers=n_layers)#.to(self.device)
        else:
            self.decoder_rnn = nn.LSTM(input_size=d_model,hidden_size=d_model,num_layers=n_layers,batch_first=True,dropout=0.1).to(self.device)

        self.pos= PositionalEncoding(d_model=self.d_model).to(self.device)
        self.output_len=self.n_loc+2+4*2
        #self.output_linear=nn.Linear(self.d_model,self.output_len).to(device)
        self.local=nn.Linear(self.x_enclen,self.d_model).to(self.device)
        self.local_decoder=nn.Linear(self.x_enclen+self.z_dim,self.d_model).to(self.device)
        self.discrete_loss=torch.nn.CrossEntropyLoss()
        #self.augment_mul=5
        self.decoder_type=Discrete(embed=self.embed,D_in=self.d_model,D_out=2,embed_len=self.embed_len,device=self.device,tau=tau).to(self.device)
        self.decoder_loc=Discrete(embed=self.embed,D_in=self.d_model,D_out=self.n_loc,device=self.device,embed_len=self.embed_len,tau=tau).to(self.device)
        self.decoder_start_hour=GMM(embed=self.embed,D_in=self.d_model,n_comp=7,device=self.device,tau=tau).to(self.device)
        self.decoder_distance=GMM(embed=self.embed,D_in=self.d_model, n_comp=3,device=self.device,tau=tau).to(self.device)
        self.decoder_duration=GMM(embed=self.embed,D_in=self.d_model, n_comp=5,device=self.device,tau=tau).to(self.device)
        self.decoder_end_soc=GMM(embed=self.embed,D_in=self.d_model, n_comp=6,device=self.device,tau=tau).to(self.device)
        #self.decoder_end_soc=Beta(embed=self.embed,D_in=self.d_model+9+2*self.embed_len,device=self.device).to(device)
        self.decoder_stay=GMM(embed=self.embed,D_in=self.d_model, n_comp=3,device=self.device,tau=tau).to(self.device)
        self.decoder=[self.decoder_type,self.decoder_loc,self.decoder_start_hour,self.decoder_distance,
                        self.decoder_duration, self.decoder_end_soc,self.decoder_stay]
        self.p0=Initial_Gen(n_loc=n_loc, h_dim=256, embed_vector_len=self.embed_len,latent_dim=6,device=self.device)
    
    
    def backbone_net_enc(self,x,eval_=False):
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
    
    def backbone_net_dec(self,x,eval_=False):
        if eval_==False:
            if self.rnn_backbone==False:
                x = self.pos(x)
                x = self.decoder_mod(x, mask=self.mask)
            else:
                x,_=self.decoder_rnn(x)
        else:
            if self.rnn_backbone==False:
                x = self.pos.eval()(x)
                x = self.decoder_mod.eval()(x)
            else:
                x,_=self.decoder_rnn.eval()(x)
        return x

    def forward(self,batch_x,init):
        # type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,week_day,month
        #decoder input process
        p0_loss=self.p0(init)
        y=batch_x.clone().detach()
        seq=torch.cat([batch_x[:,:,:-1].clone(),(batch_x[:, :, -1:] * 12).floor().clone()],axis=-1)
        #seq[:,:,6]=(seq[:,:,6]//1).detach()
        x=seq.clone().detach()
        cond_x=seq.clone().detach()
        x_encoder_init=torch.cat([x[:,:,0:1],self.embed(x[:,:,1].to(torch.int)),x[:,:,2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
        #生成x0 start_index,start_soc,uer_label,battery_capacity,week_day基于此数据t0生成t1
        x_0=torch.cat([torch.ones((x_encoder_init.shape[0],1,x_encoder_init.shape[-1]-5-self.embed_len)).to(self.device)*(-0.1),x_encoder_init[:,0,-(5+self.embed_len):].unsqueeze(1)],dim=-1)
        x_decoder_init=torch.cat([x_0,x_encoder_init[:,0:-1,:]],dim=1)

        ##decoder inputprocess
        x=self.local(x_encoder_init)
        x=self.backbone_net_enc(x)
        z=self.z_output(x)
        kl_div_loss=self.kl_divergence_loss(z)
        mean = z[..., :self.z_dim]
        log_var = z[..., self.z_dim:]
        z=self.sample_normal(mean, torch.sqrt(torch.exp(log_var)))
        #x=self.z_decoder(z)
        x=torch.cat([x_decoder_init,z],dim=-1)
        x=self.local_decoder(x)
        x=self.backbone_net_dec(x)
        results=list(map(lambda i: self.decoder[i](x),range(len(self.decoder))))       
        distributions,sample=tuple(zip(*results))
        sample=torch.cat(sample,dim=-1)
        recon_loss=self.recon_loss(distributions,y)#torch.sum(list(map(lambda i:(distributions[i].log_prob(y[:,:,i])).mean(),range(len(self.decoder)))))
        cond_loss= self.constraint_loss(sample,cond_x)
        return p0_loss,recon_loss,kl_div_loss,cond_loss
    

    
    def kl_divergence_loss(self,output):
        """
        计算 KL 散度损失函数
        :param output: 形状为 (batch, seq_len, zdim*2) 的张量，包含均值和对数方差
        :return: KL 散度损失
        """
        # 分离均值和对数方差
        zdim = output.shape[-1] // 2
        mean = output[..., :zdim]
        log_var = output[..., zdim:]

        # 计算 KL 散度
        kl_div = 0.5 * (torch.exp(log_var) + mean**2 - 1 - log_var)

        # 求和得到总的 KL 散度损失
        kl_loss = kl_div.sum()

        return kl_loss

    def recon_loss(self,distributions,y): 
        # type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,week_day
        #mask_distance=(y[:,:,0]==1)
        #mask_loc= torch.cat([torch.ones(y.size(0), 1, dtype=torch.bool,device=self.device), mask_distance[:, :-1]], dim=1)
        session_loss=-distributions[0].log_prob(y[:,:,0].to(torch.int)).mean()
        loc_loss=-distributions[1].log_prob(y[:,:,1].to(torch.int)).mean()
        start_hour_loss=-distributions[2].log_prob(y[:,:,2]).mean()
        distance_loss=-distributions[3].log_prob(y[:,:,3]).mean()
        duration_loss=-distributions[4].log_prob(y[:,:,4]).mean()
        soc_change_loss=-distributions[5].log_prob(y[:,:,5]).mean()
        stay_loss=-distributions[6].log_prob(y[:,:,6]).mean()
        loss=loc_loss+session_loss+soc_change_loss+distance_loss+\
             stay_loss+duration_loss+start_hour_loss 
        return loss
    
    
    def constraint_loss(self,sample,cond_x):
    #type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,weekday,month
        
        trip_kind=cond_x[:,:,0]
        start_soc=cond_x[:,:,8]
        #distance=cond_x[:,:,3]
        #end_soc=cond_x[:,:,5]
        
        '''
        ##trip_kind constraint
        distance_constriant=torch.mean((distance*(1-sample[:,:,0]))**2)#如果distance>0则sample出来为0被惩罚。
        
        
        flag=(start_soc-end_soc)
        soc_out_0=torch.relu((1-sample[:,:,0])*flag)##如果是充电的话那么start_soc-end_soc为正的被惩罚
        soc_out_1=torch.relu(-trip_kind*flag)##如果是驾驶的话那么start_soc-end_soc为负的被惩罚
        loss_end_soc=torch.mean(soc_out_0**2+soc_out_1**2)
        drive_charge_constraint=torch.mean(soc_out_0**2+soc_out_1**2)
        '''
        ##end SOC constraint
        flag=(start_soc-sample[:,:,5])
        soc_out_0=torch.relu((1-trip_kind)*flag)##如果是充电的话那么start_soc-end_soc为正的被惩罚
        soc_out_1=torch.relu(-trip_kind*flag)##如果是驾驶的话那么start_soc-end_soc为负的被惩罚
        loss_end_soc=torch.mean(soc_out_0**2+soc_out_1**2)
        
        ##0<x<1constraint
        out_bound=torch.relu(sample[:,:,[2,3,4,5,6]]-1)
        in_bound=torch.relu(0-sample[:,:,[2,3,4,5,6]])
        loss_con=torch.mean(out_bound**2+in_bound**2)

        ##distance_cosntraint
        #distance_constriant_2=torch.mean(torch.relu(sample[:,:,3]*(1-trip_kind))**2) ##如果trip_kind=0那么sample出来>0则惩罚
        loss=loss_con+loss_end_soc#+distance_constriant_2
        return loss
    
    
    def sample_normal(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
    
    def sample(self,batch_num,time_steps):
        #type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,month
        sample = np.zeros((batch_num, self.time_step,self.x_dim))
        x_last=torch.ones((batch_num,self.x_dim), device=self.device)#*(-0.1)
        #flag1=x_last[:,0]==0
        c0=self.p0.generate_samples(batch_num).squeeze()
        m0=c0[:,-1].cpu().numpy()
        c0[:,-1]=(c0[:,-1]*12//1)%12#月份取整
        x=x_last.unsqueeze(1)
        z_old=None
        min_,scale=self.transform
        
        for t in range(time_steps):
            xp=torch.cat([x[:,:,0:1],self.embed(x[:,:,1].to(torch.int)),x[:,:,2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
            xp[:,0,:-(5+self.embed_len)]=torch.ones(xp[:,0,:-(5+self.embed_len)].shape)*(-0.1)
            x_new,z_old=self.evaluate(xp,c0,z_old)
            #start_soc 还原 
                   #更新weekday
                #weekday 更新 0,1,
            #x_new=torch.cat([x_new,c0],dim=-1)
            day_filp=(x_last[:,2]+x_last[:,4]/24)>1 #if sample_hour< last_start_hour+duration
            flag1=(x_new[:,0]==1)
            #如果是此条是充电start_index（x[:,-5]）=end_index(x_new[:,1])
            x_last[:,1]=x_new[:,-5]*(~flag1)+x_new[:,1]*flag1
            #还原x_new 3,4,6
            x_last[:,[3,4,6]]=(torch.exp(x_new[:,[3,4,6]]*scale+min_)-1)
            #还原x_new 记录trinp_kind和start_hour
            x_last[:,[0,2,5]]=x_new[:,[0,2,5]]
            #充电则距离设置为0
            x_last[~flag1,3]=0
            #其他变量保持不变
            x_last[:,7:]=c0
            #x_last[:,5]=x_new[:,5]
            #纠正错误sample
            #充电的错误sample loc
            x_new[:,1]=x_last[:,1]
            #充电了错误sample了distance
            x_new[:,3]=x_new[:,0]*x_new[:,3]
            x_last[:,6]=x_last[:,6]//1
            x=torch.cat([x,x_new.unsqueeze(1)],dim=1)
            #更新c0
            c0[:,0]=x_new[:,1]#start_index更新
            c0[:,1]=x_new[:,5]#start_soc更新
            past_day=day_filp+x_last[:,6]//1
            c0[:,-2]=(c0[:,-2]+past_day)%7#weekday更新
            m0=update_month(m0,past_day.cpu().numpy())
            c0[:,-1]=torch.from_numpy(m0//1).float().to(self.device)#weekday更新
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
        return samples
    
    
    def sample_normal(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
    
    def evaluate(self,x,c0,z_old):
        with torch.no_grad():
            ##decoder inputprocess
            #c0=tart_index,start_soc,uer_label,battery_capacity,week_day
            ##decoder inputprocess
            #'trip_kind','end_index','distance','duration','end_soc','stay','start_hour','start_index','start_soc', 'label','battery_capacity','weekday'
            z=self.sample_normal(torch.zeros((len(x),1,self.z_dim)).to(self.device),torch.ones((len(x),1,self.z_dim)).to(self.device))
            if x.size(1)>1:
                z=torch.cat([z_old,z],dim=1)
            #x=self.z_decoder(z)
            x=torch.cat([x,z],dim=-1)
            x=self.local_decoder.eval()(x)
            x=self.backbone_net_dec(x,eval_=True)
            x=x[:,-1,:].unsqueeze(1)
            sample=[]
            #type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity
            #sample=list(map(lambda i: self.decoder[i](x)[0].sample(),range(len(self.decoder))))
            for j in range(0,7):
                if j in [2,3,4,5,6]:
                    #sample_update=self.sample_until_positive(self.decoder[j](x_new.unsqueeze(1),j)[0]).squeeze()
                    #sample_update=self.sample_truncated(self.decoder[j](x_new.unsqueeze(1),j)[0],(0,1))#
                    #sample_update=self.decoder[j](x_new.unsqueeze(1),j)[0].sample().squeeze()
                    min_x= torch.zeros(len(x)).to(self.device)
                    max_x=torch.ones(len(x)).to(self.device)
                    constraints=(min_x,max_x)
                    gmm=self.decoder[j](x)[0]
                    sample_update=self.sample_truncated(gmm,constraints)
                    sample.append(torch.clamp(sample_update,0,1).unsqueeze(1))
                else:
                    sample.append(self.decoder[j](x)[0].sample())
            sample=torch.cat(sample,dim=1)
            sample=torch.cat([sample,c0],dim=-1)
            return sample,z

