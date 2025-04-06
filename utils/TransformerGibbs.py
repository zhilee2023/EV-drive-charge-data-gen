import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from .util import PositionalEncoding, GMM_gibbs, Discrete_gibbs, update_month
from .initial_gen import Initial_Gen
import random

class TransformerGibbs(nn.Module):
    """
    该模型基于 Transformer 结构 + GMM/离散分布采样 (Gibbs) 的方式，
    用于对一段序列 (batch_x) 进行训练或采样。

    x_dim:         输入特征维度 (>= 7)
    time_step:     每个序列的时间步长度
    n_head:        多头注意力数量
    n_layers:      Transformer 编码器层数
    d_model:       Transformer 内部嵌入维度
    n_loc:         位置（location）的种类数量 (用于 embedding)
    embed_vector_len: 位置 embedding 的向量长度
    device:        运行设备 ('cpu' or 'cuda')
    tau:           温度参数 (gumble sample 相关)
    """

    def __init__(
        self, 
        x_dim, 
        time_step, 
        n_head, 
        n_layers, 
        d_model, 
        n_loc, 
        embed_vector_len, 
        device='cpu', 
        tau=0.01,
        rnn_backbone=False
    ):
        super(TransformerGibbs, self).__init__()

        self.x_dim = x_dim
        self.n_head = n_head
        self.time_step = time_step
        self.n_loc = n_loc
        self.device = device
        self.eps = torch.finfo(torch.float).eps
        self.embed_len = embed_vector_len
        self.rnn_backbone=rnn_backbone
        # Embedding 与相关维度
        self.embed = nn.Embedding(n_loc, embed_vector_len).to(self.device)
        self.x_enclen = x_dim - 2 + 2 * embed_vector_len
        self.d_model = d_model

        # Transformer 相关
        self.mask = torch.triu(torch.ones((time_step, time_step)) == 1, diagonal=1).to(self.device)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.2,
            device=self.device
        )
        if rnn_backbone==False:
            self.encoder_mod = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers).to(self.device)
        else:
            self.encoder_rnn = nn.LSTM(input_size=d_model,hidden_size=d_model,num_layers=n_layers,batch_first=True,dropout=0.2).to(self.device)

        self.pos = PositionalEncoding(d_model=d_model).to(self.device)
        self.local = nn.Linear(self.x_enclen, d_model).to(self.device)

        # 分布相关 (7 个输出分布)
        # type, loc, start_hour, distance, duration, end_soc, stay
        self.decoder_type = Discrete_gibbs(
            embed=self.embed,
            D_in=d_model + 10 + 2 * embed_vector_len,
            D_out=2,
            embed_len=embed_vector_len,
            device=self.device,
            tau=tau,
            var_len=x_dim
        )
        self.decoder_loc = Discrete_gibbs(
            embed=self.embed,
            D_in=d_model + 11 + embed_vector_len,
            D_out=n_loc,
            device=self.device,
            embed_len=embed_vector_len,
            tau=tau,
            var_len=x_dim
        )
        self.decoder_start_hour = GMM_gibbs(
            embed=self.embed,
            D_in=d_model + 10 + 2 * embed_vector_len,
            n_comp=8,
            device=self.device,
            tau=tau,
            var_len=x_dim
        )
        self.decoder_distance = GMM_gibbs(
            embed=self.embed,
            D_in=d_model + 10 + 2 * embed_vector_len,
            n_comp=5,
            device=self.device,
            tau=tau,
            var_len=x_dim
        )
        self.decoder_duration = GMM_gibbs(
            embed=self.embed,
            D_in=d_model + 10 + 2 * embed_vector_len,
            n_comp=5,
            device=self.device,
            tau=tau,
            var_len=x_dim
        )
        self.decoder_end_soc = GMM_gibbs(
            embed=self.embed,
            D_in=d_model + 10 + 2 * embed_vector_len,
            n_comp=6,
            device=self.device,
            tau=tau,
            var_len=x_dim
        )
        self.decoder_stay = GMM_gibbs(
            embed=self.embed,
            D_in=d_model + 10 + 2 * embed_vector_len,
            n_comp=3,
            device=self.device,
            tau=tau,
            var_len=x_dim
        )

        # 将 7 个解码器统一存放以便循环处理
        self.decoder = [
            self.decoder_type,
            self.decoder_loc,
            self.decoder_start_hour,
            self.decoder_distance,
            self.decoder_duration,
            self.decoder_end_soc,
            self.decoder_stay
        ]

        # 初始生成器
        self.p0 = Initial_Gen(
            n_loc=n_loc,
            h_dim=256,
            embed_vector_len=embed_vector_len,
            latent_dim=6,
            device=self.device
        )

        # 额外的 transform： 形状 (2, ) -> (min, scale)
        self.transform = torch.FloatTensor(np.load("min_scale.npy")).to(self.device)

        # 可选：如果有额外的 initial_model、PPO、DDPG 等，可以自行添加/删除
        self.initial_model = None
    
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

    def forward(self, batch_x, init):
        """
        前向传播计算: 
        1) 初始生成器的loss p0_loss
        2) 重构loss recon_loss
        3) 约束loss cond_loss

        batch_x: [B, T, x_dim]
        init:    用于 self.p0 计算初始分布的输入
        """
        # 初始生成器的 loss
        p0_loss = self.p0(init)

        # 准备训练数据
        y = batch_x.clone().detach()
        seq = torch.cat(
            [batch_x[:, :, :-1].clone(), (batch_x[:, :, -1:] * 12).floor().clone()],
            dim=-1
        )
        cond_x = seq.clone().detach()

        # 构建 x_init 以通过 Transformer 编码
        x_init = torch.cat([
            seq[:, :, 0:1],
            self.embed(seq[:, :, 1].to(torch.int)),
            seq[:, :, 2:-6],
            self.embed(seq[:, :, -6].to(torch.int)),
            seq[:, :, -5:]
        ], dim=-1)

        # 生成 t=0，用 -0.1 填充一个 token
        x_0 = torch.cat([
            torch.ones((x_init.shape[0], 1, x_init.shape[-1] - 5 - self.embed_len),
                       device=self.device) * -0.1,
            x_init[:, 0, -(5 + self.embed_len):].unsqueeze(1)
        ], dim=-1)
        x_init = torch.cat([x_0, x_init[:, :-1, :]], dim=1)

        # 送入 Transformer or rnn
        x_enc = self.local(x_init)
        x_enc = self.backbone_net(x_enc)

        # 拼接条件输入以解码
        decoder_input = torch.cat([x_enc, cond_x], dim=-1)

        # 分布 & 采样
        results = [dec(decoder_input, i) for i, dec in enumerate(self.decoder)]
        distributions, sample = zip(*results)
        sample = torch.cat(sample, dim=-1)

        # 计算各种损失
        cond_loss = self.constraint_loss(sample, cond_x)
        recon_loss = self.recon_loss(distributions, y)
        return p0_loss, recon_loss, cond_loss

    def constraint_loss(self, sample, cond_x):
        """
        针对  end_soc 与 start_soc 的约束以及区间 [0,1] 的约束。
        sample:  [B, T, 7]  (type, loc, start_hour, distance, duration, end_soc, stay)
        cond_x:  [B, T, ...]  (0:type,8:start_soc,...)
        """
        trip_kind = cond_x[:, :, 0]
        start_soc = cond_x[:, :, 8]
        # end_soc
        end_soc = sample[:, :, 5]

        # 当 trip_kind=0(充电) 时，start_soc-end_soc 应该 >=0
        # 当 trip_kind=1(驾驶) 时，start_soc-end_soc 应该 >=0
        # 下面分开处理
        flag = start_soc - end_soc
        soc_out_0 = torch.relu((1 - trip_kind) * flag)   # 充电时若 flag<0 => 惩罚
        soc_out_1 = torch.relu(-trip_kind * flag)        # 驾驶时若 flag>0 => 惩罚
        loss_end_soc = torch.mean(soc_out_0**2 + soc_out_1**2)

        # 额外的 [0,1] 约束（start_hour, distance, duration, end_soc, stay 等）
        # j in [2,3,4,5,6]
        out_of_upper = torch.relu(sample[:, :, [2, 3, 4, 5, 6]] - 1)
        below_zero = torch.relu(0 - sample[:, :, [2, 3, 4, 5, 6]])
        loss_bound = torch.mean(out_of_upper**2 + below_zero**2)

        return loss_end_soc + loss_bound

    def log_prob_forward(self, batch_x, target):
        """
        给定 batch_x, 计算目标序列 target 的对数似然 (log_prob)。
        target: [B, T, x_dim], 其中 x_dim=7  (type, loc, start_hour, distance, duration, end_soc, stay)
        """
        #with torch.no_grad():
        seq = torch.cat(
            [batch_x[:, :, :-1], (batch_x[:, :, -1:] * 12 // 1)],
            dim=-1
        )
        cond_x = seq.clone().detach()

        x_init = torch.cat([
            seq[:, :, 0:1],
            self.embed(seq[:, :, 1].to(torch.int)),
            seq[:, :, 2:-6],
            self.embed(seq[:, :, -6].to(torch.int)),
            seq[:, :, -5:]
        ], dim=-1)
        # t=0 占位
        x_0 = torch.cat([
            torch.ones((x_init.shape[0], 1, x_init.shape[-1] - 5 - self.embed_len),
                        device=self.device) * -0.1,
            x_init[:, 0, -(5 + self.embed_len):].unsqueeze(1)
        ], dim=-1)
        x_init = torch.cat([x_0, x_init[:, :-1, :]], dim=1)

        # 送入 Transformer
        x_enc = self.local(x_init)
        x_enc = self.backbone_net(x_enc)

        # 解码器
        decoder_input = torch.cat([x_enc, cond_x], dim=-1)
        results = list(map(lambda i: self.decoder[i](decoder_input,i),range(len(self.decoder)))) 
        #[dec(decoder_input, i) for i, dec in enumerate(self.decoder)]
        distributions, _ = zip(*results)

        # 拼合各个维度的 log_prob
        log_probs = [
            distributions[i].log_prob(target[:, :, i]).unsqueeze(-1)
            for i in range(len(distributions))
        ]
        log_probs = torch.cat(log_probs, dim=-1)
        return log_probs  # shape [B, T, 7]

    def recon_loss(self, distributions, y):
        """
        分布与真实值之间的负对数似然损失。
        distributions: tuple/list of dist (7个)
        y: [B, T, x_dim=7]
        """
        # type, loc, start_hour, distance, duration, end_soc, stay
        loss_type = -distributions[0].log_prob(y[:, :, 0].to(torch.int)).mean()
        loss_loc = -distributions[1].log_prob(y[:, :, 1]).mean()
        loss_start_hour = -distributions[2].log_prob(y[:, :, 2]).mean()
        loss_distance = -distributions[3].log_prob(y[:, :, 3]).mean()
        loss_duration = -distributions[4].log_prob(y[:, :, 4]).mean()
        loss_end_soc = -distributions[5].log_prob(y[:, :, 5]).mean()
        loss_stay = -distributions[6].log_prob(y[:, :, 6]).mean()

        total_loss = (loss_type + loss_loc + loss_start_hour + loss_distance
                      + loss_duration + loss_end_soc + loss_stay)
        return total_loss

    def sample_random(self, N, seq_len):
        """
        生成随机序列 (不经过模型)，主要用于部分初始化/测试。
        N: 批量大小
        seq_len: 时间步
        返回: [N, seq_len, x_dim] 的张量
        """
        type_values = torch.tensor([0, 1], device=self.device)
        loc_values = torch.arange(0, self.n_loc, device=self.device)

        data = torch.zeros((N, seq_len, 7), device=self.device)
        for i in range(seq_len):
            data[:, i, 0] = type_values[torch.randint(0, 2, (N,), device=self.device)]
            data[:, i, 1] = loc_values[torch.randint(0, self.n_loc, (N,), device=self.device)]
            data[:, i, 2:7] = torch.rand((N, 5), device=self.device)
        return data

    def load_initial_model(self, model):
        """
        加载一个额外的 initial_model (若需要使用 in forward / sample 过程中).
        """
        self.initial_model = model

    # ---------------------- 采样 / 评估 相关函数 ----------------------

    def sample_truncated(self, gmm, constraint):
        """
        从给定的 GMM 分布中进行有界 (截断) 采样。
        gmm:        torch.distributions.MixtureSameFamily
        constraint: (min_x, max_x), shape [B], 指定每个样本的最小和最大可取值
        返回:        [B] 的张量
        """
        means = gmm.component_distribution.mean.squeeze()
        stds = gmm.component_distribution.stddev.squeeze()
        weights = gmm.mixture_distribution.probs.squeeze()

        # 随机选一个组件
        comp_idx = D.Categorical(weights).sample().unsqueeze(-1)
        mu = torch.gather(means, -1, comp_idx).squeeze(-1)
        sigma = torch.gather(stds, -1, comp_idx).squeeze(-1)

        distribution = D.Normal(mu, sigma)
        min_x, max_x = constraint

        # 截断方式: 计算 cdf 区间 + uniform
        a = distribution.cdf(min_x)
        b = distribution.cdf(max_x)
        p = a + (b - a) * torch.rand_like(a)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)

        samples = distribution.icdf(p.to(self.device))
        return samples

    def sample_feature(self, x_dec, j, random_sample):
        """
        对第 j 个特征 (type=0, loc=1, start_hour=2, distance=3, duration=4, end_soc=5, stay=6) 进行一次采样。
        x_dec:         编码器输出与条件拼接后的输入
        j:             第 j 个解码器
        random_sample: 当前已采样的特征
        """
        # 根据 j 不同，设置不同的采样方式(截断或直接 sample)
        if j in [2, 3, 4, 6]:
            # start_hour / distance / duration / stay => [0,1] 范围
            min_x = torch.zeros(len(x_dec), device=self.device)
            max_x = torch.ones(len(x_dec), device=self.device)
            constraints = (min_x, max_x)
            dist = self.decoder[j](x_dec.unsqueeze(1), j)[0]
            sample_val = self.sample_truncated(dist, constraints)
            sample_val = torch.clamp(sample_val, 0, 1)

        elif j == 5:
            # end_soc: 与 trip_kind / start_soc 相关
            start_soc = random_sample[:, -5]  # c0[...,1], or cond_x[...,8]?
            trip_kind = random_sample[:, 0]

            min_x = torch.where(trip_kind == 0, start_soc, torch.tensor(0.0, device=self.device))
            max_x = torch.where(trip_kind == 1, start_soc, torch.tensor(1.0, device=self.device))
            dist = self.decoder[j](x_dec.unsqueeze(1), j)[0]
            sample_val = self.sample_truncated(dist, (min_x, max_x))
            sample_val = torch.clamp(sample_val, 0, 1)

        else:
            # type / loc => Discrete
            dist = self.decoder[j](x_dec.unsqueeze(1), j)[0]
            sample_val = dist.sample().squeeze()

        return sample_val

    def evaluate(self,x,c0,initial=None,step=15):
        with torch.no_grad():
            ##decoder inputprocess
            x=self.local.eval()(x)
            x=self.backbone_net(x,eval_=True)
            #x=self.decoder_mod(x)
            x=x[:,-1,:]
            if initial==None:
                random_sample=self.sample_random(x.shape[0],1).to(self.device).squeeze(1)
                random_sample=torch.cat([random_sample,c0],dim=-1)
            else:
                random_sample=initial
            x_new=torch.cat([x,random_sample],dim=-1)
            all_logs=[]
            for _ in range(step):
                for j in range(0,len(self.decoder)):
                    sample_update=self.sample_feature(x_new,j,random_sample)#self.decoder[j](x_new.unsqueeze(1),j)[0].sample()#self.sample_feature(x_new,j,random_sample)
                    random_sample[:,j]=sample_update.squeeze()
                    x_new=torch.cat([x,random_sample],dim=-1)
                    log=list(map(lambda i: self.decoder[i](x_new.unsqueeze(1),i)[0].log_prob(random_sample[:,i].unsqueeze(1)),range(len(self.decoder))))
                    log=torch.cat(log,dim=-1).sum(axis=-1)
                    all_logs.append(log.mean().cpu().numpy())
            sample=random_sample
        return sample,np.array(all_logs)

    # ---------------------- 采样接口示例: sample ----------------------

    def sample(self,batch_num,time_steps,initial=0,method=None,c0=None):
        #type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,weekday,month
        #self.ddpg_agent.update_decoder(self.decoder)
        sample = np.zeros((batch_num, time_steps,self.x_dim))
        x_last=torch.ones((batch_num,self.x_dim), device=self.device)#*(-0.1)
        #flag1=x_last[:,0]==0
        min_,scale=self.transform
        
        if c0==None:
            c0=self.p0.generate_samples(batch_num).squeeze()
        else:
            c0=c0.to(self.device)
        #print(c0)
        m0=c0[:,-1].cpu().numpy()*12
        c0[:,-1]=(c0[:,-1]*12//1)%12#月份取整
        x_last[:,7:]=c0
        x=x_last.unsqueeze(1)
        t_log_probs=[]
        reward=[]
        #z_old=None
        for t in range(time_steps):
            xp=torch.cat([x[:,:,0:1],self.embed(x[:,:,1].to(torch.int)),x[:,:,2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
            xp[:,0,:-(5+self.embed_len)]=torch.ones(xp[:,0,:-(5+self.embed_len)].shape)*(-0.1)
            if initial==1:
                r_sample=self.initial_model.evaluate(xp,c0)
                initial_vae=r_sample.clone()
            else:
                initial_vae=None
            #x_new,log_probs=self.evaluate(xp,c0,r_sample)
            x_new,log_probs=self.evaluate(xp,c0,initial_vae)
            
            day_filp= (x_last[:,2]+x_last[:,4]/24)>1 #if sample_hour< last_start_hour+duration
            #start_soc 还原 
                   #更新weekday
                #weekday 更新 0,1, 
            flag1=(x_new[:,0]==1)
            #如果是此条是充电start_index（x[:,-5]）=end_index(x_new[:,1])
            x_last[:,1]=x_new[:,-6]*(~flag1)+x_new[:,1]*flag1
            #还原x_new 3,4,6
            x_last[:,[3,4,6]]=(torch.exp(x_new[:,[3,4,6]]*scale+min_)-1)
            #还原x_new 记录trinp_kind和start_hour
            x_last[:,[0,2,5]]=x_new[:,[0,2,5]]
            #充电则距离设置为0
            x_last[~flag1,3]=0
            #其他变量保持不变
            x_last[:,7:]=c0
            #x_last[:,5]=x_new[:,5]
            x_last[:,6]=x_last[:,6]//1
            #更新c0
            c0[:,0]=x_new[:,1]#start_index更新
            c0[:,1]=x_new[:,5]#start_soc更新
            past_day=day_filp+x_last[:,6]#+(x_last[:,2]*24+x_last[:,4])//24
            c0[:,-2]=(c0[:,-2]+past_day)%7#weekday更新
            m0=update_month(m0,past_day)
            c0[:,-1]=torch.from_numpy(m0//1).float().to(self.device)#weekday更新
            #纠正错误sample
            #充电的错误sample loc
            x_new[:,1]=x_last[:,1]
            #充电了错误sample了distance
            x_new[:,3]=x_new[:,0]*x_new[:,3]
            x=torch.cat([x,x_new.unsqueeze(1)],dim=1)
            sample[:,t,:]=x_last.detach().cpu().numpy()
            t_log_probs.append(log_probs)
        return sample,np.array(t_log_probs),x[:,1:,:]

    def sample_one_step(self, x, step):
        """
        给定 batch_x (B,T,x_dim)，对每个时刻单独做 evaluate_one_step。
        返回一系列 (i,j) pairs，供外部对抗 / CPO 等使用。
        """
        with torch.no_grad():
            time_step = x.size(1)

            # 构建 x_init
            x_init = torch.cat([
                x[:, :, 0:1],
                self.embed(x[:, :, 1].to(torch.int)),
                x[:, :, 2:-6],
                self.embed(x[:, :, -6].to(torch.int)),
                x[:, :, -5:]
            ], dim=-1)
            x_0 = torch.cat([
                torch.ones((x_init.shape[0], 1, x_init.shape[-1] - 5 - self.embed_len),
                           device=self.device) * -0.1,
                x_init[:, 0, -(5 + self.embed_len):].unsqueeze(1)
            ], dim=-1)
            x_init = torch.cat([x_0, x_init[:, :-1, ]], dim=1)

            # 送入 Transformer
            x_enc = self.local.eval()(x_init)
            x_enc =self.backbone_net(x_enc,eval_=True)

            all_samples = []
            for t_idx in range(time_step):
                sample_t = self._evaluate_one_step(
                    x_enc[:, t_idx, :],
                    x[:, t_idx, -6:],
                    step=step
                )
                all_samples.append(sample_t)

            # 组合所有时刻的采样
            samples_cat = torch.cat(all_samples, axis=0).permute(1, 0, 2, 3)
            samples_cat = torch.cat([samples_cat, x.unsqueeze(-1)], axis=-1)

            # 选出 (i, j) pairs
            adv_sample_len = samples_cat.size(-1)
            pairs = [(adv_sample_len - 1, i) for i in range(adv_sample_len - 1)]

            out_samples = []
            for (p1, p2) in pairs:
                out_samples.append((samples_cat[..., p1], samples_cat[..., p2]))
        return out_samples

    def _evaluate_one_step(self, x_enc, c0, step=1):
        """
        内部辅助方法，针对某一时刻 x_enc + c0，做若干步的循环采样。
        """
        # 随机选择若干个时间点收集
        initial_step = random.randint(0, 7)
        all_samples_index = [i * 14 + initial_step for i in range(4)]

        with torch.no_grad():
            # 初始化
            if c0 is None:
                rand_sam = self.sample_random(x_enc.size(0), 1).squeeze(1)
                random_sample = torch.cat([rand_sam, c0], dim=-1)
            else:
                rand_sam = self.sample_random(x_enc.size(0), 1).squeeze(1)
                random_sample = torch.cat([rand_sam, c0], dim=-1)

            x_batch = torch.cat([x_enc, random_sample], dim=-1)
            all_collect = []

            for i in range(step):
                j = i % 7
                if i in all_samples_index:
                    all_collect.append(random_sample.unsqueeze(0))

                # 逐个特征采样
                dist = self.decoder[j](x_batch.unsqueeze(1), j)[0]
                sample_val = dist.sample().squeeze()
                random_sample[:, j] = sample_val

                x_batch = torch.cat([x_enc, random_sample], dim=-1)

        if len(all_collect) > 0:
            return torch.cat(all_collect, axis=0).permute(1, 2, 0).unsqueeze(0)
        else:
            return random_sample.unsqueeze(0)
