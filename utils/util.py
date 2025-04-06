import torch
import torch.nn as nn
import math
import torch.distributions as D
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def print_model_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()  # numel() 返回参数的元素个数
    print(f'Total parameters size: {total_params}')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Modify the shape to [1, max_len, d_model] for batch_first
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # Modify the addition operation for batch_first
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
##GMM and discrete for gibbs
class GMM_gibbs(torch.nn.Module):
    def __init__(self,embed, D_in,device,n_comp=2,tau=0.01,var_len=13):
        super(GMM_gibbs, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in).to(device)
        self.linear2 = torch.nn.Linear(D_in, D_in).to(device)
        self.n_comp=n_comp
        self.linear_output = torch.nn.Linear(D_in, self.n_comp*3) .to(device)
        self.activation=torch.nn.LeakyReLU()
        self.device=device
        self.tau=tau
        self.embed=embed
        self.var_len=var_len
    def forward(self,x,i):
        x=torch.cat([x[:,:,:-self.var_len+1],self.embed(x[:,:,-self.var_len+1].to(torch.int)),\
                    x[:,:,-self.var_len+2:-(self.var_len-i)],x[:,:,-(self.var_len-i)+1:-6],\
                        self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],dim=-1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        output=self.linear_output(x)
        mus=output[:,:,:self.n_comp]
        sigs=torch.exp(output[:,:,self.n_comp:2*self.n_comp]*0.5)
        weights=torch.softmax(output[:,:,2*self.n_comp:3*self.n_comp],axis=-1)
        mix = D.Categorical(weights)
        comp = D.normal.Normal(mus,sigs)
        gmm = D.MixtureSameFamily(mix, comp)
        sample=mus+sigs*torch.empty(size=mus.size(), device=self.device, dtype=torch.float).normal_()
        sample=(sample*F.gumbel_softmax(output[:,:,2*self.n_comp:3*self.n_comp], tau=self.tau, hard=True)).sum(dim=-1, keepdim=True)
        return gmm,sample


class Discrete_gibbs(torch.nn.Module):
    def __init__(self, embed,D_in, D_out,device,embed_len,tau=0.01,var_len=13):
        super(Discrete_gibbs, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in).to(device)
        self.linear2 = torch.nn.Linear(D_in, D_in).to(device)       
        self.linear_output = torch.nn.Linear(D_in, D_out).to(device)
        self.activation=torch.nn.LeakyReLU()
        self.device=device
        self.d_out=D_out
        self.embed_len=embed_len
        self.tau=tau
        self.embed=embed
        self.var_len=var_len
    def forward(self, x,i):
        # type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,week_day
        #x=torch.cat([x[:,:,:i],x[:,:,i+1:]],dim=-1)
        if i==0:
            x=torch.cat([x[:,:,:-self.var_len],self.embed(x[:,:,-self.var_len+1].to(torch.int)),x[:,:,-self.var_len+2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
        elif i==1:
            x=x=torch.cat([x[:,:,:-self.var_len+1],x[:,:,-self.var_len+2:-6],self.embed(x[:,:,-6].to(torch.int)),x[:,:,-5:]],axis=-1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear_output(x)
        distribution= torch.distributions.Categorical(torch.softmax(x,axis=-1))
        index_mat = torch.arange(0, self.d_out).float().unsqueeze(0).unsqueeze(0).expand(x.size(0),x.size(1), -1).to(self.device)
        sample=(index_mat*F.gumbel_softmax(x, tau=self.tau, hard=True)).sum(dim=-1,keepdim=True)
        return distribution,sample

def data_preprocess_test(df):
    df['battery_capacity']=round(df['battery_capacity'],1)
    df['seq_code']=[i//60  for i in range(len(df))]
    battery_capacity_list=df['battery_capacity'].unique()
    battery_diction=dict(list(zip(battery_capacity_list,range(len(battery_capacity_list)))))
    file_path='battery_diction.json'
    with open(file_path, 'w') as file:
        json.dump(battery_diction, file)
    #df['battery_capacity']=df['battery_capacity'].apply(lambda x:battery_diction[x])
    cc=['distance','duration','stay']
    df[cc]=np.log(1+df[cc])
    min_=list(df[cc].min())
    scale=list((df[cc].max()-df[cc].min()))
    df[cc]= (df[cc]-df[cc].min())/(df[cc].max()-df[cc].min())
    np.save("min_scale.npy",np.array([min_,scale]))
    return df

# def data_preprocess(df):
#     #df=pd.read_csv("BEV_All.csv")
#     df['trip_kind']=df['trip_kind'].apply(lambda x:(x=='D')*1.0)
#     df['start_time']=pd.to_datetime(df['start_time'])
#     df.sort_values(by=['seq_code', 'start_time'], inplace=True)  # 按 seq_code 和 start_time 排序
#     battery_capacity_list=df['battery_capacity'].unique()
#     battery_diction=dict(list(zip(battery_capacity_list,range(len(battery_capacity_list)))))
#     df['stay']=(df['duration_stay']-df['duration']/60)/24
#     df['month']=(df['start_time'].apply(datetime_to_fract))/12
#     #df['stay']=df['stay']/df['stay'].max()
#     df['duration']/=60
#     df['start_soc']/=100
#     df['end_soc']/=100
#     df=df[['seq_code','start_time','trip_kind','start_index','energy_change','duration','distance','stay','start_soc','end_soc','start_hour', 'weekday', 'label', 'battery_capacity','month']]
#     df['start_hour']/=24
#     df['soc_change']=(df['energy_change']/df['battery_capacity'])
#     df['end_index'] = df['start_index'].shift(-1)
#     #df.at[df.index[-1], 'end_loc'] = random.randint(0, n_loc)
#     #remove last time series
#     df=df.groupby('seq_code').apply(lambda x:x.iloc[:-1]).reset_index(drop=True)
#     file_path='battery_diction.json'
#     with open(file_path, 'w') as file:
#         json.dump(battery_diction, file)
#     #df['battery_capacity']=df['battery_capacity'].apply(lambda x:battery_diction[x])
#     cc=['distance','duration','stay']
#     df[cc]=np.log(1+df[cc])
#     min_=list(df[cc].min())
#     scale=list((df[cc].max()-df[cc].min()))
#     df[cc]= (df[cc]-df[cc].min())/(df[cc].max()-df[cc].min())
#     np.save("min_scale.npy",np.array([min_,scale]))
#     return df


##GMM and discrete for gan and vae
class GMM(torch.nn.Module):
    def __init__(self,embed, D_in,device,pos=False,n_comp=2,tau=0.01):
        super(GMM, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in).to(device)
        self.linear2 = torch.nn.Linear(D_in, D_in).to(device)
        self.n_comp=n_comp
        #self.linear2 = torch.nn.Linear(D_in//2, D_in//2)    
        self.linear_output = torch.nn.Linear(D_in, self.n_comp*3) .to(device)
        #self.batchnorm1=torch.nn.BatchNorm1d(D_in,)
        self.activation=torch.nn.LeakyReLU()
        self.softplus=torch.nn.Softplus(beta=1, threshold=20)
        self.device=device
        self.pos=pos
        self.tau=tau
        self.embed=embed
    
    def forward(self,x):
        #x=torch.cat([x[:,:,:-(9-i)],x[:,:,-(9-i)+1:]],dim=-1)
        #x=torch.cat([x[:,:,:i],x[:,:,i+1:]],dim=-1)
        #x=torch.cat([x[:,:,:-11],self.embed(x[:,:,-11].to(torch.int)),x[:,:,-10:-(12-i)],x[:,:,-(12-i)+1:-5],self.embed(x[:,:,-5].to(torch.int)),x[:,:,-4:]],axis=-1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        output=self.linear_output(x)
        mus=output[:,:,:self.n_comp]
        sigs=torch.exp(output[:,:,self.n_comp:2*self.n_comp]*0.5)
        weights=torch.softmax(output[:,:,2*self.n_comp:3*self.n_comp],axis=-1)
        mix = D.Categorical(weights)
        comp = D.normal.Normal(mus,sigs)
        gmm = D.MixtureSameFamily(mix, comp)
        sample=mus+sigs*torch.empty(size=mus.size(), device=self.device, dtype=torch.float).normal_()
        sample=(sample*F.gumbel_softmax(output[:,:,2*self.n_comp:3*self.n_comp], tau=self.tau, hard=True)).sum(dim=-1, keepdim=True)
        return gmm,sample


class Discrete(torch.nn.Module):
    def __init__(self, embed,D_in, D_out,device,embed_len,tau=0.01):
        super(Discrete, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in).to(device)
        self.linear2 = torch.nn.Linear(D_in, D_in).to(device)      
        self.linear_output = torch.nn.Linear(D_in, D_out).to(device)
        self.activation=torch.nn.LeakyReLU()
        self.device=device
        self.d_out=D_out
        self.embed_len=embed_len
        self.tau=tau
        self.embed=embed.to(device)
    def forward(self, x):
        # type, end_index,start_hour,distance,duration,end_soc,stay,start_index,start_soc,uer_label,battery_capacity,week_day
        #x=torch.cat([x[:,:,:i],x[:,:,i+1:]],dim=-1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear_output(x)
        distribution= torch.distributions.Categorical(torch.softmax(x,axis=-1))
        index_mat = torch.arange(0, self.d_out).float().unsqueeze(0).unsqueeze(0).expand(x.size(0),x.size(1), -1).to(self.device)
        sample=(index_mat*F.gumbel_softmax(x, tau=self.tau, hard=True)).sum(dim=-1,keepdim=True)
        return distribution,sample
    
#GMM and discrete for bayes
class GMM_bayes(torch.nn.Module):
    def __init__(self, D_in, device,n_comp=2,var_len=13,tau=0.01):
        super(GMM_bayes, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in).to(device)
        self.linear2 = torch.nn.Linear(D_in, D_in).to(device)
        self.n_comp=n_comp
        self.linear_output = torch.nn.Linear(D_in, self.n_comp*3).to(device)
        self.activation=torch.nn.LeakyReLU()
        self.device=device
        self.var_len=var_len
        self.tau=tau
    def forward(self, x,i=None):
        if i !=None:
            x=x[:,:,:-(self.var_len-i)]
        x=self.activation(self.linear1(x))
        x=self.activation(self.linear2(x))
        output=self.linear_output(x)
        mus=output[:,:,:self.n_comp]
        sigs=torch.exp(output[:,:,self.n_comp:2*self.n_comp]*0.5)
        weights=torch.softmax(output[:,:,2*self.n_comp:3*self.n_comp],axis=1)
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(mus.unsqueeze(-1),sigs.unsqueeze(-1)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        sample=mus+sigs*torch.empty(size=mus.size(), device=self.device, dtype=torch.float).normal_()
        sample=(sample * F.gumbel_softmax(output[:,:,2*self.n_comp:3*self.n_comp], tau=self.tau, hard=True)).sum(dim=-1, keepdim=True)
        return gmm,sample


class Discrete_bayes(torch.nn.Module):
    def __init__(self, D_in, D_out,device,var_len=13,tau=0.01):
        super(Discrete_bayes, self).__init__()
        self.linear_1 = torch.nn.Linear(D_in, D_in).to(device)
        self.linear_2 = torch.nn.Linear(D_in, D_in).to(device)       
        self.linear_output = torch.nn.Linear(D_in, D_out).to(device)
        self.activation=torch.nn.LeakyReLU()
        self.device=device
        self.d_out=D_out
        self.var_len=var_len
        self.tau=tau
    def forward(self, x,i=None):
        if i !=None:
            x=x[:,:,:-(self.var_len-i)]
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.linear_output(x)
        distribution= torch.distributions.Categorical(torch.softmax(x,axis=-1))
        index_mat = torch.arange(0, self.d_out).float().unsqueeze(0).unsqueeze(0).expand(x.size(0),x.size(1), -1).to(self.device)
        sample=(index_mat*F.gumbel_softmax(x, tau=self.tau, hard=True)).sum(dim=-1,keepdim=True)
        return distribution,sample

def update_month_one(start, days_to_add):
    # 解析起始月份和小数部分
    month = int(start%12) + 1  # 将0-11范围的输入调整为1-12的月份
    fraction = start - (month - 1)

    # 每月的天数（不考虑闰年）
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # 计算起始日期
    start_day = np.clip(int(fraction * (days_in_month[month-1]) + 1),1,days_in_month[month-1])
    start_date = datetime(2023, month, start_day)  # 年份仅用于创建日期对象

    # 计算新日期
    new_date = start_date + timedelta(days=int(days_to_add))

    # 循环月份计算
    new_month = new_date.month
    day_of_month = new_date.day
    total_days_in_new_month = days_in_month[new_month-1]
    month_fraction = (day_of_month-1) / total_days_in_new_month

    # 生成输出月份的小数表示，调整为0-11范围
    output_month = (new_month - 1) + month_fraction
    return output_month

def update_month(starts, days_to_adds):
    # 使用向量化处理
    results = np.array([update_month_one(start, days) for start, days in zip(starts, days_to_adds)])
    return results


def datetime_to_fract(date):
    # 每月天数（假设非闰年，闰年2月需手动调整为29）
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # 当前月份
    month = date.month
    # 当前月份的天数
    total_days_in_month = month_days[month-1]
    # 当前月的日数（第几天）
    day_of_month = date.day
    # 计算小数月份
    # 减1是因为month是从1开始的，我们需要的是0-11的范围
    month_decimal = (month - 1) + (day_of_month - 1) / total_days_in_month
    return month_decimal

