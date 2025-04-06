import pandas as pd
import numpy as np
from .dataset_gen import InitialDataset2
from .util import data_preprocess_test
from .rho_cal import calculate_rho
from torch.utils.data import DataLoader

def model_compare(sample_num,sample_batch_size,model,directory,time_steps=60,initial=0, save="samples.csv",filter=False):
    #model_type='Transformer_gibbs'
    x_dim=13
    data=np.array(np.zeros((0,x_dim)))
    step=0
    #time_steps=60
    log_probs=[]
    if initial==0:
        for _ in range(sample_num):
            tt,log_prob,_=model.sample(sample_batch_size,time_steps,initial=0,method=None)
            new_array =tt.reshape(tt.shape[0]*tt.shape[1],-1)
            data=np.concatenate([data,new_array],axis=0)
            step+=1
            log_probs.append(log_prob)
    else:

        df=pd.read_csv("samples.csv")
        if filter:
            file_path = 'ratio_within_center.json'
            df_from_json = pd.read_json(file_path)
            # Filtering rows where 'within_center' is greater than a specified value (e.g., 0.75)
            filtered_seq_code = df_from_json[df_from_json['within_center'] > 0.75]
            df=df[df['seq_code'].isin(filtered_seq_code.index)]
        df=data_preprocess_test(df)
        dataloader_initial = DataLoader(InitialDataset2(df), batch_size=sample_batch_size, shuffle=True,drop_last=True)
        
        for init_batch in dataloader_initial:
            tt,log_prob,_=model.sample(sample_batch_size,time_steps,initial=0,method=None,c0=init_batch)
            new_array =tt.reshape(tt.shape[0]*tt.shape[1],-1)
            data=np.concatenate([data,new_array],axis=0)
            log_probs.append(log_prob)
            step+=1
            if step>=sample_num:
                break;

    # initial_state=pd.DataFrame(data_c0)
    # initial_state.columns=["start_index","start_soc","label","battery_capacity","weekday",'month']
    # initial_state.to_csv("initial_state.csv",index=False)
    np.save(directory+"/log_prob.npy",np.array(log_probs))

    column=["trip_kind", "end_index","start_hour","distance","duration","end_soc","stay","start_index","start_soc","label","battery_capacity","weekday",'month']
    dff=pd.DataFrame(data)
    dff.columns=column
    dff.to_csv(directory+"/"+save,index=False)
    rho1,rho2=calculate_rho(directory+"/"+save,filter=filter)
    return rho1,rho2