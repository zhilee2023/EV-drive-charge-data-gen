import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon as js
from itertools import product


def calculate_rho(sample_file,filter=False):
    original_data=pd.read_csv("./samples.csv")
    if filter:
        file_path = 'ratio_within_center.json'
        df_from_json = pd.read_json(file_path)
        filtered_seq_code = df_from_json[df_from_json['within_center'] > 0.75]
        original_data=original_data[original_data['seq_code'].isin(filtered_seq_code.index)]
    #sample_data=pd.read_csv("sample_PPO.csv")
    sample_data=pd.read_csv(sample_file)
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
        # 减1是因为month是从1开始的，我们需要的是0-11的范围c
        month_decimal = (month - 1) + (day_of_month - 1) / total_days_in_month
        return month_decimal
    def normalize_min_max(data, min_x, max_x):
        return (np.array(data) - min_x) / (max_x - min_x+1e-20)
    def group_average_wdistance(grouped1,grouped2):
        wasserstein_distances = {}
        lengths={}
        keys = set(grouped1.index).intersection(set(grouped2.index))
        for key in keys:
            max_x=max([max(grouped1[key]),max(grouped2[key])])
            min_x=min([min(grouped1[key]),min(grouped2[key])])
            #print(grouped1[key]
            a=np.array(grouped1[key])#/max_x
            b=np.array(grouped2[key])#/max_x
            a = normalize_min_max(grouped1[key], min_x, max_x)
            b = normalize_min_max(grouped2[key], min_x, max_x)
            #random_samples = np.random.uniform(low=0, high=1,size=len(a))
            #distance = wasserstein_distance(a,b)/wasserstein_distance(a,random_samples)
            wasserstein_distances[key] = wasserstein_distance(a,b)
            lengths[key] = len(grouped1[key])
        #average_distance = np.mean(list(wasserstein_distances.values()))
        # 计算加权平均 Wasserstein 距离
        average_distance = sum(wasserstein_distances[key] * (lengths[key] / sum(lengths.values())) for key in keys)
        return average_distance

    # def distance.jensenshannon(p, q):
    #     """ 计算KL散度 """
    #     return entropy(p, q)

    #"["trip_kind", "end_index","start_index","label","battery_capacity","weekday",'month']"
    def compare_variable_univ(df1, df2, discrete_vars, continuous_vars):
        # KL Divergence for discrete variables
        kl = []
        all_distance={}
        for var in discrete_vars:
            if var=="battery_capacity":
                df=pd.DataFrame({"battery_capacity":[22,25,35,37.8,48.3]})
            else:
                min_x=int(min(df1[var].min(), df2[var].min()))
                max_x=int(max(df1[var].max(), df2[var].max()))
                df=pd.DataFrame({var:range(min_x,max_x+1)})
            o=df.merge(df1[var].value_counts(),on=var,how='left')['count'].fillna(0)
            s=df.merge(df2[var].value_counts(),on=var,how='left')['count'].fillna(0)
            p = (o)/o.sum()
            q = (s)/s.sum()
            jd=js(p, q)
            kl.append(jd)
            all_distance[var]=jd
        # Wasserstein Distance for continuous variables
        wasserstein_distances = []
        for var in continuous_vars:
            if var=='start_hour':
                x1=df1[var].dropna()/24
                #x1=(x1-x1.mean())/x1.std()
                x2=df2[var].dropna()/24
                max_x=max([max(x1),max(x2)])
                min_x=min([min(x2),min(x2)])
                x1= normalize_min_max(x1, min_x, max_x)
                x2= normalize_min_max(x2, min_x, max_x)
                #x2=(x2-x2.mean())/x2.std()
                #random_samples = np.random.uniform(low=0, high=1,size=len(x1))
                distance = wasserstein_distance(x1, x2)
            else:
                x1=df1[var].dropna()
                #x1=(x1-x1.mean())/x1.std()
                x2=df2[var].dropna()
                #x2=(x2-x2.mean())/x2.std()
                max_x=max([max(x1),max(x2)])
                min_x=min([min(x2),min(x2)])
                x1= normalize_min_max(x1, min_x, max_x)
                x2= normalize_min_max(x2, min_x, max_x)
                #x2=(x2-x2.mean())/x2.std()
                #random_samples = np.random.uniform(low=0, high=1,size=len(x1))
                distance = wasserstein_distance(x1, x2)
            wasserstein_distances.append(distance)
            all_distance[var]=distance
        rho1=np.sum(list(all_distance.values()))
        # Calculate average of each type
        #avg_distance.jensenshannon = np.mean(distance.jensenshannons) if distance.jensenshannons else 0
        #avg_wasserstein_distance = np.mean(wasserstein_distances) if wasserstein_distances else 0
        return all_distance,rho1


    # original_data['start_time']=pd.to_datetime(original_data['start_time'])
    # original_data['stay']=(original_data['duration_stay']-(original_data['duration']/60))//24
    # original_data['start_soc']/=100
    # #original_data['end_soc']/=100
    # original_data=original_data[['seq_code','start_time','trip_kind','start_index','energy_change','duration','distance','stay','start_soc','start_hour', 'weekday', 'label', 'battery_capacity']]
    # original_data['trip_kind']=original_data['trip_kind'].apply(lambda x:(x=='D')*1.0)
    # original_data['energy_change']=(original_data['energy_change']*(original_data['trip_kind']==0))-(original_data['energy_change']*(original_data['trip_kind']==1))
    # original_data['soc_change']=original_data['energy_change']/original_data['battery_capacity']
    # original_data['end_soc']=original_data['start_soc']+original_data['soc_change']
    # original_data['soc_change']=original_data['soc_change']*(1-original_data['trip_kind'])-original_data['soc_change']*original_data['trip_kind']
    # original_data['end_index']=original_data['start_index'].shift(-1).fillna(method="ffill")
    sample_data['battery_capacity']=sample_data['battery_capacity'].round(1)
    original_data['charge_electricity']=(original_data['end_soc']-original_data['start_soc'])*original_data['battery_capacity']
    sample_data['charge_electricity']=(sample_data['end_soc']-sample_data['start_soc'])*sample_data['battery_capacity']
    # original_data['duration']/=60


    #original_data['month']=((original_data['start_time'].apply(datetime_to_fract)))//1+1
    #sample_data['duration']*=60
    #original_data['stay']=original_data['stay']//1
    #original_data['start_hour']=original_data['start_hour']
    sample_data['start_hour']*=24
    original_data['start_hour']*=24
    #samle_data['end_soc']*=100
    #sample_data['month'] = sample_data['month'] % 12 + 1  # Adjust month range from 0-11 to 1-12


    discrete_vars=["trip_kind", "end_index","start_index","label","battery_capacity","weekday",'month',"stay"]
    continuous_vars=["start_hour","distance","duration","end_soc","start_soc"]
    _,rho1=compare_variable_univ(original_data,sample_data, discrete_vars, continuous_vars)


    original_data['speed']=original_data['distance']/(original_data['duration'])
    sample_data['speed']=sample_data['distance']/(sample_data['duration'])
    # 筛选速度在 0 到 120 之间的数据
    filtered_original = original_data[(original_data['trip_kind']==1) &(original_data['speed'] >1) & (original_data['speed'] <= 150)]
    filtered_sample = sample_data[(sample_data['trip_kind']==1)&(sample_data['speed'] >1) & (sample_data['speed'] <= 150)]
    speed_distance=group_average_wdistance(filtered_original.groupby(['battery_capacity', 'label'])['speed'].apply(list),filtered_sample.groupby(['battery_capacity', 'label'])['speed'].apply(list))
    print("Speed: "+str(speed_distance))


    # 计算每个共同分组的KL散度
    all_combinations = pd.DataFrame([(i, j) for i in range(252) for j in range(252)], columns=['start_index', 'end_index'])
    od_pair=original_data[original_data['trip_kind']==0].groupby(['start_index','end_index']).agg('count')['start_hour'].reset_index()
    result1 = pd.merge(all_combinations, od_pair, on=['start_index', 'end_index'], how='left').fillna(0)
    result1['start_hour']=result1['start_hour']
    result1['start_hour']=(result1['start_hour'])/result1['start_hour'].sum()

    #all_combinations = pd.DataFrame([(i, j) for i in range(252) for j in range(252)], columns=['start_index', 'end_index'])
    od_pair=sample_data[sample_data['trip_kind']==0].groupby(['start_index','end_index']).agg('count')['start_hour'].reset_index()
    result2 = pd.merge(all_combinations, od_pair, on=['start_index', 'end_index'], how='left').fillna(0)
    result2['start_hour']=result2['start_hour']
    result2['start_hour']=(result2['start_hour'])/result2['start_hour'].sum()
    od_pair_distance=js(result1['start_hour'],result2['start_hour'])
    #average_distance_od = np.mean(list(distance.jensenshannons.values()))

    print("OD_pair: "+str(od_pair_distance))


    original_data['ecr']=(original_data['start_soc']-original_data['end_soc'])*original_data['battery_capacity']/original_data['distance']
    sample_data['ecr']=(sample_data['start_soc']-sample_data['end_soc'])*sample_data['battery_capacity']/sample_data['distance']
    filtered_original = original_data[(original_data['trip_kind']==1) &(original_data['ecr'] >0) & (original_data['ecr'] <= 2)]
    filtered_sample = sample_data[(sample_data['trip_kind']==1) &(sample_data['ecr'] >0) & (sample_data['ecr'] <= 2)]
    ecr_distance=group_average_wdistance(filtered_original.groupby(['battery_capacity', 'label'])['ecr'].apply(list),filtered_sample.groupby(['battery_capacity', 'label'])['ecr'].apply(list))
    print("ECR: "+str(ecr_distance))


    def categorize_start_hour(start_hour):
        if start_hour >= 6 and start_hour < 12:
            return "6-12"
        elif start_hour >= 12 and start_hour < 18:
            return "12-18"
        elif start_hour >= 18 and start_hour < 24:
            return "18-24"
        else:
            return "0-6"
    hour_intervals = ['6-12', '12-18', '18-24', '0-6']
    original_data['hour_interval']=original_data['start_hour'].agg(categorize_start_hour)
    sample_data['hour_interval']=sample_data['start_hour'].agg(categorize_start_hour)


    end_indexes = np.arange(0, 252)  # end_index 从 0 到 251
    # 使用笛卡尔乘积创建完整的 DataFrame
    all_combinations = pd.DataFrame(product(hour_intervals, end_indexes), columns=['hour_interval', 'end_index'])

    filtered_original=original_data[original_data['trip_kind']==1].groupby(['hour_interval','end_index']).agg('count')['start_hour'].reset_index()
    filtered_original = all_combinations.merge(filtered_original[['hour_interval','end_index','start_hour']], on=['hour_interval', 'end_index'], how='left').fillna(0)
    filtered_original['start_hour']=filtered_original['start_hour']+1
    filtered_original['start_hour']/=filtered_original['start_hour'].sum()

    filtered_sample=sample_data[sample_data['trip_kind']==1].groupby(['hour_interval','end_index']).agg('count')['start_hour'].reset_index()
    filtered_sample = all_combinations.merge(filtered_sample[['hour_interval','end_index','start_hour']], on=['hour_interval', 'end_index'], how='left').fillna(0)
    filtered_sample['start_hour']=filtered_sample['start_hour']+1
    filtered_sample['start_hour']/=filtered_sample['start_hour'].sum()
    average_distance_spatio_drive=js(filtered_original['start_hour'],filtered_sample['start_hour'])
    print("Spatio-temp-drive-event: "+str(average_distance_spatio_drive))


    original_data['charge_power']=(original_data['end_soc']-original_data['start_soc'])*original_data['battery_capacity']/original_data['duration']
    sample_data['charge_power']=(sample_data['end_soc']-sample_data['start_soc'])*sample_data['battery_capacity']/sample_data['duration']
    filtered_original = original_data[(original_data['trip_kind'] == 0) &(original_data['charge_power'] >0.5) & (original_data['charge_power'] <= 120)]
    filtered_sample = sample_data[(sample_data['trip_kind'] == 0) &(sample_data['charge_power'] >0.5) & (sample_data['charge_power'] <= 120)]
    charge_power_distance=group_average_wdistance(filtered_original.groupby(['battery_capacity', 'label'])['charge_power'].apply(list),filtered_sample.groupby(['battery_capacity', 'label'])['charge_power'].apply(list))
    print("Charge_power: "+str(charge_power_distance))


    filtered_original = original_data[original_data['trip_kind']==0&(original_data['duration'] >0.01) & (original_data['duration'] <= 40)]
    filtered_sample = sample_data[(sample_data['trip_kind']==0) & (sample_data['duration'] >0.01) & (sample_data['duration'] <= 40)]
    charge_duration_distance=group_average_wdistance(filtered_original.groupby(['battery_capacity', 'label'])['duration'].apply(list),filtered_sample.groupby(['battery_capacity', 'label'])['duration'].apply(list))
    print("Charge_duration: "+str(charge_duration_distance))


    end_indexes = np.arange(0, 252)  # end_index 从 0 到 251
    # 使用笛卡尔乘积创建完整的 DataFrame

    all_combinations = pd.DataFrame(product(hour_intervals, end_indexes), columns=['hour_interval', 'end_index'])

    filtered_original=original_data[original_data['trip_kind']==0].groupby(['hour_interval','end_index']).agg('count')['start_hour'].reset_index()
    filtered_original = all_combinations.merge(filtered_original[['hour_interval','end_index','start_hour']], on=['hour_interval', 'end_index'], how='left').fillna(0)
    filtered_original['start_hour']=filtered_original['start_hour']+1
    #print(filtered_original['start_hour'])
    filtered_original['start_hour']/=filtered_original['start_hour'].sum()

    filtered_sample=sample_data[sample_data['trip_kind']==0].groupby(['hour_interval','end_index']).agg('count')['start_hour'].reset_index()
    filtered_sample = all_combinations.merge(filtered_sample[['hour_interval','end_index','start_hour']], on=['hour_interval', 'end_index'], how='left').fillna(0)
    filtered_sample['start_hour']=filtered_sample['start_hour']+1
    filtered_sample['start_hour']/=filtered_sample['start_hour'].sum()
    average_distance_spatio_charge=js(filtered_original['start_hour'],filtered_sample['start_hour'])
    print("Spatio-temp-charge-event: "+str(average_distance_spatio_charge))


    filtered_original = original_data[(original_data['trip_kind'] == 0) &(original_data['charge_electricity'] >0) & (original_data['charge_electricity'] <= 60)]
    filtered_sample = sample_data[(sample_data['trip_kind'] == 0) &(sample_data['charge_electricity'] >0) & (sample_data['charge_electricity'] <= 60)]
    average_distance_spatio_charge_elect=group_average_wdistance(filtered_original.groupby(['battery_capacity', 'label'])['charge_electricity'].apply(list),filtered_sample.groupby(['battery_capacity', 'label'])['charge_electricity'].apply(list))
    print("Spatio-temp-charge-elect: "+str(average_distance_spatio_charge_elect))
    #original_data[original_data['trip_kind']==0][['hour_interval','end_index','charge_electricity']].groupby(['hour_interval','end_index']).agg(list)


    rho2=(average_distance_spatio_charge_elect+average_distance_spatio_drive+charge_power_distance+average_distance_spatio_charge+\
            ecr_distance+od_pair_distance+speed_distance+charge_duration_distance)

    return rho1,rho2