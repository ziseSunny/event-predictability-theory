#本代码是为了求解discrepancy和fzt，本代码针对第1类数据
import pandas as pd
import tqdm
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing as mp
from functools import partial

#读取对应的文件
data = pd.read_pickle('/Users/duanjingyuan/Documents/事件可预测性论文/precessed_data/social_media/managed_data.pkl')
#从中提取需要使用的数据（暂时提取4列数据用于测试）
extract_columns = ['message_id','message_label','total_num','afore_num','next_num','window_num','popular','increased']
#从中筛选对应的信息
message_data = data[['message_id','message_label']].drop_duplicates(subset='message_id')
category1_ids = data[data['message_label']==0]['message_id'].drop_duplicates().tolist()
category2_ids = data[data['message_label']==1]['message_id'].drop_duplicates().tolist()
category3_ids = data[data['message_label']==2]['message_id'].drop_duplicates().tolist()
#从中采样信息id
#sample_ids = np.random.choice(a=category1_ids,size=2,replace=False)
sample_id = category1_ids[0]
#定义约束量Cs
Cs = np.linspace(0.01,10000,100)
#定义其余的求解条件
features_X = extract_columns[3:-1]
feature_Y = extract_columns[-1]
kernel = 'rbf'
#提取数据中的message_ids
message_ids = data['message_id'].drop_duplicates().tolist()
print('finished!')

#生成权重向量q
def produce_q(length,function):
    #定义function缺失的情况（此时默认生成等权重）
    if function == "":
        q = (1/length)*np.ones((length,1))
    else:
        q = function(length)
    return q

#求解误差函数f的值
def compute_f(true_Y,predict_Y):
    f = accuracy_score(true_Y,predict_Y)
    return f

#求解f(z_t)（给定SVM的条件：kernel和C，以及求解的时刻（timestamp））
def compute_fzt(data,timestamp,features_X,feature_Y,kernel,C):
    #创建对应的SVC
    SVC = svm.SVC(kernel=kernel,C=C)
    #提取对应的数据
    dataX = data[features_X]
    dataY = data[feature_Y]
    #抽取对应的时间数据
    data_X = dataX.values[:timestamp]
    data_Y = dataY.values[:timestamp]
    #创建对应的训练集和测试集
    if timestamp == 1: #构建样例
        train_X = [[0,0,0,0],[1,0,1,0]]
        train_Y = np.array([[0],[1]])
    else:
        train_X = data_X[:-1].tolist()
        train_Y = np.array([[data_Y[i]] for i in range(len(data_Y)-1)])
        #如果只有一个类别
        if sum(data_Y[:-1]) == len(train_Y) or sum(data_Y[:-1]) == 0:
            train_X.append([0,0,0,0])
            train_X.append([1,0,1,0])
            train_Y = np.append(train_Y,[[0]],axis=0)
            train_Y = np.append(train_Y,[[1]],axis=0)
    #构建测试集
    test_X = [data_X[-1]]
    test_Y = [data_Y[-1]]
    # 训练SVC
    SVC.fit(train_X,train_Y.ravel())
    #预测h
    pred_Y = SVC.predict(test_X)
    #求解f(z_t)
    fzt = accuracy_score(test_Y,pred_Y)
    return fzt

#求解对应的取值
def compute_bases(repeat_times,data,time_interval):
    # 计算数据集的时间窗口数量
    num_intervals = int(len(data) // time_interval) + 1
    blocks = [data[i:i + time_interval] for i in range(0, len(data), time_interval)]
    results = pd.DataFrame()  # 存储重复实验的各区间的结果
    results_columns = ['r','C']
    for index in range(num_intervals):
        results_columns.append('fzt1_%d'%(index+1))
        results_columns.append('qfzts_%d'%(index+1))
    # 进行重复实验
    total_rs = [] #存储rs
    total_Cs = [] #存储Cs
    total_fzt1s = [] #存储f(z_T+1)
    total_qfzts = [] #存储q*f(z_t)s
    for r in tqdm.trange(len(repeat_times)):
        repeat = repeat_times[r]
        for i in range(len(Cs)):
            total_rs.append(repeat)
        for i in tqdm.trange(len(Cs)):
            C = Cs[i] #模型的复杂度约束
            total_Cs.append(C)
            sub_fzt1s = []
            sub_qfzts = []
            for j in tqdm.trange(num_intervals):
                fzts = []
                sub_data = blocks[j]
                q = produce_q(time_interval-1,"") #生成权重向量q
                #确定计算时间长度
                for k in range(1,time_interval):
                    time_end = k
                    fzt = compute_fzt(sub_data,time_end,features_X,feature_Y,kernel,C) #计算对应的fzt
                    fzts.append(1-fzt) #存储fzt
                qfzts = np.dot(q[:,0],np.array(fzts))
                sub_qfzts.append(qfzts)
                fzt1 = compute_fzt(sub_data,time_interval,features_X,feature_Y,kernel,C) #计算对应的f(z_t+1)
                sub_fzt1s.append(1-fzt1)
            total_qfzts.append(sub_qfzts)
            total_fzt1s.append(sub_fzt1s)
    #存储信息
    results['r'] = total_rs
    results['C'] = total_Cs
    total_qfzts = np.array(total_qfzts)
    total_fzt1s = np.array(total_fzt1s)
    for i in range(num_intervals):
        results['fzt1_%d'%(i+1)] = total_fzt1s[:,i]
        results['qfzt_%d'%(i+1)] = total_qfzts[:,i]
    return results

#定义discrepancy的计算
def compute_discrepancy(bases,data,time_interval,repeat_time):
    num_intervals = int(len(data) // time_interval) + 1
    #对每一个interval求解对应的值
    computation = bases.copy()
    for i in range(num_intervals):
        computation['discrepancy_%d'%(i+1)] = np.abs(computation['fzt1_%d'%(i+1)] - computation['qfzt_%d'%(i+1)])
    #对每一个C值求均值
    r_columns = ['r','C']
    for i in range(num_intervals):
        r_columns.append('discrepancy_%d'%(i+1))
    r_extracted = computation[r_columns]
    r_values = []
    for i in range(repeat_time):
        extracted = r_extracted[r_extracted['r']==i]
        interval_values = []
        for j in range(num_intervals):
            interval_value = np.max(extracted['discrepancy_%d'%(j+1)])
            interval_values.append(interval_value)
        r_values.append(interval_values)
    r_values = np.array(r_values)
    #存储结果
    rs = [i for i in range(repeat_time)]
    d_columns = r_columns.remove('C')
    discrepancy = pd.DataFrame(columns=d_columns)
    discrepancy['r'] = rs
    for i in range(num_intervals):
        discrepancy['discrepancy_%d'%(i+1)] = r_values[:,i]
    return discrepancy

repeat_time = 50  #重复实验的次数
repeat_indexes = [i for i in range(repeat_time)]
'''
extract_data = data[data['message_id']==sample_id].reset_index()[extract_columns]
bases = compute_bases(repeat_time,extract_data,200)
discrepancy = compute_discrepancy(bases,extract_data,200,repeat_time)
'''

if __name__ == '__main__':
    extract_data = data[data['message_id']==sample_id].reset_index()[extract_columns]
    #定义默认条件
    params = partial(compute_bases,data=extract_data,time_interval=200)
    num_cpu = int(mp.cpu_count())
    print('computing discrepancy...')
    #自行构建blocks
    block_size = int(repeat_time // num_cpu)
    blocks = [repeat_indexes[i:i+block_size] for i in range(0,repeat_time,block_size)]
    with mp.Pool(num_cpu) as pool:
        data_part = pool.map(params, blocks)
        pool.close()
        pool.join()
    bases_results = pd.concat(data_part)
    print('storing bases...')
    bases_results.to_csv('/Users/duanjingyuan/Documents/事件可预测性论文/precessed_data/social_media/' + 'bases_results1_'+str(sample_id)+'_v1.csv')
    print('computing discrepancy...')
    discrepancy = compute_discrepancy(bases_results,extract_data,200,repeat_time)
    print('storing discrepancy...')
    discrepancy.to_csv('/Users/duanjingyuan/Documents/事件可预测性论文/precessed_data/social_media/' + 'discrepancy_results1_'+str(sample_id)+'_v1.csv')
    print('finish!')