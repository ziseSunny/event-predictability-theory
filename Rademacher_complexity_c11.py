#本代码是为了求解数据集的Rademacher复杂度和M，全部使用SVM进行预测（预测类别标签），本代码针对的是第1类数据
import pandas as pd
from sklearn import svm
import random
import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing as mp
import tqdm
from functools import partial #设置多进程池的默认参数

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
sigma_times = 100
features_X = extract_columns[3:-1]
feature_Y = extract_columns[-1]
kernel = 'rbf'
#提取数据中的message_ids
message_ids = data['message_id'].drop_duplicates().tolist()
print('finished!')

#创建函数，该函数用于生成随机向量sigma，该向量包含{-1,1}的均匀随机变量（概率各为0.5）
def produce_sigma(length):
    domain = [-1,1] #只有2个值可取，分别为-1和1
    sigma = []
    for i in range(length):
        current_prob = random.uniform(0,1) #生成[0,1]的均匀分布
        if current_prob <= 0.5:
            value = domain[0]
        else:
            value = domain[1]
        sigma.append(value)
    return sigma

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

extract_data = data[data['message_id']==message_ids[0]].reset_index()[extract_columns]

#创建函数，计算数据集对应的各个时间窗口的Rademacher复杂度（对sigma（影响二叉树z）和C（影响h，进而影响f）进行遍历），输入的time_interval表示时间窗口大小
def compute_Rademacher(repeat_times,data,time_interval):
    # 计算数据集的时间窗口数量
    num_intervals = int(len(data) // time_interval) + 1
    blocks = [data[i:i + time_interval] for i in range(0, len(data), time_interval)]
    results = pd.DataFrame()  # 存储重复实验的各区间Rademacher复杂度
    results_columns = [str(i+1) for i in range(num_intervals)] #记录间隔编号
    results_columns.append('M')
    #存储计算结果
    total_Rademachers = []
    total_Ms = []
    for r in tqdm.trange(repeat_times):
        Rademachers = []
        current_Ms = []
        #计算对应的信息
        for i in tqdm.trange(num_intervals):
            sub_data = blocks[i]
            sums = np.zeros((sigma_times,len(Cs)))
            sub_Ms = []
            for j in tqdm.trange(sigma_times):
                sigma = produce_sigma(time_interval-1) #生成随机向量sigma（该变量确定完全二叉树）
                sub_M1s = []
                for k in range(len(Cs)):
                    C = Cs[k] #模型的复杂度约束
                    fzts = []
                    q = produce_q(time_interval-1,"") #生成权重向量q
                    #确定计算时间长度
                    for l in range(1,time_interval):
                        time_end = l
                        fzt = compute_fzt(sub_data,time_end,features_X,feature_Y,kernel,C)
                        fzts.append(1-fzt) #损失函数为预测失误率（fzt为预测正确率）
                    #求解最大的f，即为M
                    sub_M1 = np.max(fzts)
                    sub_M1s.append(sub_M1)
                    #求解对应的sum
                    sub_sum = np.multiply(np.array(sigma),q[:,0])
                    sum = np.dot(sub_sum,np.array(fzts))
                    sums[j,k] = sum #记录该C值条件下的子和（前t项之和）
                sub_M = np.max(sub_M1s)
                sub_Ms.append(sub_M)
            #求解该数据序列对应的Rademacher复杂度
            #先求每一行的最大值
            fmaxes = np.amax(sums,axis=1) #对行求最大值
            #再求行最大值中的最大值
            Rademacher = np.amax(fmaxes) #该值即为该message_id对应的Rademacher
            Rademachers.append(Rademacher)
            M = np.max(sub_Ms)
            current_Ms.append(M)
        #将得到的信息进行汇总
        total_Rademachers.append(Rademachers)
        current_M = np.max(current_Ms)
        total_Ms.append(current_M)
    #将total_Rademacher的类型进行转换
    total_Rademachers = np.array(total_Rademachers)
    #存储对应的信息
    for index in range(num_intervals):
        results[str(index+1)] = total_Rademachers[:,index]
    results['M'] = total_Ms
    return results

repeat_time = 50  #重复实验的次数
repeat_indexes = [i for i in range(repeat_time)]
#extract_data = data[data['message_id']==sample_id].reset_index()[extract_columns]

#rademacher = compute_Rademacher(repeat_time,extract_data,200)
#print('finish!')

if __name__ == '__main__':
    extract_data = data[data['message_id']==sample_id].reset_index()[extract_columns]
    #定义默认条件
    params = partial(compute_Rademacher,data=extract_data,time_interval=200)
    num_cpu = int(mp.cpu_count())
    print('computing Rademacher...')
    #自行构建blocks
    block_size = int(repeat_time // num_cpu + 1)
    blocks = [len(repeat_indexes[i:i+block_size]) for i in range(0,repeat_time,block_size)]
    with mp.Pool(num_cpu) as pool:
        data_part = pool.map(params, blocks)
        pool.close()
        pool.join()
    Rademacher_results = pd.concat(data_part)
    print('storing...')
    Rademacher_results.to_csv('/Users/duanjingyuan/Documents/事件可预测性论文/precessed_data/social_media/' + 'Rademacher_results1_'+str(sample_id)+'.csv')
    print('finish!')