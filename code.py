# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:57:37 2024

@author: lenovo
"""
#%%导入
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import torch
import torch.optim as optim
from lightgbm import LGBMClassifier
import sklearn
import random
import pandas as pd
import os
import scipy.io as sio
from scipy import sparse


# NOTE: Θ中的置信度指的是分类器的准确率, 其形状为(c × k);
# Gj(x_i)中的元素是分类器Gj将x_i预测为各个类别的概率
#%%函数
def Train(X, y, k, deep = 10):
    """
    训练k个基分类器

    参数:
    X: 特征数据
    y: 标签数据
    k: 基分类器的个数
    deep: 基分类器的最大深度

    返回:
    classifiers: 训练好的基分类器列表
    G: 预测结果矩阵
    accuracies: 基分类器的准确率列表
    """

    #deep是最大深度
    c = len(np.unique(y))  # 类别数目
    n = len(X)

    # 初始化k个分类器和矩阵G
    classifiers = []
    # NOTE: 论文中没有提到G
    G = np.zeros((k * c, n))  # 初始化G矩阵，kc行, n列, k个分类器，每个分类器对应c维one-hot encoded vector
    # NOTE: G中每一列对应一个样本，每c行对应一个分类器对该样本在每个类别上的预测概率，总共有k个分类器，所以是k * c行。
    # G的第i列 = [G1(x_i) -> c维列向量
    #               ,
    #            G2(x_i) -> c维列向量
    #               ,
    #              ...
    #               ,
    #            Gk(x_i) -> c维列向量]
    accuracies = []

    # 训练k个不同参数设置的分类器
    for i in range(k):
        # Part1 Bootstrap Sampling
        # 使用bootstrap抽样创建不同的子样本
        size = round(len(X)*0.95)       # 抽取出占原始数据集95%dax大小的子样本集

        # 从原始数据集中有放回地随机抽取 size 个样本的索引，形成子样本的索引列表bootstrap_indices。
        # replace = True: 表示抽样时是有放回的，意味着每次抽取后，样本都会放回抽样池，可能被再次抽取到。这使得索引列表 bootstrap_indices 中的元素可能重复。
        bootstrap_indices = np.random.choice(len(X), size = size, replace = True)

        # 从原始数据集 X 和标签集 y 中提取对应的样本和标签，构成新的子样本集。
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]

        # Part2 Training Classifiers
        # 根据i的不同余数选择不同的分类器
        classifier = DecisionTreeClassifier(max_depth = deep, random_state=1)

        # 训练分类器
        classifier.fit(X_bootstrap, y_bootstrap)
        classifiers.append(classifier)

        # 预测结果
        y_pred = classifier.predict(X) - 1

        # Part3 Constructing G Matrix
        # c行, n列: classifier.predict_proba(X) 产生一个形状为(𝑛,𝑐)的矩阵，其中每一行是一个样本在𝑐个类别上的预测概率。
        # NOTE: G中的元素是样本在每个类别上的预测概率
        G[i*c: i*c+c, :] = classifier.predict_proba(X).T

        # 计算分类器的准确率
        accuracy = classifier.score(X, y)
        accuracies.append(accuracy)
        #print(f"基分类器 {i+1} 的准确率：{accuracy:.2f}")

    g = []
    for j in range(n):
        g_j = G[:, j:j+1].reshape(k, c)
        g.append(g_j)

    g = np.hstack(g)

    # 返回所有基分类器、矩阵G和准确率列表
    return classifiers, g, accuracies


def predict_g(classifiers, X, c):
    k = len(classifiers)
    n = len(X)      # 样本数

    g = np.zeros((k, n * c))  # 初始化G矩阵，kc行，n列

    # 对每个基分类器计算预测结果并更新矩阵G
    for i, clf in enumerate(classifiers):
        y_pred = clf.predict(X) - 1
        g[i, :] = clf.predict_proba(X).reshape(1, -1)

    return g


def one_hot(y):
    y = y.astype(int)
    """
    将标签向量 y 转换成 one-hot 编码的矩阵 Y

    参数：
    y : numpy数组，形状为 (n,)，标签向量，包含每个样本的类别标签（从1到c）

    返回：
    Y : numpy数组，形状为 (c, n)，每一列是 y 的相应元素的 one-hot 编码
    """
    n = len(y)  # 样本数
    c = int(np.max(y))  # 类别数，假设类别标签从1到c

    Y = np.zeros((c, n))  # NOTE: 样本是一列一列放置的，行表示类别，列表示样本

    for i in range(n):
        # 第i个样本的标签是y[i]，标签y[i]对应的索引是y[i]-1
        Y[y[i]-1 , i] = 1  # 标签从1开始，所以标签y[i]对应的索引是y[i]-1

    return Y


def init_Theta(accuracies,c):
    """ Theta的形状是c × k的，每个classifier对应一列（共k列），每个类对应一行，表示一个classifier将样本分到该类的置信度，初始化时，把一个分类器将样本分到每个类的置信度全部初始化成该分类器的准确率 """
    # 将向量中的每个元素复制c遍
    # replicated = np.repeat(accuracies, c)

    # 将accuracies(对应于一行k个分类器的准确率)整体复制c遍并竖着堆叠
    stacked = np.tile(accuracies, (c, 1))

    return stacked


def compute_S_I_Theta_g_1(Theta, g):
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=torch.float)
    Theta_g = torch.matmul(Theta, g)
    Theta_g = torch.split(Theta_g, c, dim=1)
    I_Theta_g_1 = torch.hstack(
                    [torch.diagonal(element).reshape(-1, 1) for element in Theta_g]
                  )
    S_I_Theta_g_1 = torch.softmax(I_Theta_g_1, dim=0)

    return S_I_Theta_g_1


def compute_loss(Theta, g, Y, gamma, alpha = 100):
    # Theta的形状是c × k的
    c = Theta.size()[0]
    n = int(g.shape[1] / c)

    #转化成张量
    g = torch.tensor(g, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)


    S_I_Theta_g_1 = compute_S_I_Theta_g_1(Theta, g)

    # 初始化margin和cross-entropy loss
    M = 0
    C = 0

    for idx in range(n):
        M += torch.matmul(Y[:, idx:idx+1].T, S_I_Theta_g_1[:, idx:idx+1])
        C -= torch.matmul(Y[:, idx:idx+1].T, torch.log(S_I_Theta_g_1[:, idx:idx+1]))

    M -= 1 / alpha * torch.sum(
                     torch.logsumexp(alpha * (S_I_Theta_g_1 - Y * S_I_Theta_g_1),
                                     dim=0),
                     )

    L = (C - gamma * M) / n

    return L


def generate_ring_data(num_samples, inner_radius, outer_radius, noise_std):
    # 生成第一个月牙数据（上半圆）
    theta_first = np.linspace(0, np.pi, num_samples)
    x_first = inner_radius * np.cos(theta_first) + noise_std * np.random.randn(num_samples)
    y_first = inner_radius * np.sin(theta_first) + noise_std * np.random.randn(num_samples)

    # 生成第二个月牙数据（下半圆，偏移）
    theta_second = np.linspace(0, np.pi, num_samples)
    x_second = outer_radius * np.cos(theta_second) + inner_radius
    x_second += noise_std * np.random.randn(num_samples)
    y_second = -outer_radius * np.sin(theta_second) + noise_std * np.random.randn(num_samples)

    # 合并两个月牙数据
    x = np.concatenate((x_first, x_second))
    y = np.concatenate((y_first, y_second))

    # 创建标签向量 y，第一个月牙为标签 1，第二个月牙为标签 2
    labels = np.ones(num_samples * 2)
    labels[num_samples:] = 2

    return x, y, labels


def loadmat(path, to_dense = True):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["Y"].astype(np.int32).reshape(-1)

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true


def compute_classifiers_Y_G_accracies(X_train, y_train, k, deep):
    classifiers, g, accuracies = Train(X_train, y_train, k, deep=deep)
    Y = one_hot(y_train)
    # g = torch.tensor(g, dtype=torch.float)

    return classifiers, Y, g, accuracies


def compute_results(X, labels,
                    y_train,
                    X_test, y_test,
                    classifiers, Theta,
                    g, c):
    g_test = predict_g(classifiers, X_test, c)
    g_test = torch.tensor(g_test, dtype=torch.float)

    y_dataset_test= torch.argmax(compute_S_I_Theta_g_1(Theta, g_test), dim=0).numpy() + 1
    acc_dataset_test = accuracy_score(y_dataset_test, y_test)

    y_dataset_train = torch.argmax(compute_S_I_Theta_g_1(Theta, g), dim=0).numpy() + 1
    acc_dataset_train = accuracy_score(y_dataset_train, y_train)

    g_all = predict_g(classifiers, X, c)
    g_all = torch.tensor(g_all,dtype=torch.float)
    y_dataset_all= torch.argmax(compute_S_I_Theta_g_1(Theta, g_all), dim=0).numpy() + 1
    acc_dataset_all = accuracy_score(y_dataset_all, labels)

    return y_dataset_test, acc_dataset_test, y_dataset_train, acc_dataset_train, y_dataset_all, acc_dataset_all




SEED = 1
# python内置种子
random.seed(SEED)
# torch种子
torch.manual_seed(1)
# numpy种子
np.random.seed(1)
# sklearn种子
sklearn.random.seed(SEED)

deep = 9
# 获取所有.mat文件
data_path = r'data'
mat_files = ['BASEHOCK.mat', 'breast_uni.mat', 'chess.mat', 'iris.mat', 'jaffe.mat', 'pathbased.mat', 'RELATHE.mat', 'wine.mat']

mat_files = [os.path.join(data_path, f) for f in mat_files]

# 创建结果DataFrame
columns = ['wrf_train', 'wrf_test', 'wrf_all',
           'rf50_train', 'rf50_test', 'rf50_all',
           'rf100_train', 'rf100_test', 'rf100_all',
           'svc_train', 'svc_test', 'svc_all',
           'xgb_train', 'xgb_test', 'xgb_all',
           'lgbm_train', 'lgbm_test', 'lgbm_all']
results = pd.DataFrame(columns = columns)

# 循环处理每个数据集
for mat_file in mat_files:
    dataset_name = os.path.splitext(os.path.basename(mat_file))[0]
    print(f"Processing dataset: {dataset_name}")

    try:
        # 加载数据
        X, labels, num, dim, c = loadmat(mat_file)
        X = X.astype(np.float32)
        if min(labels) == 0:
            labels = labels + 1
        if min(labels) == -1:
            labels = (labels + 1) / 2 + 1
        c = int(max(labels))
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)


        # Part1 WRF
        k = 10
        gamma = 10
        n = 1000
        loss_array = np.ones(n)

        classifiers, Y, g, accuracies = compute_classifiers_Y_G_accracies(X_train, y_train, k, deep)

        # 计算Theta
        Theta = init_Theta(accuracies, c)
        Theta = torch.tensor(Theta, requires_grad=True, dtype=torch.float)

        # 定义优化器
        optimizer = optim.SGD([Theta], lr=2)

        # 训练过程
        for epoch in range(n):
            optimizer.zero_grad()  # 清零梯度
            loss = compute_loss(Theta, g, Y, gamma)  # 计算损失
            # 使用.item()方法从包含单个元素的数组中提取出该元素的标量值。
            loss_array[epoch] = loss.detach().numpy().item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n}], Loss: {loss.item()}')

        # 获取优化后的参数值
        optimal_Theta = Theta.detach().numpy()

        y_wrf_test, acc_wrf_test, y_wrf_train, acc_wrf_train, y_wrf_all, acc_wrf_all = compute_results(X, labels, y_train, X_test, y_test, classifiers, Theta, g, c)


        # Part2 RF50
        k_rf = 50

        classifiers, Y, g, accuracies = compute_classifiers_Y_G_accracies(X_train, y_train, k_rf, deep)

        # 计算Theta
        Theta = init_Theta(accuracies, c)
        Theta = torch.tensor(Theta, requires_grad=True, dtype=torch.float)

        y_rf50_test, acc_rf50_test, y_rf50_train, acc_rf50_train, y_rf50_all, acc_rf50_all = compute_results(X, labels, y_train, X_test, y_test, classifiers, Theta, g, c)


        # Part3 RF100
        k_rf = 100

        classifiers, Y, g, accuracies = compute_classifiers_Y_G_accracies(X_train, y_train, k_rf, deep)

        # 计算Theta
        Theta = init_Theta(accuracies, c)
        Theta = torch.tensor(Theta, requires_grad=True, dtype=torch.float)

        y_rf100_test, acc_rf100_test, y_rf100_train, acc_rf100_train, y_rf100_all, acc_rf100_all = compute_results(X, labels, y_train, X_test, y_test, classifiers, Theta, g, c)


        # SVC
        svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale', decision_function_shape = 'ovr')

        # 训练模型
        svm_clf.fit(X_train, y_train)

        # 预测
        y_svc_test = svm_clf.predict(X_test)
        # 计算准确率
        acc_svc_test = accuracy_score(y_test, y_svc_test)
        y_svc_train = svm_clf.predict(X_train)
        acc_svc_train = accuracy_score(y_train, y_svc_train)
        y_svc_all = svm_clf.predict(X)
        acc_svc_all = accuracy_score(labels, y_svc_all)


        # xgboost
        xgb_clf = XGBClassifier(max_depth = deep, learning_rate = 0.2, n_estimators = k, objective = 'multi:softmax',  # 或者 'multi:softprob'
            num_class = c)

        # 训练模型
        xgb_clf.fit(X_train, y_train-1)

        # 预测
        y_xgb_test = xgb_clf.predict(X_test)
        # 计算准确率
        acc_xgb_test = accuracy_score(y_test, y_xgb_test+1)
        y_xgb_train = xgb_clf.predict(X_train)
        # 计算准确率
        acc_xgb_train = accuracy_score(y_train, y_xgb_train+1)
        y_xgb_all = xgb_clf.predict(X)
        # 计算准确率
        acc_xgb_all = accuracy_score(labels, y_xgb_all+1)


        # lgbm
        lgbm_clf = LGBMClassifier(max_depth = deep, learning_rate = 0.225, n_estimators = k, objective = 'multiclass', num_class = c)

        # 训练模型
        lgbm_clf.fit(X_train, y_train-1)

        # 预测
        y_lgbm_test = lgbm_clf.predict(X_test)
        # 计算测试集准确率
        acc_lgbm_test = accuracy_score(y_test, y_lgbm_test+1)

        # 预测训练集
        y_lgbm_train = lgbm_clf.predict(X_train)
        # 计算训练集准确率
        acc_lgbm_train = accuracy_score(y_train, y_lgbm_train+1)

        # 预测所有数据
        y_lgbm_all = lgbm_clf.predict(X)
        # 计算所有数据的准确率
        acc_lgbm_all = accuracy_score(labels, y_lgbm_all+1)


            # 存储结果
        results.loc[dataset_name] = [
            acc_wrf_train, acc_wrf_test, acc_wrf_all,
            acc_rf50_train, acc_rf50_test, acc_rf50_all,
            acc_rf100_train, acc_rf100_test, acc_rf100_all,
            acc_svc_train, acc_svc_test, acc_svc_all,
            acc_xgb_train, acc_xgb_test, acc_xgb_all,
            acc_lgbm_train, acc_lgbm_test, acc_lgbm_all
        ]
    except Exception as e:
        print(f"######## Error processing dataset {dataset_name}: {str(e)} ########")
        continue

# 保存结果到CSV
results.to_csv(r"results.csv")
print("Results saved to algorithm_comparison_results.csv")














# 边界
# from matplotlib.colors import ListedColormap
# def boundary(clf,X,labels,Theta = np.array([[1,2],[3,4]]),c = 2):
#     cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
#     cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
#     h = .1
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     data = np.c_[xx.ravel(), yy.ravel()]
#     if sum(sum(Theta)) =  = 10:
#         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z + 1  # 调整预测标签范围到1, 2
#         Z = Z.reshape(xx.shape)

#         plt.figure(figsize = (10, 6))
#         plt.contourf(xx, yy, Z, cmap = cmap_light, alpha = 0.8)
#         plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = cmap_bold, edgecolor = 'k', s = 20)
#         plt.xlim(xx.min(), xx.max())
#         plt.ylim(yy.min(), yy.max())
#         plt.title("XGBoost Decision Boundary")
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.show()
#     else:
#         G = predict_g(clf, data , c)
#         G = torch.tensor(G,dtype=torch.float)
#         Z= torch.argmax(torch.matmul(Theta,G), dim = 0).numpy()+1
#         Z = Z.reshape(xx.shape)
#         plt.figure(figsize = (10, 6))
#         plt.contourf(xx, yy, Z, cmap = cmap_light, alpha = 0.8)
#         plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = cmap_bold, edgecolor = 'k', s = 20)
#         plt.xlim(xx.min(), xx.max())
#         plt.ylim(yy.min(), yy.max())
#         plt.title("XGBoost Decision Boundary")
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.show()
#     return (xx,yy,Z)

#boundary(classifiers,X,labels,Theta,c = 2)
