# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 23:40:05 2021
FR-AGCN的混淆矩阵
@author: ROG
"""

import pickle
import numpy as np
from tqdm import tqdm

# 读取测试集所有动作编号及其真实标签，记为label
label = open (r'E:\PyTorchTest\FRAGCN\data\ntu\xsub\val_label.pkl','rb')
label = np.array(pickle.load(label))

# 读取测试集所有动作编号，及对应的softmax预测的概率，记为r1, r2, r3, r4
r1 = open(r'E:\PyTorchTest\FRAGCN\work_dir\ntu\xsub\agcn_test_forward\epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())

r2 = open(r'E:\PyTorchTest\FRAGCN\work_dir\ntu\xsub\agcn_test_reverse\epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())

r3 = open(r'E:\PyTorchTest\FRAGCN\work_dir\ntu\xsub\agcn_test_forward_bone\epoch1_test_score.pkl', 'rb')
r3 = list(pickle.load(r3).items())

r4 = open(r'E:\PyTorchTest\FRAGCN\work_dir\ntu\xsub\agcn_test_reverse_bone\epoch1_test_score.pkl', 'rb')
r4 = list(pickle.load(r4).items())

# 新建list，用于存放预测标签和真实标签
pred  = []
true  = []

for i in tqdm(range(len(label[0]))):
# 一个接着一个的动作序列，获取标签，i为动作编号
    _, l = label[:, i]      # l为真实标签，需要注意转化为int，方便添加进list中
    _, r11 = r1[i]          # r11为对每个动作预测的60个softmax分数
    _, r22 = r2[i] 
    _, r33 = r3[i]
    _, r44 = r4[i]
    
    r   = r11 + r33 + r22 + r44
    r   = np.argmax(r)
    
    true.append(int(l))           # 按动作编号，逐个添加真实标签
    pred.append(r.tolist())
    
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 50,
}

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(60, 60), dpi=80)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title, font1)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label', font1)
    plt.xlabel('Predict label', font1)
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png', bbox_inches='tight')
    plt.show()

# classes表示不同类别的名称，比如这有60个类别
classes = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25',
            'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50',
            'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60']

# 获取混淆矩阵
cm = confusion_matrix(true, pred)

# print(cm_normalized)
# plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, 'Fig10.jpeg', title='Confusion matrix')
