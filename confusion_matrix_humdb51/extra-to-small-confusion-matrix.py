# -*- coding: utf-8 -*-
#何师兄要写的小矩阵来突出错误
#approach 2
print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
############################################################################

labels = [ 'carwheel','handstand']
class_names = np.array(labels)
cm = [[18,7],[0,29]]
cnf_matrix=np.array(cm)
# #############################################################################
# #load labels.
# labels = []
# file = open('hmdb_label.txt', 'r')
# lines = file.readlines()
# for line in lines:
#     labels.append(line.split()[0])
# file.close()
# class_names = np.array(labels)
# #############################################
# #load confusion matrix from npy-file
# cnf_matrix = np.load("confusion_matrix.npy")
# #############################################################################


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #for i in range(len(cm1))
        
    print(cm)
    print(type(cm))
    plt.imshow(cm, interpolation='nearest', cmap='Greys')  
#     plt.title(title,fontsize=35)      #显示标题
#     plt.colorbar()        #增加一个彩色条
    tick_marks = np.arange(len(classes))
    print(tick_marks)
    print("!!!!!")
    plt.xticks(tick_marks, classes,fontsize=50)##################xzuobiaoziti 长度、内容、旋转、大小fontweight='semibold'
    
    alphabet1 =[]
    for i in range(len(cm1)):
        b=(cm1[i][i])
        alphabet1.append(classes[i])   #为y轴添加精度显示
    plt.yticks(tick_marks, alphabet1,fontsize=50)##################yzuobiaoziti
    
    fmt = '.2f' if normalize else '.0f'
    print(fmt)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),    #显示主图数字值
                 horizontalalignment="center",verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=65)
        
    plt.tight_layout()
#     plt.ylabel('True label',fontsize=35)
#     plt.xlabel('Predicted label',fontsize=35)

#  # Compute confusion matrix
# print("#############3")
# print(y_test)
# print(type(y_test))
# print(y_pred)
# ###########################################
np.set_printoptions(precision=2) #Number of digits of precision for floating point output (default 8).

# Plot non-normalized confusion matrix
plt.figure() #Draw multiple graphs 
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig('small_confusion_matrix-sword-relation.png', format='png')
# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# plt.savefig('Normalized-confusion-matrix.png', format='png')  

plt.show()
