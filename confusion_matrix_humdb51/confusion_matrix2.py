# -*- coding: utf-8 -*-

# python 2.7
# 根据.npy文件和label文件画图，黑白模式
#approach 2
print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# ############################################################################
# # import some data to play with
# iris = datasets.load_iris() # gain data
#
# X = iris.data               #real
# y = iris.target             #predict
# # print(X)
# # print(type(X))
# # print(y)
# # print(type(y))
# class_names = iris.target_names #the number of class
# print(type(class_names))
# print(class_names)
# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results
# classifier = svm.SVC(kernel='linear', C=0.01)
# y_pred = classifier.fit(X_train, y_train).predict(X_test) #training
#############################################################################
#load labels.
labels = []
file = open('hmdb_label.txt', 'r')
lines = file.readlines()
for line in lines:
    labels.append(line.split()[0])
file.close()
class_names = np.array(labels)
#############################################
#load confusion matrix from npy-file
cnf_matrix = np.load("confusion_matrix.npy")
#############################################################################
print(cnf_matrix)
print(type(cnf_matrix))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #找到
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #for i in range(len(cm1))

    print(cm)
    print(type(cm))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)  #增加标题
    plt.colorbar()  #增加侧边条
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,fontsize=8.5,fontweight='medium')##################x坐标字体

    alphabet1 =[]
    for i in range(len(cm1)):
        b=(cm1[i][i])
        alphabet1.append('('+str('%.2f' % b)+')'+classes[i])
    plt.yticks(tick_marks, alphabet1,fontsize=8.5,fontweight='medium')##################y坐标字体

    fmt = '.2f' if normalize else '.0f'
    print(fmt)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=8,fontweight='semibold')

    plt.tight_layout()
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)

# # Compute confusion matrix
# print("#############3")
# print(y_test)
# print(type(y_test))
# print(y_pred)
# cnf_matrix = confusion_matrix(y_test, y_pred)
############################################
np.set_printoptions(precision=2) #Number of digits of precision for floating point output (default 8).

# Plot non-normalized confusion matrix
plt.figure() #Draw multiple graphs
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, HMDB51')
plt.savefig('confusion_matrix2.png', format='png')
# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# plt.savefig('Normalized-confusion-matrix.png', format='png')

plt.show()
