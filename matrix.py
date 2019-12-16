from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import codecs

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=8, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

# # classes表示不同类别的名称，比如这有6个类别
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
#
# random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
# y_true = random_numbers.copy()  # 样本实际标签
# random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
# y_pred = random_numbers  # 样本预测标签
y_true = []
y_pred = []
f = codecs.open('./pred.txt','r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    gt = line.split(',')[-2]
    y_true.append(gt)
    pred = line.split(',')[-1]
    y_pred.append(pred)

# print(y_pred)
# 获取混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
