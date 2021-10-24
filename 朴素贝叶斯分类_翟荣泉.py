import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split#训练集测试集拆分





moons = make_moons()  #加载数据集
circles= make_circles()
classify=make_classification()




print(moons)#数据初探





print(circles)





print(classify)





print(classify[0].shape)#有20个特征，画图的话只能选两个





dfmoons = pd.DataFrame(moons[0],columns=['feature1','feature2'])
dfmoons['label'] = moons[1]

print(dfmoons)

datamoons = np.array(dfmoons)

Xmoons=datamoons[:, :-1]
ymoons=datamoons[:, -1]

print(Xmoons)

print(ymoons)

Xmoons_train, Xmoons_test, ymoons_train, ymoons_test = train_test_split(Xmoons, ymoons, test_size=0.2)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()#构造高斯贝叶斯分类实例
clf.fit(Xmoons_train, ymoons_train)#拟合模型

print(clf.score(Xmoons_test, ymoons_test))#模型评估





#和上面的类似
dfcircles = pd.DataFrame(circles[0],columns=['feature1','feature2'])
dfcircles['label'] = circles[1]

print(dfcircles)

datacircles = np.array(dfcircles)

Xcircles=datacircles[:, :-1]
ycircles=datacircles[:, -1]

print(Xcircles)

print(ycircles)

Xcircles_train, Xcircles_test, ycircles_train, ycircles_test = train_test_split(Xcircles, ycircles, test_size=0.2)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(Xcircles_train, ycircles_train)

print(clf.score(Xcircles_test, ycircles_test))





#和上面的类似，不过有20个特征
dfclassify = pd.DataFrame(classify[0],columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
dfclassify['label'] =classify[1]

print(dfclassify)

dataclassify = np.array(dfclassify)

Xclassify=dataclassify[:, :-1]
yclassify=dataclassify[:, -1]

print(Xclassify)

print(yclassify)

Xclassify_train, Xclassify_test, yclassify_train, yclassify_test = train_test_split(Xclassify, yclassify, test_size=0.2)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(Xclassify_train, yclassify_train)

print(clf.score(Xclassify_test, yclassify_test))





from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

###可视化###

h = .02  # 画图的mesh的步长

names = ["NaiveBayes"]#模型名称

classifiers = [ GaussianNB()]#分类器名称

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)#划分数据集
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=3),
            make_circles(noise=0.2, factor=0.5, random_state=15),
            linearly_separable
            ]

figure = plt.figure(figsize=(15,15))
i = 1
# 在这三个数据集上循环
for ds_cnt, ds in enumerate(datasets):
    # 划分数据集
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=1)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 先画input数据图像
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # 画训练集
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # 画测试集
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # 对分类器迭代，如果后续想把后三个加进去的话可以在classifier中加入
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # 画出决策边界，给每一个点分配一种颜色
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # 画结果
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # 画训练集点
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # 画测试集点
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),#添加score（分类结果）
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()



