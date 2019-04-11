---
title: 莺尾花的分类
date: 2018-11-01 22:43:05
categories:
- 机器学习
tags:
- Python
- 莺尾花
---


## 写在前面
&#160; &#160; &#160; &#160; 刚刚接触机器学习，参考魏贞原老师的教材，完成了我的第一个机器学习项目即莺尾花分类问题。数据集是含莺尾花三个亚属的分类信息，通过机器学习算法生成一个模型，自动分类新数据到这三个亚属的其中的一个。下面是这个项目实现的步骤：

<!-- more -->
&#160; &#160; &#160; &#160; 
 （1）导入数据。
 （2）概述数据。
 （3）数据可视化。
 （4）评估算法。
 （5）实施预测。
&#160; &#160; &#160; &#160; 
## 导入数据
### 导入类库

导入在项目中要使用的数据使用的类库和方法，代码如下：

    #导入类库
    from pandas import read_csv
    from pandas.plotting import scatter_matrix
    from matplotlib import pyplot
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
  

所有类型的库的输入都不应有错误。如果出错，请暂时停下来配置一个完整的Scipy环境。

### 导入数据集
&#160; &#160; &#160; &#160;我们可以在UCI机器学习仓库下载[莺尾花（Irirs Flower）数据集](http://archive.ics.uci.edu/ml/datasets/Iris)，下载完成后保存在项目的统计目录下或通过链接访问数据。在这里用Pandas来导入数据和对数据进行描述性统计分析，并利用Matplotlib实现数据的可视化。需要注意的是在导入数据时，为每一个数据特征设定了名称，这有助于后面对数据工作的展开，尤其是通过图表展示数据，代码如下：



  ```#导入数据集
url ="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names =['sepal-length', 'sepal-width','petal-length', 'petal-width','class']
dataset = read_csv(url, names=names)
```
## 概述数据
&#160; &#160; &#160; &#160; 我们需要先看一下数据，增加对数据的了解，以便选择合适的算法。我们将从以下几个方面来审查数据：

（1）数据的维度。
（2）查看数据自身。
（3）统计描述所有的数据特征。
（4）数据分类的分布情况。

&#160; &#160; &#160; &#160;不必怛心这会敲很多代码，其实每一种审查方法只有一行代码，这些代码在以后的许多项目中也会用到。

### 数据维度
&#160; &#160; &#160; &#160; 通过审查数据维度，可以对数据集有一个大概的了解，如看一下数据有多少行，多少列。代码如下：

  ```#显示数据维度
print('数据维度： 行：%s, 列: %s' % dataset.shape)
```
将会得到如下执行结果：

`数据维度： 行：150, 列: 5`
### 查看数据自身
&#160; &#160; &#160; &#160; 查看数据自身能让我们很好地理解数据，让我们直观的看到数据的特征、类型，以及数据大概分布的范围。代码如下：

  ```#查看数据的前十行
 print (dataset.head(10))
```
 在这里查看前十行的记录，将会得到如下执行结果：

	    sepal-length  sepal-width     ...       petal-width        class
	0           5.1          3.5     ...               0.2  Iris-setosa
	1           4.9          3.0     ...               0.2  Iris-setosa
	2           4.7          3.2     ...               0.2  Iris-setosa
	3           4.6          3.1     ...               0.2  Iris-setosa
	4           5.0          3.6     ...               0.2  Iris-setosa
	5           5.4          3.9     ...               0.4  Iris-setosa
	6           4.6          3.4     ...               0.3  Iris-setosa
	7           5.0          3.4     ...               0.2  Iris-setosa
	8           4.4          2.9     ...               0.2  Iris-setosa
	9           4.9          3.1     ...               0.1  Iris-setosa

### 统计数据的描述
&#160; &#160; &#160; &#160; 接下来看一下数据在不同分类的分布情况，执行程序后得到的是每个分类数据量的绝对值的数值，查看各类数据的分布是否均衡。

	#分类分布情况
	print (dataset.groupby('class').size())

 这里就是通过前面设定的数据特征名称来查看数据的，执行结果如下：

  ``` class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64 
```
&#160; &#160; &#160; &#160; 这里我们可以看出莺尾花的三个亚属的数据个50条，分布非常均衡。如果分布不均衡可能会影响到模型的精度。故当数据分布不均衡时，需要对其进行处理，调整数据平衡时有以下几种方法：
* **扩大样本数据：** 这是一个容易被忽略的选择，一个更大的数据可能挖掘出更平衡或不同方面提高算法模型的准确度。
* **数据的重新抽样：**  过抽样（复制少数类样本）和欠抽样（删除多数类样本）。当数据量很大时（大于一万条记录）可以考虑欠抽样，当数据量较少时可考虑过抽样。
* **尝试生成人工样本：**  一种简单的方法是从少数类的实例中随机抽取特征属性，生成更多数据。
* **异常检测和变化检测：**  尝试用不同的观点进行思考。异常检测是对罕见事件的检测 。这种思维的转变在于考虑以小类作为异常值类，他可以帮助获得来分离和分类样本。

## 数据可视化
 &#160; &#160; &#160; &#160;通过数据可视化来进一步了解数据特征的分布情况和不同特征属性之间的关系。
* 使用单变量图表可以更好地理解每一个特征属性。
* 多变量图表用于理解不同特征属性之间的关系。

&#160; &#160; &#160; &#160; 单变量图表显示每一个特征属性，因为每个特征属性都是数字，故采用箱线图来展示属性与中位值的离散速度。代码如下：

  ```#箱线图
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```
执行结果如下图:

![Markdown](https://github.com/dreamlovesft/Markdown-Photos/raw/master/1.png)


&#160; &#160; &#160; &#160; 还可以通过直方图来显示每个特征属性的分布情况。代码如下：
```
dataset.hist()
pyplot.show()
```

在输出的图表中，我们可以看到separ-length和separ-width符合高斯分布。执行结果如下图：
![Markdown](http://i2.bvimg.com/663692/c4a08c0bbef2e750.png)
### 多变量图表
通过多变量图表可知不同特征属性之间的关系。我们通过散点矩阵图来查看每个属性之间的影响关系。

    #多变量图表&散点矩阵图
    scatter_matrix(dataset)
    pyplot.show()
    
执行结果如下：
![Markdown](http://i2.bvimg.com/663692/60d490dea4fafce0.png)

## 评估算法
创建不同的算法模型，评估他们的准确度，进而找到最合适的算法。步骤如下：
（1）分离出评估数集。
（2）采用十折交叉法评估算法模型。
（3）生成六个不同模型来检测新数据。
（4）选择最优模型。
### 分离出评估数据集
&#160; &#160; &#160; &#160; 模型创建以后需要知道创建的模型是否足够好，在选择算法的过程中会采用统计方法来评估算法模型，但是我们更想知道，算法模型对真实数据的准确度如何？这就是保留一部分数据来评估算法模型的主要原因下面将按照80%的训练数据集20%的评估数据集来分离数据。代码如下：

```
#分离数据集
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
```
&#160; &#160; &#160; &#160; 现在分离出了X_train和Y_train用来训练算法创建模型，X_validation和Y_validation用来验证评估模型。

### 评估模型
&#160; &#160; &#160; &#160; 这里采用十折交叉验证来奋力训练数据集，并评估算法模型的准确性。十折交叉验证是随机将数据分成十份：九份用来训练模型，一份用来评估算法。后面我们将使用相同的数据对每一种算法进行训练和评估，并从中选择最好的模型。
### 创建模型
对任何问题来说，不能仅通过对数据进行审查旧版的哪个算法最有效。通过前面的图表发现有些数据符合线性分布，故评估下面六种不同算法：
* 线性回归（LR）
* 线性判别分析（LDA）
* K近邻（KNN）
* 分类与回归树（CART）
* 贝叶斯分类器（NB）
* 支持向量机（SVM）

&#160; &#160; &#160; &#160; 在每次对算法的评估前都会重置随机数的种子，以确保每次对算法评估都使用相同的数据集，接下来就创建并评估六种算法模型。代码如下:
```
#算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
#评估算法
results = []
for key in models:
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_reults = model_selection.cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy' )
  results.append(cv_reults)
  print('%s: %f (%f)' %(key, cv_reults.mean(), cv_reults.std()))
``` 
执行结果如下：

    LR: 0.966667 (0.040825)
    LDA: 0.975000 (0.038188)
    KNN: 0.983333 (0.033333)
    CART: 0.983333 (0.033333)
    NB: 0.975000 (0.053359)

&#160; &#160; &#160; &#160; 通过结果可知SVM算法的准确度最高，接下来箱线图，通过箱线图来比较算法的评估结果。代码如下：

    #箱线图的比较算法
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparision')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(models.keys())
    pyplot.show()
    
执行结果如下：

![Markdown](https://github.com/dreamlovesft/Markdown-Photos/raw/master/4.png)

## 实施预测
&#160; &#160; &#160; &#160; 评估结果显示，支持向量机的是最准确的的算法。。现在使用数据集来验证这个算法模型。这将会对生成的算法模型的准确度有一个更加直观的认识。现使用全部训练集的数据来生成SVM的算法模型，并用预留的评估数据给出一个算法模型的报告。代码如下：


    #使用评估数据集评估算法
    svm = SVC()
    svm.fit(X=X_train, y=Y_train)
    predictions = svm.predict(X_validation)
    print (accuracy_score(Y_validation, predictions))
    print (confusion_matrix(Y_validation, predictions))
    print (classification_report(Y_validation, predictions))


&#160; &#160; &#160; &#160; 执行程序后，看到算法模型的准确度是0.93.通过冲突矩阵看到只有两个数据预测错误。最后还提供一个包含精确率（precision）、召回率（recell）、F1值（F1-score）等数据的报告。结果如下：

    0.9333333333333333
    [[ 7  0  0]
     [ 0 10  2]
     [ 0  0 11]]
                     precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00         7
    Iris-versicolor       1.00      0.83      0.91        12
    Iris-virginica       0.85      1.00      0.92        11

    micro avg       0.93      0.93      0.93        30
    macro avg       0.95      0.94      0.94        30
    weighted avg       0.94      0.93      0.93        30


## 总结
&#160; &#160; &#160; &#160; 这是我机器学习的第一的项目，主要参考了魏贞原老师的《机器学习Python实践》，其中的大部分代码也是来源于书中，其中在hexo部署的时候主要是多个代码段连续插入出现问题。且许多代码还待深深咀嚼，更重要的是学会机器学习的各个步骤，很有收获，嘻嘻。 