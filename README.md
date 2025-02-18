# 💿 Kaggle-Projects 

Kaggle项目学习记录


##  [CIFAR10-Recognition](https://www.kaggle.com/competitions/cifar-10/overview)

入门级难度，比赛时间已过。基于已给出的50,000张CIFAR10数据集图片的图像分类任务。额外给出300,000张未知标签的图片，将预测结果以特定格式写入csv文件中提交，平台反馈正确率。

目标：熟悉 PyToch 环境配置、经典图像分类神经网络的构建与训练。分别采用 VGG-13 和 ResNext 2种架构实现。

### [* VGG-13](https://arxiv.org/abs/1409.1556)
代码文件 cifar10-cnn.py 模型结构如下，由于CIFAR10数据集中都是32*32的小图，所以应用了层数较少的一种。预测正确率 0.8507。
![1](https://github.com/user-attachments/assets/dd43cc5a-42a2-4e53-ba78-1cb76a277ee7)

### [* ResNext](https://ieeexplore.ieee.org/document/8100117)
代码文件 cifar10-resnext.py

正在施工中...敬请期待！


##  [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview)

入门级难度，永久开放。基于csv文件形式乘客信息（存在缺失项）的监督学习任务。额外以相同的格式给出一些乘客的信息，将预测结果以特定格式写入csv文件中提交，平台反馈正确率。非常友好的是介绍中给出了一个有详细注释的已完成代码帮助上手！

目标：熟悉 tensorflow 环境配置，从csv文件出发的大数据预测，用TF-DF库实现决策树

代码文件 main.py 预测正确率 0.7964（主要时间放在pd库处理NaN数据或者数据类型转换上了，反而模型训练这块直接调库一行解决(?) 给出的已完成代码中提到的几种决策树算法都有现成的库，就算是“微调”也只是变一变形参，感觉非常不好，适合学习完理论后再来上手）PS:似乎调用tfdf的决策树接口训练的模型保存后重新加载会报错，又浏览了项目的许多公开代码似乎也都是直接训完了马上预测，难道是因为模型在tfdf库中而load_model方法在tf库中不兼容(?) 不过确实训的很快几秒钟就结束了也没必要纠结保存就是了
