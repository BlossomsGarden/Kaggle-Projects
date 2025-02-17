import torch
# print(torch.cuda.is_available())

# CIFAR10数据集：60000样本，其中50000用于训练，10000用于测试。每个样本为32*32三通道。
# 共10类物体，标签0~9分别是：
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Batch、Batch Size 和 Epoch 的区别
#
# （1）Batch
# 定义：Batch 是一次训练中使用的样本集合。
# 作用：用于计算梯度和更新参数。
# 示例：如果数据集有 1000 个样本，Batch Size 为 100，则每次训练使用 100 个样本作为一个 Batch。
#
# （2）Batch Size
# 定义：Batch Size 是每个 Batch 中样本的数量。
# 作用：
# 影响梯度计算：Batch Size 越大，梯度计算越稳定，即训练损失减少慢，但内存需求也越大、收敛到最小验证损失所需的 epoch 越多、。
# 影响训练速度：Batch Size 越大，训练速度越快（因为并行计算更多）、每个epoch训练所需的时间越少，但即使单epoch时间少，它也无法在总训练时间上与低batch-size匹敌。
# （参考：https://zhuanlan.zhihu.com/p/420167970）
# 示例：Batch Size = 32 表示每次训练使用 32 个样本。
#
# （3）Epoch
# 定义：Epoch 是整个数据集被完整遍历一次的训练过程。
# 作用：用于衡量训练进度。
# 示例：如果数据集有 1000 个样本，Batch Size 为 100，则 1 个 Epoch 包含 10 个 Batch。
#
# （4）三者的关系
# Batch：一次训练中使用的样本集合。
# Batch Size：每个 Batch 中样本的数量。
# Epoch：整个数据集被完整遍历一次的训练过程。
#
# （5）示例
# 假设数据集有 1000 个样本，Batch Size 为 100：
# Batch：每次训练使用 100 个样本。
# Epoch：1 个 Epoch 包含 10 个 Batch
#
# 值得注意的是：epoch指的是训练过程中某张图片都被用了几次，batch_size和batch则在训练过程中的梯度下降部分使用，
# 而迭代次数又是另一个新概念，是指对于某batch而言直到损失函数下降到几乎不在变化时已经进行w=w-a*partial w计算的次数，
#  batch-size选的好可以减少迭代次数，进而使得经过更少epoch损失函数就下降到真正的低谷

# ----------------------------------------------------------------------
# 1. 数据处理流程定义
# ----------------------------------------------------------------------

# transform，即data transform，数据转换流程，也就是俗称的预处理。分为Augmentation（前三项）、normalization（后两项）。
# transform_train 表示这是对专门对训练集设计的数据转换流程（是一种转换流程而非数据！！！）
# transforms.Compose 库将多个数据预处理/增强步骤按顺序串联（类似流水线），其中
# randomHorizontalFlip 随即水平翻转图像，概率默认50%
# randomRotation 随即旋转图像，范围-10°~+10°
# randomCrop 中padding表示在原始图像边缘填充4像素（防止裁剪丢失关键信息）然后在里面随机裁剪为32*32
# toTensor 只是转数据格式为tensor
# normalize 第一组元组是RGB三通道的均值（mean）第二组元组是RGB三通道的标准差（std）。对张量进行标准化，将输入数据分布调整到接近标准正态分布，加速模型收敛
from torchvision import transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
# 相比之下，测试数据集则不用做增强，但还是要归一化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# ----------------------
# 2. Dataset和DataLoader（ImageLoader）
# ----------------------
from torch.utils.data import Dataset, DataLoader
import pandas as pd     # 主要是有很多预设数据结构和接口，方便文件读写CSV/Excel/Json/SQL、数据筛选、统计、可视化等
from PIL import Image
import os
# 采用自己重写Dataset方法的形式实现，括号中传入Dataset指继承pyTorch库中的Dataset，便于与DataLoader无缝集成
# class定义的叫类，里面def定义的代码块叫函数；用类定义的实例叫对象，对象打点可以调用它的方法
class MyCIFAR10Dataset(Dataset):
    # 无需手动调用，直接MyCIFAR10(传参)即可
    # img_dir (string): 图像目录路径
    # csv_dir (string): 包含图像序号和标签的CSV文件路径
    # transform (callable, optional): 数据转换流程
    def __init__(self, dataset_dir, csv_dir, transform=None):
        self.labels = pd.read_csv(csv_dir)
        self.dataset_dir = dataset_dir
        self.transform = transform

    # 在 PyTorch 的 Dataset 类中，__len__() 的作用是返回数据集的样本数量，通常通过 len(ds) 来调用
    def __len__(self):
        return len(self.labels)

    # __getitem__ 是用于实现索引访问的特殊方法。隐式，使用索引访问对象如 obj[index]，Python 自动调用 __getitem__
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, f"{self.labels.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert('RGB')
        label = classes.index(self.labels.iloc[idx, 1])    # 获取self.labels（一个pandas数据结构） 中获取第 idx 行的第 1 列的值，对应csv文件里那些0~9的标签

        if self.transform:
            image = self.transform(image)
        return image, label

# 开始创建数据集
train_dataset = MyCIFAR10Dataset(
    dataset_dir='../train/',
    csv_dir='../trainLabels.csv',
    transform=transform_train
)
test_dataset = MyCIFAR10Dataset(
    dataset_dir='../train/',
    csv_dir='../testLabels.csv',  # 假设测试集使用相同标签文件
    transform=transform_test
)

# 开始创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)
test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=4
)


# ----------------------------------------------------------------------
# 2. CNN模型定义
# ----------------------------------------------------------------------
# PyTorch 中，nn.Module 是构建神经网络模型的基类，后面的def forward()什么其实是在重写方法（PyTorch自动实现反向传播）
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        # super一下是调用父类 nn.Module 的构造函数，确保 CNN 类正确初始化
        # Python 中子类的构造函数不会自动调用父类的构造函数
        # nn.Module 的构造函数会初始化一些内部状态（如参数管理、子模块管理等），ctrl可以进去看，否则无法正确继承
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入: 3x32x32
            # 卷积层：
            # 输入通道3，输出通道32，卷积核3x3（权重值不是确定的，得后续训练来调整它！），填充1
            # padding表示在外面填充1圈0值像素，即黑点，于是图像变为34*34.在3*3的卷积核处理后得到的结果仍32*32
            # 默认bias=True。添加偏置项有如下作用：
            # 提供更好的拟合能力：为每个输出通道提供额外的自由度，可以让每个卷积通道独立地调整其输出值，优化网络性能
            # 避免输出为零：若输入0或者卷积核输出0，没有偏置项时卷积层的输出可能始终0，限制了网络的学习能力
            # 提升模型灵活性：没有偏置项时卷积输出只会受到卷积核权重控制，可能限制模型对数据的学习能力。加上偏置项每个输出的计算结果都可以进行平移调整，
            nn.Conv2d(3, 32, 3, padding=1),
            # 批归一化层
            # 对每一批（batch）数据的每个特征通道进行归一化，使其均值为 0，方差为 1。可以加速训练
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 激活函数
            nn.Conv2d(32, 32, 3, padding=1),  # 卷积层：输入通道32，输出通道32，卷积核3x3，填充1
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2, 2),  # 最大池化层：池化核2x2，步幅2，输出: 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 64x8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: 128x4x4
        )

        # fc_layers 是全连接层（Fully Connected Layers或Dense Layers），用于将卷积层提取的特征映射到最终的分类结果
        #  512 → 256 → 10，而不是一步到位的原因：
        # 1.分阶段映射的参数量大，提升模型表达能力，找更多深层次的高级特征啊！防止20480直接跑到10网络太简单欠拟合了
        # 2.分阶段可以使梯度在反向传播时更稳定，避免梯度消失或梯度爆炸

        # 分块小矩阵好，避免单一大矩阵导致梯度剧烈变化。
        # 反向传播中，梯度是通过链式法则逐层计算的。每一层的梯度是前一层的梯度乘以当前层的权重矩阵。
        # 如果网络很深，梯度会经过多次连乘，导致以下问题：
        #
        # 梯度消失：
        # 反向传播时，梯度随着网络深度指数级减小，导致深层参数无法更新（如果每层的梯度小于 1，多次连乘后梯度会趋近于 0）
        #
        # 梯度爆炸：
        # 梯度随着网络深度指数级增大，导致参数更新步长过大，模型无法收敛（如果每层的梯度大于 1，多次连乘后梯度会趋近于无穷大）
        #
        #
        # 这里深层梯度指网络中靠近输入层的梯度（即反向传播时经过多层传递后的梯度），浅层梯度指网络中靠近输出层的梯度。

        # 3.里面加入了多个全连接层和非线性激活函数（如 ReLU），模型可以学习更复杂的特征表示
        # 4.从 2048 维（128 * 4 * 4）映射到 512 维，可以提取更抽象的特征（低级特征（如边缘、纹理）到更抽象的高级特征）从 512 维映射到 256 维（对高级特征进一步压缩，提取更关键的分类信息）又高维到低维了
        self.fc_layers = nn.Sequential(
            # 1. nn.Linear 与 nn.Conv2d 的区别
            # nn.Linear 和 nn.Conv2d 都是神经网络中的常用层，它们的作用和使用场景有所不同：
            # nn.Linear（全连接层）：
            # 用于 将输入张量通过一个线性变换映射到一个新的空间。它将每个输入样本的每个特征与一组权重进行加权，并加上一个偏置，最终输出一个新的特征向量。
            # 输入和输出是 一维张量。假设输入是一个大小为 (batch_size, in_features) 的张量，输出是一个大小为 (batch_size, out_features) 的张量。
            # 常用于 高层特征学习，如在卷积神经网络（CNN）中，将卷积层的输出展平并送入全连接层进行分类。
            #
            # nn.Conv2d（卷积层）：
            # 用于 局部感受域的特征提取，常用于图像处理等任务。卷积层通过卷积核对输入图像或特征图进行卷积运算，提取局部特征并生成新的特征图。
            # 输入和输出是 二维张量（通常是形状为 (batch_size, channels, height, width) 的四维张量）。
            # 常用于 图像处理、语音处理等任务，能够有效捕捉局部特征并减少参数量。
            #
            #
            # 将view展开后的（128 * 4 * 4 = 2048 维）输入特征映射到 512 维空间
            # 计算公式： y=Wx+b，其中W 是权重矩阵，b 是偏置向量。
            # 参数量计算方式：输入维度*输出维度（即映射矩阵）+输出维度（即偏置项）
            nn.Linear(128 * 4 * 4, 512),
            #  正则化 技术，旨在 减少过拟合，提高网络的泛化能力
            # 训练时随机丢弃部分神经元的输出，使每次更新时网络的结构都不同。
            # 迫使网络训练时不能依赖于某些特定的神经元或特征，每次训练相当于一个不同的子网络，最终效果类似多个模型的平均。
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # x.view() 用于展平张量，将多维张量转换为一维或二维张量。
        # 即4元组tensor (batch_size, channels, height, width) 变成2元组tensor (batch_size, 128 * 4 * 4)，即batch_size和”特征向量“
        # 保留batch_size是为了并行计算和批量梯度更新
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    # 这里省略了反向传播的步骤，pyTorch自动实现。（https://www.bilibili.com/video/BV1Pa411X76s 参见3.4代价函数4.1梯度下降4.5为何平方损失函数必唯一最佳值）
    # 反向传播是如何计算和更新神经网络权重的？（核心是minimize J(w,b)，即求偏导）
    # 1、前向传播：
    # 输入数据通过神经网络进行处理，直到得到最终输出。
    # 计算预测值和真实值之间的 损失（例如，交叉熵损失或均方误差）。
    # 2、反向传播：
    # 计算 损失函数相对于每个权重的梯度，即网络中每个参数（包括卷积核、全连接层的权重等）的梯度。
    # 这个过程通过 链式法则 实现，从输出层开始，逐层计算每个层的梯度，直到输入层。
    # 梯度计算的公式基于链式法则，逐步更新权重。
    # 3、更新权重：
    # 通过优化算法（如 SGD、Adam 等）使用计算得到的梯度来更新权重。
    # 每次更新时，优化器会根据梯度的大小调整权重，通常是通过减去梯度的一个比例（由学习率控制）来更新参数。


# ----------------------------------------------------------------------
# 3. 设置训练和验证方法、超参数
# ----------------------------------------------------------------------
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
# 即损失函数，他叫评价标准了
criterion = nn.CrossEntropyLoss()
# Adam 是一种常用的优化算法，基于传入的lr自动微调每个参数的学习率
# lr为学习率，常用默认值0.001。权重衰减为L2 正则化 的一种形式，数值为正则化强度，加速收敛
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# 配合上面的Adam算法，每经过 15 个训练周期（epoch），就会衰减一次学习率，变为原来的0.1倍
# 有助于在训练过程中动态调整学习率，从而避免过早设置学习率过大或过小的问题，帮助找到快速下降又不爆炸的好率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 执行一次即为1 epoch
def train(epoch):
    # 只为将模型设置为训练模式
    # 训练过程中，一些特定的层（如 Dropout 和 BatchNorm）的行为会有所不同。
    #  Dropout 层在训练过程中随机丢弃一部分神经元，但在评估模式（model.eval()）下，它会保持所有神经元的输出。
    # BatchNorm 层在训练过程中会使用每个批次的统计信息来更新其均值和方差，但在评估模式下会使用整个训练集的统计信息。
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):        # 对之前构造好并用DataLoader加载过的训练集进行遍历
        inputs, targets = inputs.to(device), targets.to(device)           # 转换数据格式
        # PyTorch 中，梯度是通过反向传播计算的，并且默认是累加的（每次计算梯度时会加到现有的梯度上）
        # 每次梯度更新前需要清零梯度，避免梯度不断累积导致不正确的参数更新
        optimizer.zero_grad()
        # 使用模型进行一次前向传播，获得预测结果
        outputs = model(inputs)
        # 通过预测结果与目标，计算损失函数
        loss = criterion(outputs, targets)


        # 踏马我要看看loss是什么
        # print(loss)


        # 反向传播自动更新梯度
        loss.backward()
        # 基于计算所得梯度，根据优化算法调整模型权重
        optimizer.step()

        running_loss += loss.item()         # running_loss为当前epoch总损失，.item()将数据类型由tensor转为python数据
        # outputs.max(1) 返回每个样本的最大值和其对应的索引。1 表示按每个样本的类进行最大值选择，返回的是每个样本的预测类。
        # _：我们并不关心最大值，只关心后面那个最大值对应的类索引（即预测类别）。
        # predicted：要的预测的类别。
        _, predicted = outputs.max(1)
        # targets.size(0) 返回当前批次中的样本数（通常是 batch size），加到 total 上，用于计算准确率。
        total += targets.size(0)
        # predicted和targets均为一维tensor，形状为 ([batch_size])。
        # .eq()为tensor比较，返回一维布尔tensor，形状也为 ([batch_size])
        # .sum()统计一维布尔tensor中True的个数
        # .item()将数据类型由tensor转为python数据
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Acc: {acc:.2f}%")
    return train_loss, acc

# 纯在test数据集上测试效果，不算梯度不更新参数不启动Dropout固定BatchNorm
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # 禁用梯度计算
    # 直接调用 torch.no_grad() 是无效的，必须配合 with 语句使用，离开with作用域马上自动恢复默认的梯度计算模式
    # 因此简单地在每个for循环里面第一句加torch.no_grad()是没用的
    #
    # with语句用于上下文管理。torch.no_grad()是一个上下文管理器，里面实现了__enter__和__exit__方法
    # 进入 with 代码块时，会执行其 __enter__() 方法
    # 离开 with 代码块时（无论是否发生异常），会自动执行其 __exit__() 方法，进行资源释放
    #
    # open()有异曲同工之妙
    # with open('./test_runoob.txt', 'w') as my_file:
    #     my_file.write('hello world!')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / total
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {acc:.2f}%\n")
    return test_loss, acc


# ----------------------------------------------------------------------
# 4. 启动训练
# ----------------------------------------------------------------------

# 报错如下，原因是没有正确使用 if __name__ == '__main__' 保护主程序
# RuntimeError:
#         An attempt has been made to start a new process before the
#         current process has finished its bootstrapping phase.
#
#         This probably means that you are not using fork to start your
#         child processes and you have forgotten to use the proper idiom
#         in the main module:
#
#             if __name__ == '__main__':
#                 freeze_support()

# if __name__ == '__main__':
#     best_acc = 0
#     epochs = 30
#     train_losses = []
#     test_losses = []
#     train_accs = []
#     test_accs = []
#
#     for epoch in range(epochs):
#         train_loss, train_acc = train(epoch)
#         test_loss, test_acc = test()
#         scheduler.step()
#
#         # 记录指标
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
#         train_accs.append(train_acc)
#         test_accs.append(test_acc)
#
#         # 保存最佳模型
#         if test_acc > best_acc:
#             best_acc = test_acc
#             torch.save(model.state_dict(), 'cifar10_cnn.pth')

# ----------------------------------------------------------------------
# 5. 预测示例
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)

    model.load_state_dict(torch.load('cifar10_cnn.pth'))
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # plt.imshow(Image.open(image_path))
    # plt.title(f"Predicted: {classes[predicted[0]]}")
    # plt.axis('off')
    # plt.show()

    return classes[predicted[0]]

from tqdm import tqdm
# 示例使用（假设有一个test_image.png）
if __name__ == '__main__':
    # res=predict('../ready-for-test/2.png')
    # print(res)

    file_path='../ready-for-test/'
    image_files = [f for f in os.listdir(file_path) ]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))     # 按数字名称排序
    # 创建一个空列表来存储所有的数据
    data = []

    model.load_state_dict(torch.load('cifar10_cnn.pth'))
    model.eval()
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        for image_file in tqdm(image_files, desc="Processing Images", ncols=100):
            image_id = os.path.splitext(image_file)[0]  # 使用 os.path.splitext 去除后缀

            # 构造元组 {'id': image_name, 'label': res}
            # data.append({'id': image_id, 'label': predict(os.path.join(file_path, image_file))})

            image = Image.open(os.path.join(file_path, image_file)).convert('RGB')
            image = transform_test(image).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            data.append({
                'id':image_id,
                'label':classes[predicted[0]]
            })

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    # 将 DataFrame 写入 CSV 文件，设置 index=False 防止写入行索引
    df.to_csv('./sampleSubmission.csv', index=False)
    print("CSV file has been created successfully.")
