# ----------------------------------------------------------------------
# 1. 数据预处理
# ----------------------------------------------------------------------
import pandas as pd
# 传入csv文件的路径，返回预处理完毕的列表。
# 由于test和train的处理方式不一样，增加了type形参
# 返回值参数1为完整数据，返回值参数2为输入模型的数据
def transform(csv_path, type="TRAIN"):
    # 数据加载
    full_data = pd.read_csv(csv_path)
    data = full_data.copy()

    # 将Cabin列细分为Deck、Number、Side
    # Deck 指的是邮轮的楼层或甲板
    # Number 是指船舱的编号。邮轮上每个舱房都有一个独特的编号
    # Side 用来表示舱房所在的邮轮侧面，一般分为Port Side（左舷）和Starboard Side（右舷）
    data[['Deck', 'Number', 'Side']] = data['Cabin'].str.split('/', expand=True)
    data = data.drop('Cabin', axis=1)

    # PassengerId和Name关系不大，可以删除
    data = data.drop(['PassengerId', 'Name'], axis=1)

    # 处理布尔型特征，NaN变成0
    data['CryoSleep'] = data['CryoSleep'].fillna(0).astype(int)
    data['VIP'] = data['VIP'].fillna(0).astype(int)
    if type == 'TRAIN':  # 'TEST'模式读入的csv文件不包含Transported列
        data['Transported'] = data['Transported'].fillna(0).astype(int)

    # 填补缺失值NaN
    # 数值型
    for col in ['Age', 'Number', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        # 将列字符串转换为int，无法转换的设置为NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if col in ['Age', 'Number']:
            data[col] = data[col].fillna(data[col].median())  # 年龄和船舱的编号有点特殊，使用中位数填充
        else:
            data[col] = data[col].fillna(0)
    # 类别型
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        data[col] = data[col].fillna(data[col].mode()[0])  # 用出现频率最高的值（即众数）填充

    return full_data, data


_, data = transform('/kaggle/input/spaceship-titanic/train.csv', type="TRAIN")
# print(data)
full_test_data, test_data = transform('/kaggle/input/spaceship-titanic/test.csv', type="TEST")
# print(test_data)


# ----------------------------------------------------------------------
# 2. 划分数据集
# ----------------------------------------------------------------------
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
# 取出Transported列得到标签向量y
lables = data['Transported']
# 将预处理后的数据data划分为：80%测试集：特征features_train + 是否被送走标签lables_train 与 20%验证集：features_test + lables_test
features_train, features_test, lables_train, lables_test = train_test_split(data, lables, test_size=0.1, random_state=42)
# 转换为TensorFlow数据集格式
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(features_train, label='Transported')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(features_test, label='Transported')


# ----------------------------------------------------------------------
# 3. 建立模型
# ----------------------------------------------------------------------
# 构建随机森林模型
model = tfdf.keras.RandomForestModel(
    task=tfdf.keras.Task.CLASSIFICATION,    # 指定任务类型为分类任务
    num_trees=500,  # 增加树的数量
    max_depth=12    # 控制树的深度，防止过拟合
)
model.compile(metrics=['accuracy'])

# ----------------------------------------------------------------------
# 4. 启动训练
# ----------------------------------------------------------------------
# 训练模型
model.fit(train_ds)
# 评估模型
evaluation = model.evaluate(test_ds, return_dict=True)
print(f"Test Accuracy: {evaluation['accuracy']:.4f}")


# ----------------------------------------------------------------------
# 5. 作单人预测
# ----------------------------------------------------------------------
sample_data = pd.DataFrame([{
    'HomePlanet': 'Europa',
    'CryoSleep': 1,
    'Destination': 'TRAPPIST-1e',
    'Age': 30,
    'VIP': 0,
    'RoomService': 0,
    'FoodCourt': 0,
    'ShoppingMall': 0,
    'Spa': 0,
    'VRDeck': 0,
    'Deck': 'B',
    'Number': 0,
    'Side': 'P'
}])
# B模型只能实时训练实时用，不能保存
prediction = model.predict(tfdf.keras.pd_dataframe_to_tf_dataset(sample_data))
print(f"\nPrediction Probability: {prediction[0][0]:.4f}")
print(f"Predicted Class: {'Transported' if prediction[0][0] > 0.5 else 'Not Transported'}")


# ----------------------------------------------------------------------
# 6. 作csv文件批量预测
# ----------------------------------------------------------------------
prediction = model.predict(tfdf.keras.pd_dataframe_to_tf_dataset(test_data))>0.5
submission = pd.DataFrame({
    'PassengerId': full_test_data['PassengerId'],
    'Transported': prediction.flatten()
})
submission.to_csv('submission.csv', index=False)