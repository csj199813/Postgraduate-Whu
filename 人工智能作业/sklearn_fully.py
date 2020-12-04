from sklearn.neural_network import MLPClassifier
import numpy as np
import os



def test():
	global matrix, label_list
	right = 0  # 保存正确预测个数
	error = 0
	for num, file in enumerate(os.listdir(test_path)):
		print('正在预测第%s个样本' % num, file)
		label = file[0]
		file = os.path.join(test_path, file)
		test = []
		with open(file, 'r') as f:
			f1 = f.readlines()
			for i in f1:
				for j in range(len(i) - 1):
					test.append(int(i[j]))
		test = np.array(test).reshape(1, -1)  # 单个测试样本向量
		predict = knn.predict(test)
		print(predict[0], label)
		if predict[0] == label:
			right += 1
		else:
			error += 1
	print('已经测试完毕 精确度为：{:.4f}'.format(right/test_len))
	print('已经测试完毕 错误个数：{:.4f}'.format(error))

#设置初始参数
train_path = 'digits/trainingDigits'
test_path = 'digits/testDigits'
label_list = [] #保存训练label
matrix = [] #保存训练样本向量
train_len = len(os.listdir('digits/trainingDigits')) #训练集个数
test_len = len(os.listdir('digits/testDigits')) #测试集个数
#读取训练样本文件，并转换成(样本数，1024列)
for file in os.listdir(train_path):
	# print(file[0], file)
	label_list.append(file[0])
	file = os.path.join(train_path, file)
	with open(file, 'r') as f:
		f1 = f.readlines()
		for i in f1:
			for j in range(len(i) - 1):
				matrix.append(int(i[j]))
matrix = np.array(matrix).reshape(-1, 1024)
#构建MLP模型
knn = MLPClassifier(hidden_layer_sizes=(2000, ), solver='sgd', shuffle=True, learning_rate_init=0.0001)
print('开始训练')
knn.fit(matrix, label_list)
print('训练结束')
#预测，计算测试集精确度
test()