import numpy as np
import os
import operator
from sklearn.neighbors import KNeighborsClassifier as KNN #用于方法二：调用sklearn库 （效率较低，但精度高一点）

#主程序
def main(k):
	global matrix, label_list
	right = 0 #保存正确预测个数
	error = 0
	# 方法二：调用sklearn库的实现预测
	# knn = KNN(n_neighbors=k, algorithm='auto')
	# knn.fit(matrix, label_list)

	for num, file in enumerate(os.listdir(test_path)):
		print('正在预测第%s个样本'%num, file)
		label = file[0]
		file = os.path.join(test_path, file)
		test = []
		with open(file, 'r') as f:
			f1 = f.readlines()
			for i in f1:
				for j in range(len(i) - 1):
					test.append(int(i[j]))
		test = np.array(test).reshape(1, -1) #单个测试样本向量
		# 计算测试样本与训练集向量的欧式距离
		distance = np.tile(test, (train_len, 1)) - matrix
		#给距离排序，从小到大
		distance = np.sqrt((distance ** 2).sum(axis=1)).argsort()
		#定义字典，用于保存取出的最近的k个label，并累计
		my_dict = {}
		for i in range(k):
			key = label_list[distance[i]]
			my_dict[key] = my_dict.get(key, 0) + 1
		#对保存的k个label排序，取出可能性最大的label（降序）
		predict = sorted(my_dict.items(), key=operator.itemgetter(1), reverse=True)
		print(predict[0][0], label) #打印预测label和真实label 用于观察

		# 方法二：调用sklearn库的实现预测
		# predict = knn.predict(test)

		if predict[0][0] == label:
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
main(7)









