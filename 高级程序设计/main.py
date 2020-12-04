import random
import matplotlib.pyplot as plt
import math
from SA import sa
from TS import ts
from GA import ga
from my_dis import dis_map,distance

# city_9 = []
# city_17 = []
# city_24 = []
# city_31 = []
# city_39 = []
# for i in range(9):
# 	city_9.append('a' + str(i))
# for i in range(17):
# 	city_17.append('b' + str(i))
# for i in range(24):
# 	city_24.append('c' + str(i))
# for i in range(31):
# 	city_31.append('d' + str(i))
# for i in range(39):
# 	city_39.append('e' + str(i))

x_list = random.sample(list(range(50)), 39)
y_list = random.sample(list(range(50)), 39)
point_list = [i for i in zip(x_list, y_list)]
city_1 = point_list[:9]
city_2 = point_list[:17]
city_3 = point_list[:24]
city_4 = point_list[:31]
city_5 = point_list[:]
things_1 = []
things_2 = []
things_3 = []
things_4 = []
things_5 = []
for i in range(9):
	if i < 5:
		things_1.append(i)
	else:
		n = random.randint(0, 4)
		things_1.append(n)

for i in range(17):
	if i < 11:
		things_2.append(i)
	else:
		n = random.randint(0, 16)
		things_2.append(n)

for i in range(24):
	if i < 15:
		things_3.append(i)
	else:
		n = random.randint(0, 14)
		things_3.append(n)

for i in range(31):
	if i < 16:
		things_4.append(i)
	else:
		n = random.randint(0, 15)
		things_4.append(n)

for i in range(39):
	if i < 25:
		things_5.append(i)
	else:
		n = random.randint(0, 24)
		things_5.append(n)

example_1 = dict(zip(city_1, things_1))
example_2 = dict(zip(city_2, things_2))
example_3 = dict(zip(city_3, things_3))
example_4 = dict(zip(city_4, things_4))
example_5 = dict(zip(city_5, things_5))

def run(num, city_num, things_num, things_list, city_list, table_length):
	saved_city = []
	dis_map_ = dis_map(city_list, city_num)

	sa_lest_path = list(range(things_num))
	sa_lest_dis = distance(dis_map_, sa_lest_path, things_num)

	ts_lest_path = list(range(things_num))
	ts_lest_dis = distance(dis_map_, ts_lest_path, things_num)

	ga_lest_path = list(range(things_num))
	ga_lest_dis = distance(dis_map_, ga_lest_path, things_num)

	for i in range(int(num)):
		city = random.sample(list(range(city_num)), things_num)
		things = set()
		for j in city:
			things.add(things_list[j])
		if len(things) < things_num:
			continue
		elif set(city) in saved_city:
			continue
		else:
			saved_city.append(set(city))
			sa_path, sa_dis = sa(city, dis_map_, things_num)
			ts_path, ts_dis = ts(city, dis_map_, table_length)
			ga_path, ga_dis = ga(city,things_list, dis_map_, len(things_list) * (len(things_list) - 1))
		if sa_dis < sa_lest_dis:
			sa_lest_dis = sa_dis
			sa_lest_path = sa_path

		if ts_dis < ts_lest_dis:
			ts_lest_dis = ts_dis
			ts_lest_path = ts_path

		if ga_dis < ga_lest_dis:
			ga_lest_dis = ga_dis
			ga_lest_path = ga_path

	print('sa最短路径为%s'%sa_lest_path)
	print('sa最短距离为%s'%sa_lest_dis)

	print('ts最短路径为%s' % ts_lest_path)
	print('ts最短距离为%s' % ts_lest_dis)

	print('ga最短路径为%s' % ga_lest_path)
	print('ga最短距离为%s' % ga_lest_dis)


run(1e4, 9, 5, things_1, city_1, 2)
print('city_1:', city_1)
print('things_1', things_1)
#打印数据 并且观察点坐标
# print('x', x_list, 'y', y_list, 'p', point_list)
# plt.figure()
# plt.scatter(x_list, y_list)
# plt.xlabel('x轴', fontproperties='SimHei')
# plt.ylabel('y轴', fontproperties='SimHei')
# plt.show()


