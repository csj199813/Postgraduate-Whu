import operator
import numpy as np
from my_dis import distance

def init_path_list(city_path):
	path_list = []
	for k in range(500):
		i = np.random.randint(0, len(city_path))
		j = np.random.randint(0, len(city_path))
		while(i == j):
			i = np.random.randint(0, len(city_path))
			j = np.random.randint(0, len(city_path))
		city_path[i], city_path[j] = city_path[j], city_path[i]
		path_list.append(city_path)
	# for i in range(len(city_path) - 1):
		# for j in range(i + 1, len(city_path)):
		# 	new_path = city_path[:]
		# 	new_path[i], new_path[j] = new_path[j], new_path[i]
		# 	path_list.append(new_path)
	return path_list

def fitness(path_list, dis_map):
	fit_list = []
	for i in path_list:
		dis = distance(dis_map, i, len(i))
		fit_list.append(1/dis)
	return fit_list

def select(path_list, fit_list, num):
	new_city_list = []
	dic = [z for z in zip(path_list, fit_list)]
	new_dic = sorted(dic, key=operator.itemgetter(1))
	new_path, new_fit = zip(*new_dic)
	new_path = list(new_path)
	new_fit = np.array(new_fit)
	sum_fit = np.sum(new_fit)
	new_fit = new_fit/sum_fit
	new_fit = np.cumsum(new_fit)
	rand = np.random.rand(num)
	rand = np.sort(rand)
	fit_num = 0
	new_num = 0
	while new_num < num:
		if rand[new_num] < new_fit[fit_num]:
			new_city_list.append(new_path[fit_num])
			new_num += 1
		else:
			fit_num += 1
	return new_city_list

def crossover(path_list, things_list, pc):
	for i in range(0, len(path_list) - 1, 2):
		list_1 = []
		list_2 = []
		path_1 = []
		path_2 = []
		ex_things_1 = []
		ex_things_2 = []
		new_path_1 = []
		new_path_2 = []
		rand = np.random.rand()
		if rand < pc:
			m = np.random.randint(0, len(path_list[0]) - 2)
			n = np.random.randint(m + 1, len(path_list[0]))
			for t in range(m, n):
				ex_things_1.append(things_list[path_list[i + 1][t]])
				ex_things_2.append(things_list[path_list[i][t]])
			for j in range(n, len(path_list[0])):
				list_1.append(things_list[path_list[i][j]])
				list_2.append(things_list[path_list[i + 1][j]])
				path_1.append(path_list[i][j])
				path_2.append(path_list[i + 1][j])
			for k in range(0, n):
				list_1.append(things_list[path_list[i][k]])
				list_2.append(things_list[path_list[i + 1][k]])
				path_1.append(path_list[i][k])
				path_2.append(path_list[i + 1][k])
			# path_list[i][m:n], path_list[i + 1][m:n] = path_list[i + 1][m:n], path_list[i][m:n]
			for z in range(len(path_list[0])):
				if len(new_path_1) != m:
					if list_1[z] not in ex_things_1:
						new_path_1.append(path_1[z])
					else:
						continue
				else:
					new_path_1.extend(path_list[i + 1][m:n])
					if list_1[z] not in ex_things_1:
						new_path_1.append(path_1[z])
					else:
						continue

			for z in range(len(path_list[0])):
				if len(new_path_2) != m:
					if list_2[z] not in ex_things_2:
						new_path_2.append(path_2[z])
					else:
						continue
				else:
					new_path_2.extend(path_list[i][m:n])
					if list_2[z] not in ex_things_2:
						new_path_2.append(path_2[z])
					else:
						continue


			path_list[i][:] = new_path_2[:]
			path_list[i + 1][:] = new_path_1[:]

	return path_list

def mutation(path_list, pm):
	for i in range(len(path_list)):
		rand = np.random.rand()
		if rand < pm:
			m = np.random.randint(0, len(path_list[0]) - 2)
			n = np.random.randint(m + 1, len(path_list[0]))
			path_list[i][:] = path_list[i][:m] + path_list[i][m:n][::-1] + path_list[i][n:]

	return path_list

def best(city_path, fit_list, dis_map):
	best_path = city_path[0]
	best_fit = fit_list[0]
	for i in range(1, len(fit_list)):
		if fit_list[i] > best_fit:
			best_fit = fit_list[i]
			best_path = city_path[i]
	best_dis = distance(dis_map, best_path, len(best_path))
	return best_path, best_dis



def ga(city_path, things_list, dis_map, pop_num):
	pc = 0.9
	pm = 0.05
	pop = init_path_list(city_path)
	for i in range(1000):
		fit_list = fitness(pop, dis_map)
		new_pop = select(pop, fit_list, pop_num)
		new_pop = crossover(new_pop, things_list, pc)
		new_pop = mutation(new_pop, pm)
		pop = new_pop
		fit_list = fitness(pop, dis_map)
		best_path, best_dis = best(pop, fit_list, dis_map)

	return best_path, best_dis


# a = [1, 2, 3, 4, 5, 6]
# b = a[0:2] + a[2:4][::-1] + a[4:5]
# print(b)
# print(a[2:4][::-1])




			# for z_0 in range(m):
			# 	if list_1[z_0] in ex_things_1:
			# 		continue
			# 	else:
			# 		new_path_1.append(path_1[z_0])
			# for z_0 in range(m):
			# 	if list_2[z_0] in ex_things_2:
			# 		continue
			# 	else:
			# 		new_path_2.append(path_2[z_0])
			# for z_1 in range(m, n):
			# 	new_path_1.append(path_list[i + 1][z_1])
			# 	new_path_2.append(path_list[i][z_1])
			# for z_2 in range(n, len(path_list[0])):
			# 	if list_1[z_2] in ex_things_1:
			# 		continue
			# 	else:
			# 		new_path_1.append(path_1[z_2])
			#








# p = [['0', '1', '2'], ['9', '2', '8']]
# f = [1.4, 0.7]
# dic = [z for z in zip(p, f)]
# new_dic = sorted(dic, key=operator.itemgetter(1))
# print(new_dic)
# c, d = zip(*new_dic)
# d = np.array(d)
# print(c, d.shape[0])



