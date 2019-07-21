import math
import random
import matplotlib.pyplot as plt
import copy

#First function to optimize
def function1(x):
    return x[0]


#Second function to optimize
def function2(x):
    gx = funcationG(x)
    value = 1 - math.sqrt(x[0] / gx)
    return gx * value


def funcationG(x):
    sumX = 0
    for i in range(1, varNum):
        sumX = sumX + x[i]
    return 1 + (9 * sumX) / (varNum - 1)


#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1


#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list


#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0,len(values1))] #解所支配的集合
    front = [[]] #排序结果
    n = [0 for i in range(0,len(values1))]  #支配者数量
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]

    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])

    distance[0] = math.inf
    distance[len(front) - 1] = math.inf

    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values1[sorted1[k-1]]) / (max(values1) - min(values1))

    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values2[sorted2[k+1]] - values2[sorted2[k-1]]) / (max(values2) - min(values2))

    return distance


def crossover(individuala, individualb, a, b):
    individual1 = copy.deepcopy(individuala)
    individual2 = copy.deepcopy(individualb)

    for j in range(min(a, b), max(a, b) + 1):
        u = random.random()
        if u < 0.5:
            r = (2 * u) ** (1 / (NC + 1))
        else:
            r = (1 / (2 * (1 - u))) ** (1 / (NC + 1))
        individual1[j] = 0.5 * ((1 + r) * individual1[j] + (1 - r) * individual2[j])
        individual2[j] = 0.5 * ((1 - r) * individual1[j] + (1 + r) * individual2[j])

        individual1[j] = 1 if individual1[j] > 1 else individual1[j] # 此处需要修改为常量
        individual2[j] = 1 if individual2[j] > 1 else individual2[j]

        individual1[j] = 0 if individual1[j] < 0 else individual1[j]
        individual2[j] = 0 if individual2[j] < 0 else individual2[j]
    return individual1, individual2


def mutation(individual, a):
    individualTemp = copy.deepcopy(individual)
    u = random.random()
    if u < 0.5:
        r = (2 * u) ** (1 / (NM + 1)) - 1
    else:
        r = (1 - (2 * (1 - u))) ** (1 / (NM + 1))

    individualTemp[a] = individualTemp[a] + r

    individualTemp[a] = 1 if individualTemp[a] > 1 else individualTemp[a] #此处需要修改为常量
    individualTemp[a] = 0 if individualTemp[a] < 0 else individualTemp[a]
    return individualTemp

# 使用此函数之前要确保非支配排序解中都按照拥挤度排序过了
def competition(non_dominated_sorted, numOfSelect):
    selectionResult = []
    non_dominated_sortedTemp = []
    for i in range(0, len(non_dominated_sorted)):
        for j in range(0, len(non_dominated_sorted[i])):
            non_dominated_sortedTemp.append(non_dominated_sorted[i][j])

    while len(selectionResult) < numOfSelect:
        selections = [random.randint(0, popSize - 1) for i in range(0, popSize // 2)]
        selectionResult.append(non_dominated_sortedTemp[min(selections)])

    return selectionResult


min_x = 0
max_x = 1
varNum = 30
popSize = 100
max_gen = 500
pc = 0.9
pm = 0.01
NC = 20
NM = 20

#初始化种群
solution = []
for i in range(0, popSize):
    solution.append([random.random() for j in range(0, varNum)])


times = 0
non_dominated_sorted_solution = []
function1_values = []
function2_values = []

while times < max_gen:
    # 计算种群中，每个个体的两个目标函数值
    function1_values = [function1(solution[i]) for i in range(0, popSize)]
    function2_values = [function2(solution[i]) for i in range(0, popSize)]

    # 快速非支配排序
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])

    # 计算拥挤度
    crowding_distance_values = []
    for i in range(0, len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))

    # 按照拥挤度排序
    for i in range(0, len(non_dominated_sorted_solution)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution[i][j],non_dominated_sorted_solution[i]) for j in range(0,len(non_dominated_sorted_solution[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values[i][:])#按照拥挤度排序
        non_dominated_sorted_solution[i] = [non_dominated_sorted_solution[i][front22[j]] for j in range(0, len(non_dominated_sorted_solution[i]))]
        non_dominated_sorted_solution[i].reverse()#翻转之后是从大到小排序，拥挤度

    solution2 = copy.deepcopy(solution)

    #竞标赛选取
    offSpring = competition(non_dominated_sorted_solution, popSize)
    #交叉
    for i in range(0, len(offSpring) // 2):
        rc = random.random()
        if(rc < pc):
            index1 = random.randint(0, varNum - 1)
            index2 = random.randint(0, varNum - 1)
            while index1 == index2:
                index1 = random.randint(0, varNum - 1)
            indi1, indi2 = crossover(solution[offSpring[i]], solution[offSpring[len(offSpring) - i - 1]], index1, index2)
            solution2.append(indi1)
            solution2.append(indi2)
    #变异
    for i in range(0, len(offSpring)):
        rm = random.random()
        if rm < pm:
            indexOfM = random.randint(0, varNum - 1)
            muIndi = mutation(solution[offSpring[i]], indexOfM)
            solution2.append(muIndi)

    #计算函数值
    lengthOfSolution2 = len(solution2)
    function1_values2 = [function1(solution2[i])for i in range(0, lengthOfSolution2)]
    function2_values2 = [function2(solution2[i])for i in range(0, lengthOfSolution2)]

    #非支配快速排序
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])

    #计算拥挤度
    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

    #产生新的种群
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i]) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])#按照拥挤度排序
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()#翻转之后是从大到小排序，拥挤度
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==popSize):
                break
        if (len(new_solution) == popSize):
            break
    solution = [solution2[i] for i in new_solution]
    times = times + 1
    print('This is ', times, '\n')


#Lets plot the final front now
minF1 = []
minF2 = []
for k in non_dominated_sorted_solution[0]:
    minF1.append(function1_values[k])
    minF2.append(function2_values[k])

# 打印第一层
print("The best front for Generation number ", times, " is")
for valuez in non_dominated_sorted_solution[0]:
    print(solution[valuez:], end=" ")
print("\n")

plt.xlabel('F 1', fontsize=15)
plt.ylabel('F 2', fontsize=15)
plt.scatter(minF1, minF2)
plt.show()
