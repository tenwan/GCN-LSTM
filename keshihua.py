import matplotlib.pyplot as plt                 #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA           #加载PCA算法包

x, y = [], []
with open("labels.txt", "r") as f:  # 打开文件
    data1 = f.read().split("\n")  # 读取文件
    for i in data1:
        if i == '':
            y.append(-1)
        else:
            y.append(int(i))
with open("embeddings.txt", "r") as f:  # 打开文件
    data1 = f.read().split("\n")  # 读取文件
    index = 1
    for item in data1:
        a = []
        item1 = item.split(" ")
        for i in item1:
            a.append(float(i))
        #print(index)
        index += 1
        x.append(a)

pca=PCA(n_components=2)     #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x)#对样本进行降维
print(reduced_x)

# #可视化
color = ['#F0F8FF', 'green', 'b', 'r', '#7FFFD4', '#FFC0CB', '#00022e']
for index, item in enumerate(reduced_x):
    plt.scatter(item[0], item[1], c= color[y[index]])
plt.show()