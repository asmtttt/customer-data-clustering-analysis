import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering


# veri seti okuma islemi
dataset = pd.read_csv('customer_data.csv')
print(dataset.to_string())
print(dataset.columns)

#

# veri setimizi tanıyalım
print(dataset.shape)


# kolon isimleri değiştiriliyor

kolonlar =["CustomerId","Age","Edu","Years_Employed",
              "Income","Card_Debt","Other_Debt","Defaulted","Address","DebtIncomeRatio"]

for i,k in zip(dataset.columns, kolonlar):

    dataset.rename(columns={i:k}, inplace=True)

print(dataset.columns)


# gereksiz olan veri setlerini siliyoruz.
dataset_new = dataset.drop(columns = ['CustomerId', 'Address'], axis=1)

print(dataset_new.head().to_string())

# veri setine dair bilgi alalım

print(dataset_new.describe().to_string())

# bu kısımda standart sapmalar incelediğimizde a normal bir durum farketmiyoruz
# minimum değerler içinde bir inceleme yapıldığında a normal bir durum farketmiyoruz
# count değerleri incelendiğinde veri setinde bazı kolonların eksik olabileceğinden şüphe ediyoruz.
# bu neden buradan bir eksik veri analizi yapmamız gerektiğini farkediyoruz

# eksik veri analizi

print(dataset_new.isnull().sum())

# buradan Defaulted kolonunda toplam 150 adet satırın eksik olduğunu görüyoruz.
# diğer kolonlarda eksik bir satır bulunmamaktadır.

# defaulted kolonunu inceliyoruz.

print(dataset_new["Defaulted"])

# burada defaulted kolonundaki veriler 0 ya da 1 degerlerinde olusmaktadir.
# yani kategorik verilerdir.
# normal sartlarda eksik veri analizinde cozum stratejileri
# doldurmak ya da silmek uzerinedir
# doldurma stratejisini ele aldigimizda burada yine normal sartlarda
# kolonlarin ortalamasini almak ya da mod degeri yuksek olan deger ile doldurmak vb.
# bircok doldurma yontemi vardir.
# ancak veriler 0 ya da 1 lerden olusacagi icin burada biz 2 farkli yontemi dusunuyoruz

# ilk dusuncemiz eksik satirlari silmek

# ikinci yontemimiz ise 0 ya da 1 lerin veri kolonundaki dagilimini inceleyerek hangi degerin
# daha cok agir bastigini gormek ve bos kolonlari bu deger ile doldurmak
# ikinci yontemi degerlendirmeye aliyoruz

defaulted_count = dataset_new["Defaulted"].value_counts()
print(defaulted_count)

# inceleme sonucunda
# 0.0 :    517 deger var
# 1.0 :    183 deger var

# buradan eksik kolonlari 0 ile dolduracagimiza karar veriyoruz.

dataset_new["Defaulted"].fillna(value= 0.0, inplace=True)
print(dataset_new.isnull().sum())

##################################

# son durumda veri setimiz

print(dataset_new.describe().to_string())

# korelasyon ısı haritası gösteriliyor
corrmat= dataset_new.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corrmat,annot=True)
plt.show()

########################################################################

# ALGORITMALAR

# K DEGERI BULMAK

from sklearn.cluster import KMeans

# elbow yontemi ile k degerini bulmaya calisiyoruz.

sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state= 100)
    kmeans.fit(dataset_new)
    sonuclar.append(kmeans.inertia_)  #inertia wcss değerleri

plt.plot(range(1,10),sonuclar)
plt.show()

# elbow yani dirsek yontemi ile k degerinin 4 olabilecegini dusunuyoruz
# grafikte son kirilmanin 4 degerinde yasandigini tespit ediyoruz.


x1 = dataset_new.iloc[:, 3]
x2 = dataset_new.iloc[:, 4]


print(x1.head())
print("\n\n")
print(x2.head())

# gorsellestirmede kullanacagimiz kolonlari birlestiriyoruz
data = pd.concat([x1, x2], axis=1)

X = data.iloc[:, 0:2].values


# K-MEANS ALGORITMASI

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 200, n_init = 10, random_state = 42)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'purple')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'Küme Merkezleri')
plt.title('K-MEANS GRAFİK')
plt.xlabel('Gelir')
plt.ylabel('Borc')
plt.legend()
plt.show()


# HIYERARSIK KUMELEME ALGORITMASI

from sklearn.cluster import AgglomerativeClustering


# ELBOW YONTEMI ILE BULDUGUMUZ K = 3 DEGERINI BURADA DA KULLANIYORUZ

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

y_predict = ac.fit_predict(X)

plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100, c='red')
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100, c='blue')
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100, c='green')
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s=100, c='purple')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'Küme Merkezleri')
plt.title('HIYERARSIK KUMELEME GRAFIK')
plt.xlabel('Gelir')
plt.ylabel('Borc')
plt.legend()

plt.show()


# SOM KUMELEME ALGORITMASI

from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM

som = SOM(m=4, n=1, dim=2, random_state=100)
som.fit(X)
predictions = som.predict(X)


x = X[:,0]
y = X[:,1]

colors = ['red','blue','green','purple']
plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'Küme Merkezleri')
plt.title('SOM GRAFİK')
plt.xlabel('Gelir')
plt.ylabel('Borc')
plt.legend()

plt.show()










