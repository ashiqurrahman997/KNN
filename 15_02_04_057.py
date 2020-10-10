import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def distance(a,b,c,d):
    return (np.power(a-b,2)+np.power(c-d,2))

print("Ashiqur Rahman, ID : 150204057")

df_train = pd.read_csv('train.txt', sep=",", header = None,dtype='Int64')
df_train = pd.DataFrame(df_train.values, columns = ['X', 'Y', 'Class'])


df_test= pd.read_csv('test.txt', sep="," ,  header = None,dtype='Int64')
df_test = pd.DataFrame(df_test.values, columns = ['X', 'Y'])
df_test_cls=df_test.copy();
df_test_cls['Class']=0;

for j in range(0, df_test.shape[0]):
    df_train[j]=0

f = open("prediction.txt", "w")

for i in range(0,df_test.shape[0]):
    for j in range(0, df_train.shape[0]):
        a=df_test.iloc[i,0];
        b=df_train.iloc[j,0];
        c=df_test.iloc[i,1];
        d=df_train.iloc[j,1];

        df_train.iloc[j,i+3]=distance(a,b,c,d)

while(True):
    k = int(input('Input The value of K :'))
    if (k > df_test.shape[0]-1):
        print("K must be less than " + str(df_test.shape[0]-1))
    else : break;


#print(df_train)
for i in range(0, df_test.shape[0]):

    f.write('Test Point : '+str(df_test.iloc[i,0])+','+str(df_test.iloc[i,1])+'\n')
    df_sorted=df_train.sort_values(by=[i])
    df_knn=df_sorted.iloc[0:k,[0,1,2,i+3]]

    #print(df_knn)

    for j in range(0,df_knn.shape[0]):
        f.write('Distance ' +str(j)+' : '+str(df_knn.iloc[j,3])+'     Class: '+str(df_knn.iloc[j,2])+'\n')


    cls=df_knn['Class'].value_counts(sort=True).index[0]
    df_test_cls.loc[i,'Class']=cls;
    f.write('Predicted class: '+str(cls)+'\n'+'\n')

f.close();
#print(df_test_cls)


df1=df_train[df_train['Class'] == 1];
df1=df1.iloc[:,0:2]
w1=df1.values

df2=df_train[df_train['Class'] == 2];
df2=df2.iloc[:,0:2]
w2=df2.values

df1=df_test_cls[df_test_cls['Class'] == 1];
df1=df1.iloc[:,0:2]
w3=df1.values

df2=df_test_cls[df_test_cls['Class'] == 2];
df2=df2.iloc[:,0:2]
w4=df2.values


plt.figure(0);
plt.scatter(w1[:,0],w1[:,1],color = 'red', marker = 'o',label="w1_train")
plt.scatter(w2[:,0],w2[:,1],color = 'blue', marker = '+',label="w2_train")

plt.scatter(w3[:,0],w3[:,1],color = 'green', marker = '>',label="w1_Test")
plt.scatter(w4[:,0],w4[:,1],color = 'Black', marker = '*',label="w2_test")
plt.title('KNN')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend(loc="best",fontsize="small")


plt.show()
