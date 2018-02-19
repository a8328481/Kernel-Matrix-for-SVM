
# coding: utf-8

# In[1]:

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.datasets import load_iris
import time
import seaborn as sb
from sklearn import svm
get_ipython().magic(u'matplotlib inline')
from __future__ import division


# In[2]:

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[3]:

iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[4]:

df.head()


# In[ ]:




# In[3]:

# 1.extract a new dataset
iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]=scaler.fit_transform(df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']])
df=pd.DataFrame(df,columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','target'])


# In[5]:

df.describe()


# In[5]:

set(df['target'])


# In[6]:

df_extract=df[df['target']!=2.0]


# In[7]:

df_extract.to_csv('Iris_binary_data.cvs')


# In[8]:

df_extract.head()


# In[9]:

# 2. compare SVM classification under four kernels


# In[10]:

training_data=df_extract[['sepal length (cm)','sepal width (cm)']]
training_data_label=df_extract['target']


# In[11]:

training_data.head()


# In[12]:

cmap_light=ListedColormap(['#FFFAAA','#AAFFAA'])
cmap_bold=ListedColormap(['#FF0000','#00FFAA'])


# In[13]:

x_min,x_max = training_data['sepal length (cm)'].min()-1,training_data['sepal length (cm)'].max()+1
y_min,y_max=training_data['sepal width (cm)'].min()-1,training_data['sepal width (cm)'].max()+1
h=0.01


# In[14]:

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))


# In[15]:

Test_data=np.c_[xx.ravel(),yy.ravel()]


# In[16]:

kernel_list=['linear','rbf','poly','sigmoid']

for kernel in kernel_list:
    clf=svm.SVC(kernel=kernel,tol=0.0001,gamma='auto')
    clf.fit(training_data,training_data_label)
    test_data_labels=clf.predict(Test_data)
    test_data_labels= test_data_labels.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,test_data_labels ,cmap=cmap_light)

    plt.scatter(training_data['sepal length (cm)'],training_data['sepal width (cm)'],c=training_data_label, cmap=cmap_bold)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())

    plt.title('SVM classification kernel= '+ str(kernel))


# It seems that kernel with rbf and poly have a nice decision boundary while kernel with linear did not bad. Kernel with sigmoid performed terrible.

# In[17]:

clf=svm.SVC(kernel=kernel,tol=0.0001,gamma=0.1)
clf.fit(training_data,training_data_label)


# In[18]:

clf.score(training_data,training_data_label)


# In[19]:

# 3.Partition Iris binary data into training data that counts 55% and test 45%


# In[20]:

from sklearn.model_selection import train_test_split


# In[21]:

test_percent=0.45
training_data,test_data,training_data_label, test_data_label=train_test_split(df_extract[['sepal length (cm)','sepal width (cm)']], df_extract['target'], test_size=test_percent, random_state=42)


# In[22]:

kernel_list=['linear','rbf','poly','sigmoid']

for kernel in kernel_list:
    clf=svm.SVC(kernel=kernel,tol=0.0001,gamma='auto')
    clf.fit(training_data,training_data_label)
    result=clf.predict(test_data)
    
    t_idx = result==test_data_label    # trult predicted
    f_idx = np.logical_not(t_idx)     # falsely predicted
    p_idx = test_data_label>0             # positive target
    n_idx = np.logical_not(p_idx)      # negative target
    tp=np.sum(np.logical_and(t_idx,p_idx))   # TP
    tn=np.sum(np.logical_and(t_idx,n_idx))   # TN
    fp=np.sum(n_idx)-tn # FP
    fn=np.sum(p_idx)-tp # FN
    
    sensitivity=(1.0*tp)/(tp+fn)
    specificity=(1.0*tn)/(tn+fp)        
    ppr=(1.0*tp)/(tp+fp)      
    npr=(1.0*tn)/(tn+fn)            
    accuracy=(tp+tn)*1.0/(tp+fp+tn+fn)      
    F1= (2*tp)/(2*tp+fp+fn)
    
    print "Kernel is: "+str(kernel)
    print "sensitivity: "+ str(sensitivity)
    print "specificity: "+ str(specificity)
    print "positive predictive ratio: "+ str(ppr)
    print "negative predictive ratio: "+ str(npr)

    print "accuracy: "+str(accuracy)
    print "F1-score: "+str(F1)+'\n'


# In[23]:

# 4. multiclass classification


# In[24]:

data=iris.data[:,0:2]
label=iris.target


# In[25]:

data=scaler.fit_transform(data)


# In[26]:

len(label)


# In[27]:

from sklearn.metrics import classification_report
kernel_list=['linear','rbf','poly','sigmoid']

for kernel in kernel_list:
    clf=svm.SVC(kernel=kernel,tol=0.0001,gamma='auto',decision_function_shape='ovo')
    clf.fit(data,label)
    result=clf.predict(data)
    target_names = ['class 0', 'class 1', 'class 2']
    print "kernel: "+str(kernel)
    print classification_report(label,result,target_names=target_names)
    


#  # Selective Learning analytics

# In[28]:

df=pd.read_csv('NBOption.csv')


# In[29]:

data=df[['LastPrice','time_to_maturity']]
label=df['Volatility']


# In[30]:

data.head(10)


# In[31]:

scaler_sl=StandardScaler()


# In[32]:

data=pd.DataFrame(scaler_sl.fit_transform(data),columns=['LastPrice','time_to_maturity'])


# In[33]:

len(data)


# In[34]:

test_percent=0.2
training_data,test_data,training_data_label, test_data_label=train_test_split(data,label, test_size=test_percent, random_state=42)


# In[89]:

clf1=svm.SVR(kernel='rbf',tol=0.0001,gamma='auto')
clf1.fit(training_data,training_data_label)
result1=clf1.predict(test_data)

abs(test_data_label-result1).mean()


# In[ ]:




# In[ ]:




# In[ ]:




# In[35]:

len(training_data)


# In[36]:

test_percent=0.2
train_train_data,train_test_data,train_train_label,train_test_label=train_test_split(training_data,training_data_label,test_size=test_percent,random_state=42)


# In[37]:

len(train_train_data)


# In[38]:

clf=svm.SVR(kernel='rbf',tol=0.0001,gamma='auto')


# In[39]:

clf.fit(train_train_data,train_train_label)


# In[40]:

svm_result=clf.predict(train_test_data)


# In[ ]:




# In[ ]:




# In[41]:

bottom_svc=(abs(svm_result-train_test_label)).sort_values(ascending=False).quantile(0.9)


# In[42]:

bottom_svc


# In[43]:

svm_keys=(((abs(svm_result-train_test_label))[(abs(svm_result-train_test_label))>bottom_svc])).keys()
svm_bad=(((abs(svm_result-train_test_label))[(abs(svm_result-train_test_label))>bottom_svc]))


# In[44]:

len(svm_keys)


# In[45]:

train_test_data.loc[svm_keys]


# In[46]:

from sklearn.neighbors import NearestNeighbors


# In[47]:

svm_keys


# In[48]:

len(svm_keys)


# In[49]:

train_test_data.loc[9649]


# In[50]:

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(train_train_data) 


# In[51]:

# bad neighbors in train_train_data
bad_neighbors=[]
for key in svm_keys:
    print key
    bad_neighbors.extend(neigh.kneighbors(train_test_data.loc[key].values.reshape(1,-1))[1][0].tolist())


# In[52]:

unique_bad_neighbors=(set(bad_neighbors))


# In[53]:

len(unique_bad_neighbors)


# In[54]:

clean_train_test_data=train_test_data.drop(svm_keys)


# In[55]:

len(clean_train_test_data)


# In[56]:

clean_train_test_label=train_test_label.drop(svm_keys)


# In[57]:

clean_train_train_data=train_train_data.drop(train_train_data.index[list(unique_bad_neighbors)])


# In[58]:

len(train_train_data)


# In[59]:

clean_train_train_label=train_train_label.drop(train_train_data.index[list(unique_bad_neighbors)])


# In[60]:

# clean the test data set


# In[61]:

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(test_data) 


# In[62]:

bad_neighbors_test=[]
for key in svm_keys:
    print key
    bad_neighbors_test.extend(neigh.kneighbors(train_test_data.loc[key].values.reshape(1,-1))[1][0].tolist())


# In[64]:

unique_bad_neighbors_test=set(bad_neighbors_test)


# In[65]:

len(unique_bad_neighbors_test)


# In[66]:

clean_test_data=test_data.drop(test_data.index[list(unique_bad_neighbors_test)])


# In[67]:

clean_test_label=test_data_label.drop(test_data_label.index[list(unique_bad_neighbors_test)])


# In[74]:

frame1=[clean_train_train_data,clean_train_test_data]
frame2=[clean_train_train_label,clean_train_test_label]


# In[75]:

clean_train_test_label


# In[76]:

clean_train_data=pd.concat(frame1)
clean_train_label=pd.concat(frame2)


# In[77]:

clean_train_label


# In[80]:

kernel_list=['linear','rbf','poly','sigmoid']

for kernel in kernel_list:
    clf=svm.SVR(kernel=kernel,tol=0.0001,gamma='auto')
    clf.fit(clean_train_data,clean_train_label)
    result=clf.predict(clean_test_data)
    
    abs(result-clean_test_label)
    


# In[90]:

clf2=svm.SVR(kernel='rbf',tol=0.0001,gamma='auto')
clf2.fit(clean_train_data,clean_train_label)
result2=clf2.predict(clean_test_data)

abs(clean_test_label-result2).mean()


# In[91]:

# we can notice that th accuracy error improve almost 20% , its very good.


# # Credit Risk Analytics

# In[92]:

credit=pd.read_csv('credit_data_.csv',header=None,names=['WC_TA','RE_TA','EBIT_TA','MVE_BVTD','S_TA','sector label'])


# In[93]:

len(credit)


# In[95]:

scaler2=StandardScaler()
credit=scaler2.fit_transform(credit)


# In[97]:

credit=pd.DataFrame(credit,columns=['WC_TA','RE_TA','EBIT_TA','MVE_BVTD','S_TA','sector label'])


# In[98]:

credit.head()


# In[99]:

target=[1 for i in range(1540)]


# In[100]:

target_n=[0 for i in range(130)]


# In[101]:

target.extend(target_n)


# In[102]:

len(target)


# In[103]:

from sklearn.model_selection import cross_val_score


# In[104]:

kernel_list=['linear','rbf','poly','sigmoid']

for kernel in kernel_list:
    k=10
    
    clf=svm.SVC(kernel=kernel,tol=0.0001,gamma='auto')
    scores=cross_val_score(clf,credit,target,cv=k)
    print "kernel: "+str(kernel)
    print scores 
    print "\n"
   
    


# In[105]:

clf=svm.SVC(kernel='rbf',tol=0.0001,gamma='auto')


# In[106]:

clf.fit(credit,target)


# In[107]:

# compare the eigenvalues of kernel matrics 


# In[108]:

from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import eigvals


# In[109]:

kernel_list=['linear','rbf','poly','sigmoid']

for kernel in kernel_list:
    k=10
    
    kernel_matrix=pairwise_kernels(credit,metric=kernel)
    print "Kernel is "+ str(kernel)
    
    print eigvals(kernel_matrix)
    
    print "\n"
    


# In[156]:

kernel_matrix.shape


# In[157]:

eigvals(kernel_matrix).shape


# In[144]:

target=pd.DataFrame(target)


# In[145]:

target.shape


# In[ ]:



