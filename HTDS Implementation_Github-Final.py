#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Implemenetation of HTDS Approcah


# In[32]:



# importing the datasets

import numpy as np
import pandas as pd
import pyswarms as ps
from scipy.io import arff

df1 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.3.csv')
df2 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.4.csv')
df3 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.5.csv')
df4 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.6.csv')
df5 = pd.read_csv(r'D:\datasets\JURECZKO\ant\ant-1.7.csv')
df6 = pd.read_csv(r'D:\datasets\JURECZKO\arc\arc.csv')
df7=pd.read_csv(r'D:\datasets\JURECZKO\berek\berek.csv')
df8=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.0.csv')
df9=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.2.csv')
df10=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.4.csv')
df11=pd.read_csv(r'D:\datasets\JURECZKO\camel\camel-1.6.csv')
df12=pd.read_csv(r'D:\datasets\JURECZKO\ckjm\ckjm.csv')
df13=pd.read_csv(r'D:\datasets\JURECZKO\elearning\e-learning.csv')
df14=pd.read_csv(r'D:\datasets\JURECZKO\forrest\forrest-0.7.csv')
df15=pd.read_csv(r'D:\datasets\JURECZKO\forrest\forrest-0.8.csv')
df16=pd.read_csv(r'D:\datasets\JURECZKO\intercafe\intercafe.csv')
df17=pd.read_csv(r'D:\datasets\JURECZKO\ivy\ivy-1.1.csv')
df18=pd.read_csv(r'D:\datasets\JURECZKO\ivy\ivy-1.4.csv')
df19=pd.read_csv(r'D:\datasets\JURECZKO\ivy\ivy-2.0.csv')
df20=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-3.2.csv')
df21=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.0.csv')
df22=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.1.csv')
df23=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.2.csv')
df24=pd.read_csv(r'D:\datasets\JURECZKO\jedit\jedit-4.3.csv')
df25=pd.read_csv(r'D:\datasets\JURECZKO\kalkulator\kalkulator.csv')
df26=pd.read_csv(r'D:\datasets\JURECZKO\log4j\log4j-1.0.csv')
df27=pd.read_csv(r'D:\datasets\JURECZKO\log4j\log4j-1.1.csv')
df28=pd.read_csv(r'D:\datasets\JURECZKO\log4j\log4j-1.2.csv')
df29=pd.read_csv(r'D:\datasets\JURECZKO\lucene\lucene-2.0.csv')
df30=pd.read_csv(r'D:\datasets\JURECZKO\lucene\lucene-2.2.csv')
df31=pd.read_csv(r'D:\datasets\JURECZKO\lucene\lucene-2.4.csv')
df32=pd.read_csv(r'D:\datasets\JURECZKO\nieruchomosci\nieruchomosci.csv')
df33=pd.read_csv(r'D:\datasets\JURECZKO\pbeans\pbeans1.csv')
df34=pd.read_csv(r'D:\datasets\JURECZKO\pbeans\pbeans2.csv')
df35=pd.read_csv(r'D:\datasets\JURECZKO\pdftranslator\pdftranslator.csv')
df36=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-1.5.csv')
df37=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-2.0.csv')
df38=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-2.5.csv')
df39=pd.read_csv(r'D:\datasets\JURECZKO\poi\poi-3.0.csv')
df40=pd.read_csv(r'D:\datasets\JURECZKO\redaktor\redaktor.csv')
df41=pd.read_csv(r'D:\datasets\JURECZKO\serapion\serapion.csv')
df42=pd.read_csv(r'D:\datasets\JURECZKO\skarbonka\skarbonka.csv')
df43=pd.read_csv(r'D:\datasets\JURECZKO\sklebagd\sklebagd.csv')
df44=pd.read_csv(r'D:\datasets\JURECZKO\synapse\synapse-1.0.csv')
df45=pd.read_csv(r'D:\datasets\JURECZKO\synapse\synapse-1.1.csv')
df46=pd.read_csv(r'D:\datasets\JURECZKO\synapse\synapse-1.2.csv')
df47=pd.read_csv(r'D:\datasets\JURECZKO\systemdata\systemdata.csv')
df48=pd.read_csv(r'D:\datasets\JURECZKO\szybkafucha\szybkafucha.csv')
df49=pd.read_csv(r'D:\datasets\JURECZKO\termoproject\termoproject.csv')
df50=pd.read_csv(r'D:\datasets\JURECZKO\tomcat\tomcat.csv')
df51=pd.read_csv(r'D:\datasets\JURECZKO\velocity\velocity-1.5.csv')
df52=pd.read_csv(r'D:\datasets\JURECZKO\velocity\velocity-1.6.csv')
df53=pd.read_csv(r'D:\datasets\JURECZKO\workflow\workflow.csv')
df54=pd.read_csv(r'D:\datasets\JURECZKO\wspomaganiepi\wspomaganiepi.csv')
df55=pd.read_csv(r'D:\datasets\JURECZKO\xalan\xalan-2.4.csv')
df56=pd.read_csv(r'D:\datasets\JURECZKO\xalan\xalan-2.5.csv')
df57=pd.read_csv(r'D:\datasets\JURECZKO\xalan\xalan-2.6.csv')
df58=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-init.csv')
df59=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-1.2.csv')
df60=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-1.3.csv')
df61=pd.read_csv(r'D:\datasets\JURECZKO\xerces\xerces-1.4.csv')
df62=pd.read_csv(r'D:\datasets\JURECZKO\zuzel\zuzel.csv')




# In[33]:


# filtering the features and the defect count from each dataset as first three features are representing informations related to  name and version

df1=df1.iloc[:,3:24]
df2=df2.iloc[:,3:24]
df3=df3.iloc[:,3:24]
df4=df4.iloc[:,3:24]
df5=df5.iloc[:,3:24]
df6=df6.iloc[:,3:24]
df7=df7.iloc[:,3:24]
df8=df8.iloc[:,3:24]
df9=df9.iloc[:,3:24]
df10=df10.iloc[:,3:24]
df11=df11.iloc[:,3:24]
df12=df12.iloc[:,3:24]
df13=df13.iloc[:,3:24]
df14=df14.iloc[:,3:24]
df15=df15.iloc[:,3:24]
df16=df16.iloc[:,3:24]
df17=df17.iloc[:,3:24]
df18=df18.iloc[:,3:24]
df19=df19.iloc[:,3:24]
df20=df20.iloc[:,3:24]
df21=df21.iloc[:,3:24]
df22=df22.iloc[:,3:24]
df23=df23.iloc[:,3:24]
df24=df24.iloc[:,3:24]
df25=df25.iloc[:,3:24]
df26=df26.iloc[:,3:24]
df27=df27.iloc[:,3:24]
df28=df28.iloc[:,3:24]
df29=df29.iloc[:,3:24]
df30=df30.iloc[:,3:24]
df31=df31.iloc[:,3:24]
df32=df32.iloc[:,3:24]
df33=df33.iloc[:,3:24]
df34=df34.iloc[:,3:24]
df35=df35.iloc[:,3:24]
df36=df36.iloc[:,3:24]
df37=df37.iloc[:,3:24]
df38=df38.iloc[:,3:24]
df39=df39.iloc[:,3:24]
df40=df40.iloc[:,3:24]
df41=df41.iloc[:,3:24]
df42=df42.iloc[:,3:24]
df43=df43.iloc[:,3:24]
df44=df44.iloc[:,3:24]
df45=df45.iloc[:,3:24]
df46=df46.iloc[:,3:24]
df47=df47.iloc[:,3:24]
df48=df48.iloc[:,3:24]
df49=df49.iloc[:,3:24]
df50=df50.iloc[:,3:24]
df51=df51.iloc[:,3:24]
df52=df52.iloc[:,3:24]
df53=df53.iloc[:,3:24]
df54=df54.iloc[:,3:24]
df55=df55.iloc[:,3:24]
df56=df56.iloc[:,3:24]
df57=df57.iloc[:,3:24]
df58=df58.iloc[:,3:24]
df59=df59.iloc[:,3:24]
df60=df60.iloc[:,3:24]
df61=df61.iloc[:,3:24]
df62=df62.iloc[:,3:24]


# In[34]:


# labeling the modules as faulty ( with integer value "1") if their defect count value is greater than 0, otherwise as  non-faulty with integer value "0"

df1['bug']=(df1["bug"]> 0).astype(int)
df2['bug'] = (df2['bug'] > 0).astype(int)
df3['bug']=(df3["bug"]> 0).astype(int)
df4['bug'] = (df4['bug'] > 0).astype(int)
df5['bug'] = (df5['bug'] > 0).astype(int)
df6['bug'] = (df6['bug'] > 0).astype(int)
df7['bug'] = (df7['bug'] > 0).astype(int)
df8['bug'] = (df8['bug'] > 0).astype(int)
df9['bug']=(df9["bug"]> 0).astype(int)
df10['bug'] = (df10['bug'] > 0).astype(int)
df11['bug']=(df11["bug"]> 0).astype(int)
df12['bug'] = (df12['bug'] > 0).astype(int)
df13['bug'] = (df13['bug'] > 0).astype(int)
df14['bug'] = (df14['bug'] > 0).astype(int)
df15['bug'] = (df15['bug'] > 0).astype(int)
df16['bug'] = (df16['bug'] > 0).astype(int)
df17['bug']=(df17["bug"]> 0).astype(int)
df18['bug'] = (df18['bug'] > 0).astype(int)
df19['bug']=(df19["bug"]> 0).astype(int)
df20['bug'] = (df20['bug'] > 0).astype(int)
df21['bug'] = (df21['bug'] > 0).astype(int)
df22['bug'] = (df22['bug'] > 0).astype(int)
df23['bug'] = (df23['bug'] > 0).astype(int)
df24['bug'] = (df24['bug'] > 0).astype(int)

df25['bug']=(df25["bug"]> 0).astype(int)
df26['bug'] = (df26['bug'] > 0).astype(int)
df27['bug']=(df27["bug"]> 0).astype(int)
df28['bug'] = (df28['bug'] > 0).astype(int)
df29['bug'] = (df29['bug'] > 0).astype(int)
df30['bug'] = (df30['bug'] > 0).astype(int)
df31['bug'] = (df31['bug'] > 0).astype(int)
df32['bug'] = (df32['bug'] > 0).astype(int)
df33['bug']=(df33["bug"]> 0).astype(int)
df34['bug'] = (df34['bug'] > 0).astype(int)
df35['bug']=(df35["bug"]> 0).astype(int)
df36['bug'] = (df36['bug'] > 0).astype(int)
df37['bug'] = (df37['bug'] > 0).astype(int)
df38['bug'] = (df38['bug'] > 0).astype(int)
df39['bug'] = (df39['bug'] > 0).astype(int)
df40['bug'] = (df40['bug'] > 0).astype(int)
df41['bug']=(df41["bug"]> 0).astype(int)
df42['bug'] = (df42['bug'] > 0).astype(int)
df43['bug']=(df43["bug"]> 0).astype(int)
df44['bug'] = (df44['bug'] > 0).astype(int)
df45['bug'] = (df45['bug'] > 0).astype(int)
df46['bug'] = (df46['bug'] > 0).astype(int)
df47['bug'] = (df47['bug'] > 0).astype(int)
df48['bug'] = (df48['bug'] > 0).astype(int)
df49['bug'] = (df49['bug'] > 0).astype(int)
df50['bug']=(df50["bug"]> 0).astype(int)
df51['bug'] = (df51['bug'] > 0).astype(int)
df52['bug'] = (df52['bug'] > 0).astype(int)
df53['bug'] = (df53['bug'] > 0).astype(int)
df54['bug'] = (df54['bug'] > 0).astype(int)
df55['bug'] = (df55['bug'] > 0).astype(int)
df56['bug']=(df56["bug"]> 0).astype(int)
df57['bug'] = (df57['bug'] > 0).astype(int)
df58['bug']=(df58["bug"]> 0).astype(int)
df59['bug'] = (df59['bug'] > 0).astype(int)
df60['bug'] = (df60['bug'] > 0).astype(int)
df61['bug'] = (df61['bug'] > 0).astype(int)
df62['bug'] = (df62['bug'] > 0).astype(int)


# In[35]:


# Eliminating the instances with zero LOC

df1=df1[df1["loc"]>0]
df2=df2[df2["loc"]>0]
df3=df3[df3["loc"]>0]
df4=df4[df4["loc"]>0]
df5=df5[df5["loc"]>0]
df6=df6[df6["loc"]>0]
df7=df7[df7["loc"]>0]
df8=df8[df8["loc"]>0]
df9=df9[df9["loc"]>0]
df10=df10[df10["loc"]>0]
df11=df11[df11["loc"]>0]
df12=df12[df12["loc"]>0]
df13=df13[df13["loc"]>0]
df14=df14[df14["loc"]>0]
df15=df15[df15["loc"]>0]
df16=df16[df16["loc"]>0]
df17=df17[df17["loc"]>0]
df18=df18[df18["loc"]>0]
df19=df19[df19["loc"]>0]
df20=df20[df20["loc"]>0]
df21=df21[df21["loc"]>0]
df22=df22[df22["loc"]>0]
df23=df23[df23["loc"]>0]
df24=df24[df24["loc"]>0]
df25=df25[df25["loc"]>0]
df26=df26[df26["loc"]>0]
df27=df27[df27["loc"]>0]
df28=df28[df28["loc"]>0]
df29=df29[df29["loc"]>0]
df30=df30[df30["loc"]>0]

df31=df31[df31["loc"]>0]
df32=df32[df32["loc"]>0]
df33=df33[df33["loc"]>0]
df34=df34[df34["loc"]>0]
df35=df35[df35["loc"]>0]
df36=df36[df36["loc"]>0]
df37=df37[df37["loc"]>0]
df38=df38[df38["loc"]>0]
df39=df39[df39["loc"]>0]
df40=df40[df40["loc"]>0]
df41=df41[df41["loc"]>0]
df42=df42[df42["loc"]>0]
df43=df43[df43["loc"]>0]
df44=df44[df44["loc"]>0]
df45=df45[df45["loc"]>0]

df46=df46[df46["loc"]>0]
df47=df47[df47["loc"]>0]
df48=df48[df48["loc"]>0]
df49=df49[df49["loc"]>0]
df50=df50[df50["loc"]>0]
df51=df51[df51["loc"]>0]
df52=df52[df52["loc"]>0]
df53=df53[df53["loc"]>0]
df54=df54[df54["loc"]>0]
df55=df55[df55["loc"]>0]
df56=df56[df56["loc"]>0]
df57=df57[df57["loc"]>0]
df58=df58[df58["loc"]>0]
df59=df59[df59["loc"]>0]
df60=df60[df60["loc"]>0]
df61=df61[df61["loc"]>0]
df62=df62[df62["loc"]>0]


# In[36]:


df1=df1.reset_index(drop=True)
df2=df2.reset_index(drop=True)
df3=df3.reset_index(drop=True)
df4=df4.reset_index(drop=True)
df5=df5.reset_index(drop=True)
df6=df6.reset_index(drop=True)
df7=df7.reset_index(drop=True)
df8=df8.reset_index(drop=True)
df9=df9.reset_index(drop=True)
df10=df10.reset_index(drop=True)
df11=df11.reset_index(drop=True)
df12=df12.reset_index(drop=True)
df13=df13.reset_index(drop=True)
df14=df14.reset_index(drop=True)
df15=df15.reset_index(drop=True)
df16=df16.reset_index(drop=True)
df17=df17.reset_index(drop=True)
df18=df18.reset_index(drop=True)
df19=df19.reset_index(drop=True)
df20=df20.reset_index(drop=True)
df21=df21.reset_index(drop=True)
df22=df22.reset_index(drop=True)
df23=df23.reset_index(drop=True)
df24=df24.reset_index(drop=True)
df25=df25.reset_index(drop=True)
df26=df26.reset_index(drop=True)
df27=df27.reset_index(drop=True)
df28=df28.reset_index(drop=True)
df29=df29.reset_index(drop=True)
df30=df30.reset_index(drop=True)
df31=df31.reset_index(drop=True)
df32=df32.reset_index(drop=True)
df33=df33.reset_index(drop=True)
df34=df34.reset_index(drop=True)
df35=df35.reset_index(drop=True)
df36=df36.reset_index(drop=True)
df37=df37.reset_index(drop=True)
df38=df38.reset_index(drop=True)
df39=df39.reset_index(drop=True)
df40=df40.reset_index(drop=True)
df41=df41.reset_index(drop=True)
df42=df42.reset_index(drop=True)
df43=df43.reset_index(drop=True)
df44=df44.reset_index(drop=True)
df45=df45.reset_index(drop=True)
df46=df46.reset_index(drop=True)
df47=df47.reset_index(drop=True)
df48=df48.reset_index(drop=True)
df49=df49.reset_index(drop=True)
df50=df50.reset_index(drop=True)
df51=df51.reset_index(drop=True)
df52=df52.reset_index(drop=True)
df53=df53.reset_index(drop=True)

df54=df54.reset_index(drop=True)
df55=df55.reset_index(drop=True)
df56=df56.reset_index(drop=True)
df57=df57.reset_index(drop=True)
df58=df58.reset_index(drop=True)
df59=df59.reset_index(drop=True)
df60=df60.reset_index(drop=True)
df61=df61.reset_index(drop=True)
df62=df62.reset_index(drop=True)


# In[37]:


# combining all the datasets into a single dataset with keys as their names

datasets=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,df59,df60,df61,df62], keys=['ant1','ant2','ant3','ant4','ant5','arc1','berek1','camel1','camel2','camel3','camel4','ckjm1','elearn1','forrest2','forrest3','intercafe1','ivy1','ivy2','ivy3','jedit1','jedit2','jedit3','jedit4','jedit5','kalkulator1','log4j1','log4j2','log4j3','lucene1','lucene2','lucene3','nieru1','pbeans1','pbeans2','pdftranslator1','poi1','poi2','poi3','poi4','redaktor1','serapion1','skarbonka1','sklebagd1','synapse1','synapse2','synapse3','systemdata1','szybkafucha1','termoproject1','tomcat1','velocity2','velocity3','workflow1','wspomagani1','xalan1','xalan2','xalan3','xerces1','xerces2','xerces3','xerces4','zuzel1'])
print(datasets)


# In[38]:



list2=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,df59,df60,df61,df62]


# In[39]:


# list1  contains the name of all the projects, where the number along with it helps in differentiating between its different versions such that 'ant1' is the initial version, 'ant2' is the next relaeased version,and 'ant3' is the version released after 'ant2' and so on. 

list1=['ant1','ant2','ant3','ant4','ant5','arc1','berek1','camel1','camel2','camel3','camel4','ckjm1','elearn1','forrest2','forrest3','intercafe1','ivy1','ivy2','ivy3','jedit1','jedit2','jedit3','jedit4','jedit5','kalkulator1','log4j1','log4j2','log4j3','lucene1','lucene2','lucene3','nieru1','pbeans1','pbeans2','pdftranslator1','poi1','poi2','poi3','poi4','redaktor1','serapion1','skarbonka1','sklebagd1','synapse1','synapse2','synapse3','systemdata1','szybkafucha1','termoproject1','tomcat1','velocity2','velocity3','workflow1','wspomagani1','xalan1','xalan2','xalan3','xerces1','xerces2','xerces3','xerces4','zuzel1']


# In[40]:


from math import sqrt

# Function to calculate the Euclidean distance between two vectors row1" and "row2"

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Function to find 'num_neighbors' nearest instances  of target data instance 'test_row' from the source dataset 'train' 

def get_neighbors(train, test_row, num_neighbors):
	distances = []
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


# In[41]:


# This function will accept  name, index and the list of the all  related releases (previous and furure both) of a particular target project

# index of a target project means its location in 'list1', for example, "ant1" target project has index value "0" as it is at '0th' position in 'list1'
# Taking all inputs, it will generate the the source data for that particular target project by removing all its versions, if any
# output will contain the source dataset "train_data", the target project dataset "X_test1" and its defect labels "y_test"

def KNN (target_data_label, target_data_index,list_related_releases):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    get_ipython().run_line_magic('matplotlib', 'inline')
    import numpy as np
    from sklearn import metrics
    list1=['ant1','ant2','ant3','ant4','ant5','arc1','berek1','camel1','camel2','camel3','camel4','ckjm1','elearn1','forrest2','forrest3','intercafe1','ivy1','ivy2','ivy3','jedit1','jedit2','jedit3','jedit4','jedit5','kalkulator1','log4j1','log4j2','log4j3','lucene1','lucene2','lucene3','nieru1','pbeans1','pbeans2','pdftranslator1','poi1','poi2','poi3','poi4','redaktor1','serapion1','skarbonka1','sklebagd1','synapse1','synapse2','synapse3','systemdata1','szybkafucha1','termoproject1','tomcat1','velocity2','velocity3','workflow1','wspomagani1','xalan1','xalan2','xalan3','xerces1','xerces2','xerces3','xerces4','zuzel1']


    j=target_data_index
    result=[]
    if list_related_releases:
        X=datasets.drop(index=target_data_label, level=0) # removing the target project first
        X=X.drop(index=list_related_releases, level=0) # removing all previous and future versions of the target project, if exists
    else:
         X=datasets.drop(index=target_data_label, level=0)
    
    Z=X.copy(deep=True)
   
    X_test=list2[j] # target project 
    X_test1= X_test.drop("bug",axis=1)
    y_test = list2[j]['bug']
    X_train=X.values.tolist()
    n=len(X_test1)
    neighbour=[]
    for i in range(n):
        a=np.array(X_test1.iloc[i,:])
        b=get_neighbors(X_train,a,10)
        neighbour.append(b)
    listOfList = neighbour
 
   # Use list comprehension to convert a list of lists to a flat list 
    flatList = [ item for elem in listOfList for item in elem]
 
    l2=['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3',
       'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc',
       'avg_cc', 'bug']
    train_data=pd.DataFrame(flatList,columns=l2)
    train_data.drop_duplicates(keep='first',inplace=True) 
    train_data=train_data.reset_index(drop=True)
    return train_data,X_test1,y_test


# In[42]:


# function to compute confusion matrix based on actual and predicted labels
def compute_tp_tn_modified(actual,pred):
    tp=sum((actual==1)&(pred==1))
    tn=sum((actual==0)&(pred==0))
    fp=sum((actual==0)&(pred==1))
    fn=sum((actual==1)&(pred==0))
    return tp,tn,fp,fn


# In[43]:


#function for applying SMOTE to handle class imbalance issue
def smote(X_train,y_train):
    
    from imblearn.over_sampling import SMOTE
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)
    return X_train,y_train


# In[44]:


# it will return the list of all related versions of a target project having index 'target_data_index' excluding the target project
def find_related_release(list1,target_data_index):
    releases=[]
    a=list1[target_data_index]
    c=a[0:-1]
    for i in range(0,62):
        word=list1[i]
        if word.find(c) != -1:
            releases.append(word)
    releases.remove(a)  
    return releases


# In[45]:


""" This function will take target data "X_test", probability score "prob_score" generated by NB classifier , 
predicted labels "y_pred" generated by NB classifier and actual labels " y_test" 
for preparing a dataframe containing all faulty and all non-faulty modules in order, sorted by {(prob_score/loc)*avg_cc } individually.
This dataframe will be used in three functions namely "PII_20, costeffort20 and IFA20" to calculate effort based measures.
"""
def sorted_df_epm1 (X_test, prob_score,  y_pred,y_test):
    
    import pandas as pd
    
    import numpy as np
   
    test_df_pred=pd.DataFrame()
    test_df_pred=X_test.copy(deep=True)
    test_df_pred['score']=prob_score
    test_df_pred['risk']=np.multiply(np.divide(prob_score,X_test["loc"]),X_test["avg_cc"])
   
    test_df_pred['pred']=  y_pred
    test_df_pred['actual']=y_test
    
    defective=pd.DataFrame()
    non_defective=pd.DataFrame()
    defective=test_df_pred[test_df_pred.pred==1].copy()
    
    
    non_defective=test_df_pred[test_df_pred.pred==0].copy()
    
    defective.sort_values(["risk"] ,axis = 0, ascending = [False], 
                inplace = True)
    defective=defective.reset_index(drop=True)
    
    non_defective.sort_values(["risk"], axis = 0, ascending = [False], 
                 inplace = True)
    non_defective=non_defective.reset_index(drop=True)
    target=pd.DataFrame()
    target=pd.concat([defective,non_defective],axis=0) 
    target=target.reset_index(drop=True)
    
    return target


# In[46]:


# function to calculate PII@20 effort based measure
def PII_20(target_data):
    import pandas as pd
    d=pd.DataFrame()
    d= target_data.copy(deep=True)
    import numpy as np
    total_loc=np.sum(d["loc"])
    
    m=0.2*total_loc
    
    sum_loc=0
    count_module=0
    for i in range(len(d)):
        sum_loc=sum_loc+d.loc[i,"loc"]
        if sum_loc>m:
            count_module=i
            break

            
    M=d.shape[0] # total modules
    
    return count_module/M


# In[47]:


# function to calculate CostEffort@20 effort based measure
def costeffort20(target_data):
    import pandas as pd
    d=pd.DataFrame()
    d_defective=pd.DataFrame()
    d= target_data.copy(deep=True)
    import numpy as np
    total_loc=np.sum(d["loc"])
    m=0.2*total_loc
    sum_loc=0
    count_module=0
    count_actualdefective_module=0
    total_defective=0
    for i in range(len(d)):
        sum_loc=sum_loc+d.loc[i,"loc"]
        if sum_loc>m:
            count_module=i
            break
        
    for i in range(0,count_module):
        if d.loc[i,"actual"]==1:
            count_actualdefective_module+=1
    d_defective=d[d.actual==1]
    total_defective=len(d_defective)
    cost=count_actualdefective_module/ total_defective
    return cost


# In[48]:


# Function to calculate IFA 

def IFA20(target_data):
    import pandas as pd
    d=pd.DataFrame()
    d= target_data.copy(deep=True)
    
    
    import numpy as np
    total_loc=np.sum(d["loc"])
    m=0.2*total_loc
    sum_loc=0
    count_module=0
    count_actualdefective_module=0
    total_defective=0
    for i in range(len(d)):
        sum_loc=sum_loc+d.loc[i,"loc"]
        if sum_loc>m:
            count_module=i
            break
        else:
            continue
    false_alarm=0
    for i in range(0,count_module): 
        if d.loc[i,"pred"]==1 and d.loc[i,"actual"]==1:
            break
        else:
            false_alarm+=1
            
    return false_alarm


# In[49]:


# The pyswarms library is used for BPSO implementation.


# In[50]:


def f_per_particle(m):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score,precision_score,roc_auc_score
    from sklearn.metrics import matthews_corrcoef
    
    total_features = 20
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = new_source # new source is the source data retrieved after the application of smote
        X_test=target_test # target_test is the 20% of the target data to be used in the feature selection phase
        
    
    else:
        X_subset = new_source.iloc[:,m==1]
        X_test=target_test.iloc[:,m==1]
        
    # Perform classification and store performance in P
    clf.fit(X_subset, y_train)  # clf is the NB classifier
    y_pred =clf.predict(X_test)
    rec=recall_score(target_test_labels,y_pred)
    
    tp=sum((target_test_labels==1)&(y_pred==1))
    tn=sum((target_test_labels==0)&(y_pred==0))
    fp=sum((target_test_labels==0)&(y_pred==1))
    fn=sum((target_test_labels==1)&(y_pred==0))
    
    #tn, fp, fn, tp = confusion_matrix(target_test_labels,y_pred).ravel()
    prob_false_alarm=fp/(fp+tn) 
    a=1-prob_false_alarm
    g_measure=2*rec*a/(rec+a)
    
    
    P=1-g_measure

    return P


# In[ ]:





# In[51]:


def f(x):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed performance for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i]) for i in range(n_particles)]
    return np.array(j)


# In[52]:



avg_result=[]


# In[53]:


""" Execution of the approach starts from here. 
for every target project, the instance selection phase is applied first which constitutes calling the 'KNN' function and 
then the 'smote' function in sequence.After that, feature selection phase will be executed 10 times for that target project 
and mean of the NEPMs and EPMs are calculated for that target project. The entire process is repeated for all 62 datasets 
and average of the NEPMs and EPMs are calculated over all datasets.

"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,precision_score,roc_auc_score
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import matthews_corrcoef
from sklearn import linear_model
import math

import numpy as np

for k in range(0,62): #  we have set this loop to run 62 times for 62 datasets
    result_target_project=[]
    
    # applying KNN function to retrieve the source data for a particular target project  
    print("KNN in progress on target project",list1[k])
    related_projects_list=find_related_release(list1,k)
    train_data,X_test,y_test=KNN(list1[k],k,related_projects_list)

    # applying smote on the source data retreievd from KNN function    
    print("SMOTE in progress on target project",list1[k])
    X_train12= train_data.iloc[:,0:20]
    y_train12= train_data.iloc[:,-1]
    X_train2,y_train2=smote(X_train12,y_train12)
    X_train1=pd.DataFrame(X_train2,columns=['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3',
       'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc',
       'avg_cc'])
    
    X_train1["bug"]=y_train2
    final_train_data= X_train1

    """
    Here, BPSO is called 10 times, in every iteration, it is randomly selecting the features from the source data  retreived 
    after applying smote and validating its performance on 20% of the target data. After we obtain the imporatnt features, 
    the source and the target data are modified to have these selected features only. 
    Finally, a NB model is constructed with modified source data and is validated on rest 80% of
    target data containing selected features only. The NEPMs and EPMs are calculated and finally the mean of 10 rounds are calculated.
    """
    print("feature selection phase started on target project",list1[k])
    for j in range(0,10):
        print("\n Iteration", j+1)
        y_train=final_train_data.iloc[:,-1]
        y_actual=y_test
        new_source=final_train_data.iloc[:,0:20]
        new_target=X_test.iloc[:,0:20]
        target_train, target_test, target_train_labels, target_test_labels = train_test_split(new_target, y_actual, test_size=0.2,random_state=0)
    
        clf = GaussianNB()
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

    # Call instance of BPSO
        dimensions = 20 # dimensions should be the number of features

        optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
    # Perform optimization
        cost, pos = optimizer.optimize(f,  iters=250, verbose= True)
   
    # now building and testing the final classifier on source data with selected features and target data with selected features respectively
        classifier = GaussianNB()

        X_train_selected_features=new_source.iloc[:,pos==1]

      # testing on 80% of target data
        X_test_selected_features=target_train.iloc[:,pos==1]


        classifier.fit(X_train_selected_features,y_train)

        y_pred2=classifier.predict(X_test_selected_features)
        prob_score=classifier.predict_proba(X_test_selected_features)[:,1]
    # calculation of NEPMs
        rec1=recall_score(target_train_labels,y_pred2)
        tn, fp, fn, tp = confusion_matrix(target_train_labels,y_pred2).ravel()
        pf=fp/(fp+tn) # PF
        a=1-pf
        g_measure=2*rec1*a/(rec1+a)
        mcc=matthews_corrcoef(target_train_labels,y_pred2)

    # calculation of EPMs
    
        target2=sorted_df_epm1 (target_train, prob_score,  y_pred2, target_train_labels)
    
        PII=PII_20(target2)  
        costeffort=costeffort20(target2)  
        IFA=IFA20(target2)
        NEPM_EPM_list=[g_measure,mcc,pf,PII,costeffort,IFA]
        result_target_project.append(NEPM_EPM_list) 
    t=np.mean(result_target_project, axis=0)
    avg_result.append(t)

output= np.mean(avg_result, axis=0)  


# In[54]:


print("the average G-measure, MCC, PF, PII@20, CostEffort@20 and IFA are:\n", output)


# In[ ]:




