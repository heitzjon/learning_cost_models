import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

PATH = "tables/"



def createRandomData(name, spec, len):
    content=pd.DataFrame()
    for row in spec:
        v=np.random.randint(0,row['max_val'],len)
        content[row['name']]=pd.Series(v)
    print(name)
    print(content.head())
    content.to_csv(PATH + name + ".csv", sep=',',index=False, encoding='utf-8')
    return content


def createCorrelatedData(name, spec, length):
    columns = []
    max_val = []
    for row in spec:
        columns.append(row['name'])
        max_val.append(int(row['max_val']))
    content=pd.DataFrame(columns=columns)
    for i in range(0, length):
        x=[]
        for j in range(0, len(max_val)):
            x.append(int(i / (length / max_val[j])))
        content.loc[i] = x
    print(name)
    print(content.head())
    content.to_csv(PATH + name + ".csv", sep=',',index=False, encoding='utf-8')
    return content

print(dataToCreate.keys())
for key, value in dataToCreate.items():
    if value['type'] is "rand":
        createRandomData(key,dataToCreate[key]['rows'],int(value['len']))
    elif value['type'] is "corr":
        createCorrelatedData(key, dataToCreate[key]['rows'], int(value['len']))


#x = createRandomData('x',dataToCreate.get('a').get('rows'),10)
#x = createCorrelatedData('y',dataToCreate['b']['rows'],1000)
#plt.scatter(x['v'], x['u'])
#plt.scatter(x['w'], x['u'], c='red')
#plt.show()
