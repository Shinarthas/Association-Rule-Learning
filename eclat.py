#Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# говорим что у нас нет хедера в csv
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#пересоздадим данные которые подойдут классу которые мы будем исп
#сделаем это в виде списка
transactions=[]
for row in dataset.values:
    transactions.append([str(el)  for el in row if str(el)!='nan'])

#подключим модуль априорной вероятности
from apyori import apriori
#просто совпадение что аргумент это название элемента
# следующий параметр, минимальный уровень потдержки, для того что-бы не считать все комбинации
# на сколько часто встречались эти комбинации, мы выбрали 3 раза в день
#ищем комбинацию только из 2 продуктов
rules=apriori(transactions=transactions,min_support=3*7/len(transactions), min_confidence=0.2, min_lift=3, min_length=2,max_length=2)
# min_support как часто встречается комбинация
# min_confidence как часто продукт B покупался с продуктом A

#ВИЗУАЛИЗАЦИЯ

# отобразим первый результат
result=list(rules)
print(result[0])
print(result[1])
print(result[2])

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(result), columns = ['Left Hand Side', '| Right Hand Side', '| Support'])

## Displaying the results non sorted
print(resultsinDataFrame)
print('-------------')
## Displaying the results sorted by descending support
print(resultsinDataFrame.nlargest(n = 10, columns = '| Support'))