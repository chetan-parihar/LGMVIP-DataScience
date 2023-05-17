# Download the groceries dataset. Write a python program to read the dataset and display its
# information. Preprocess the data (drop null values etc.) Convert the categorical values into numeric
# format. Apply the apriori algorithm on the above dataset to generate the frequent itemsets and association
# rules.

import pandas as pd
# from mlxtend.frequent_patterns import apriori,association_rules
# from mlxtend.preprocessing import TransactionEncoder

# transaction=[["eggs","milk","bread"],
#             ["eggs","apple"],
#             ['milk','bread'],
#             ['apple','milk'],
#             ['milk','apple','bread'],
#             ['milk','apple']]


# model=TransactionEncoder()
# model_arr=model.fit(transaction).transform(transaction)
# df=pd.DataFrame(model_arr,columns=model.columns_)
# print(df)

# df=df.dropna()
# freq_items= apriori(df,min_support=0.5,use_colnames=True)
# print(freq_items)


# rules=association_rules(freq_items,metric="support",min_threshold=0.5)
# rules=rules.sort_values(['support',"confidence"],ascending=[False,False])
# print(rules)


from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

data=[["eggs","milk","bread"],
            ["eggs","apple"],
            ['milk','bread'],
            ['apple','milk'],
            ['milk','apple','bread'],
            ['milk','apple']]

model=TransactionEncoder()
data_arr=model.fit(data).transform(data)
df=pd.DataFrame(data_arr,columns=model.columns_)

df=df.dropna()
freq_items=apriori(df)
print(freq_items)

rules=association_rules(freq_items,metric="support",min_threshold=0.5)
# rules=rules.sort_values(['support',"confidence"],ascending=[False,False])
print(rules)