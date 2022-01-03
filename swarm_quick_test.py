import pandas as pd

print("START!")
df = pd.DataFrame({'a': [1,2,3,4], 'b': [5,6,7,8]})
df.to_csv('/home/jasonvallada/test.csv')
print("DONE!")
