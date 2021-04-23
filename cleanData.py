import pandas as pd

# Clean housing data
df = pd.read_csv('data/housing.csv')

# Column 0 is the index column, not needed
df = df.drop(columns=df.columns[0], axis=1)

print(df)
df.to_csv('data/housingCleaned.csv', index=False, header=False)

# Clean bikes data
df = pd.read_csv('data/bikes.csv')

# Column 0 is the index column and column 1 is repeated date information, not needed
# The second and third to last columns always sum to the target value, hence they trivialize the task and are dropped
df = df.drop(columns=df.columns[[0, 1, -2, -3]], axis=1)

print(df)
df.to_csv('data/bikesCleaned.csv', index=False, header=False)