import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('C:\\Users\\User\\Downloads\\archive\\Iris.csv')

# Adjust the aspect ratio of the pairplot
#sns.pairplot(df, hue='Species', aspect=1.2)
#plt.show()

#Print the entire dataframe
print(df.to_string())