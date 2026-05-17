#Students have to create relationship and correlation charts for checking the relationship between various features of the given dataset.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv('your_dataset.csv')
# Create a pairplot to visualize relationships between features
sns.pairplot(data)
plt.show()
# Create a correlation heatmap
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()  
# Create a scatter plot for two specific features
sns.scatterplot(x='Feature1', y='Feature2', data=data)
plt.show()
# Create a regression plot to check the relationship between two features
sns.regplot(x='Feature1', y='Feature2', data=data)
plt.show()