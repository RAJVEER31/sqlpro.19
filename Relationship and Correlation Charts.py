import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv('FuelConsumption.csv')
# Check for null values
print(data.isnull().sum())
# Check the details of the dataset
print(data.info())
# --- Fix column name mismatches (dataset columns are uppercase with underscores) ---
# Print columns once if needed: print(data.columns)

x_col = 'FUELCONSUMPTION_COMB_MPG'
y_col = 'CO2EMISSIONS'

# Make sure we have expected columns (avoid seaborn ValueError)
missing = [c for c in (x_col, y_col) if c not in data.columns]
if missing:
    raise KeyError(f"Missing columns in FuelConsumption.csv: {missing}. Available columns: {list(data.columns)}")

# Relationship between CO2 emissions and fuel consumption comb. Mpg
sns.scatterplot(x=x_col, y=y_col, data=data)
plt.title('Relationship between CO2 Emissions and Combined MPG')
plt.xlabel('Combined MPG')
plt.ylabel('CO2 Emissions')
plt.show()

# Relationship between CO2 emissions and fuel consumption comb. Mpg for different fuel types and engine
hue_col = 'FUELTYPE'
size_col = 'ENGINESIZE'
missing = [c for c in (hue_col, size_col) if c not in data.columns]
if missing:
    raise KeyError(f"Missing columns needed for hue/size: {missing}. Available columns: {list(data.columns)}")

sns.scatterplot(x=x_col, y=y_col, hue=hue_col, size=size_col, data=data)
plt.title('Relationship between CO2 Emissions and Combined MPG by Fuel Type and Engine Size')
plt.xlabel('Combined MPG')
plt.ylabel('CO2 Emissions')
plt.legend(title='Fuel Type')
plt.show()

# Correlation between all numeric features
correlation_matrix = data.select_dtypes(include='number').corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Relationship between CO2 emissions and fuel consumption comb. Mpg for different fuel types individually
sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=data)
plt.title('Relationship between CO2 Emissions and Combined MPG by Fuel Type')
plt.xlabel('Combined MPG')
plt.ylabel('CO2 Emissions')
plt.legend(title='Fuel Type')
plt.show()

# Relationship between all features with fuel type as hue
# (pairplot works best with numeric columns; keep hue categorical)
sns.pairplot(data.select_dtypes(include='number').join(data[[hue_col]]), hue=hue_col, palette='Set2')
plt.suptitle('Pairplot of Numeric Features with Fuel Type as Hue', y=1.02)
plt.show()
