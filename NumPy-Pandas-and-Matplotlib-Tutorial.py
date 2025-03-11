# ========================================================================
# COMPREHENSIVE PYTHON DATA SCIENCE TUTORIAL
# NumPy, Pandas, and Matplotlib: Essential Guide
# ========================================================================

"""
This tutorial provides a concise yet comprehensive introduction to the 
essential Python data science stack: NumPy, Pandas, and Matplotlib.

By the end, you'll understand:
- NumPy's powerful array operations and mathematical functions
- Pandas' data manipulation and analysis capabilities
- Matplotlib's visualization tools
- How these libraries work together for data science tasks
- Real-world applications in AI and data processing

Author: Claude
Date: March 11, 2025
"""

# ========================================================================
# PART 1: NUMPY - NUMERICAL PYTHON
# ========================================================================

import numpy as np

print("=" * 50)
print("NUMPY TUTORIAL")
print("=" * 50)

# ---------- 1.1 NUMPY ARRAYS ----------

# NumPy's main object is the homogeneous multidimensional array
# Creating arrays
print("\n1.1 NumPy Arrays")

# From Python lists
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"1D array: {array_1d}")
print(f"2D array:\n{array_2d}")

# Array properties
print(f"\nArray shape: {array_2d.shape}")  # Dimensions
print(f"Array dimensions: {array_2d.ndim}")
print(f"Array data type: {array_2d.dtype}")
print(f"Array size: {array_2d.size}")  # Total number of elements

# Special arrays
print("\nSpecial arrays:")
zeros = np.zeros((2, 3))  # 2x3 array of zeros
ones = np.ones((2, 2))    # 2x2 array of ones
identity = np.eye(3)      # 3x3 identity matrix
sequence = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced points from 0 to 1

print(f"Zeros:\n{zeros}")
print(f"Ones:\n{ones}")
print(f"Identity:\n{identity}")
print(f"Sequence: {sequence}")
print(f"Linspace: {linspace}")

# Random arrays
print("\nRandom arrays:")
uniform = np.random.rand(2, 2)  # Random values from uniform distribution [0,1)
normal = np.random.randn(2, 2)  # Random values from standard normal distribution
integers = np.random.randint(0, 10, (2, 3))  # Random integers from 0 to 9

print(f"Uniform random:\n{uniform}")
print(f"Normal random:\n{normal}")
print(f"Random integers:\n{integers}")

# ---------- 1.2 ARRAY OPERATIONS ----------

print("\n1.2 NumPy Array Operations")

# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a + b = {a + b}")  # Element-wise addition
print(f"a - b = {a - b}")  # Element-wise subtraction
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"a / b = {a / b}")  # Element-wise division
print(f"a ** 2 = {a ** 2}")  # Element-wise exponentiation
print(f"a > 2: {a > 2}")  # Element-wise comparison

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\nMatrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"Matrix multiplication (A @ B):\n{A @ B}")  # Or np.matmul(A, B)
print(f"Matrix transpose (A.T):\n{A.T}")

# ---------- 1.3 ARRAY INDEXING AND SLICING ----------

print("\n1.3 Array Indexing and Slicing")

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Original array:\n{arr}")

# Basic indexing
print(f"\nElement at (0,0): {arr[0, 0]}")
print(f"Element at (2,3): {arr[2, 3]}")

# Slicing [start:stop:step]
print(f"\nFirst row: {arr[0, :]}")
print(f"First column: {arr[:, 0]}")
print(f"Subarray (first 2 rows, last 2 columns):\n{arr[:2, 2:]}")

# Boolean indexing
mask = arr > 5
print(f"\nMask (elements > 5):\n{mask}")
print(f"Elements > 5: {arr[mask]}")

# Fancy indexing
indices = np.array([0, 2])  # Select rows 0 and 2
print(f"\nRows 0 and 2:\n{arr[indices, :]}")

# ---------- 1.4 ARRAY MANIPULATION ----------

print("\n1.4 Array Manipulation")

# Reshaping
a = np.arange(12)
print(f"Original 1D array: {a}")
b = a.reshape(3, 4)  # Reshape to 3x4
print(f"Reshaped 3x4:\n{b}")
c = a.reshape(2, 2, 3)  # Reshape to 2x2x3
print(f"Reshaped 2x2x3:\n{c}")

# Concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
vertical = np.vstack((a, b))  # Vertical stack
horizontal = np.hstack((a, b))  # Horizontal stack
print(f"\nVertical stack:\n{vertical}")
print(f"Horizontal stack:\n{horizontal}")

# Splitting
arr = np.arange(16).reshape(4, 4)
print(f"\nOriginal array:\n{arr}")
vertical_split = np.vsplit(arr, 2)  # Split into 2 along first axis
horizontal_split = np.hsplit(arr, 2)  # Split into 2 along second axis
print(f"Vertical split (first part):\n{vertical_split[0]}")
print(f"Horizontal split (first part):\n{horizontal_split[0]}")

# ---------- 1.5 UNIVERSAL FUNCTIONS (UFUNCS) ----------

print("\n1.5 Universal Functions (ufuncs)")

x = np.linspace(0, 2*np.pi, 5)
print(f"x = {x}")
print(f"sin(x) = {np.sin(x)}")
print(f"cos(x) = {np.cos(x)}")
print(f"exp(x) = {np.exp(x)}")
print(f"log(x) = {np.log(np.abs(x) + 1)}")  # Adding 1 to avoid log(0)
print(f"sqrt(x) = {np.sqrt(x)}")

# ---------- 1.6 AGGREGATION FUNCTIONS ----------

print("\n1.6 Aggregation Functions")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array:\n{arr}")
print(f"Sum of all elements: {np.sum(arr)}")
print(f"Sum along rows (axis=0): {np.sum(arr, axis=0)}")
print(f"Sum along columns (axis=1): {np.sum(arr, axis=1)}")
print(f"Mean of all elements: {np.mean(arr)}")
print(f"Max of all elements: {np.max(arr)}")
print(f"Min of all elements: {np.min(arr)}")
print(f"Standard deviation: {np.std(arr)}")

# ---------- 1.7 LINEAR ALGEBRA ----------

print("\n1.7 Linear Algebra")

A = np.array([[1, 2], [3, 4]])
print(f"Matrix A:\n{A}")
print(f"Determinant: {np.linalg.det(A)}")
print(f"Inverse:\n{np.linalg.inv(A)}")
print(f"Eigenvalues: {np.linalg.eigvals(A)}")

eigvals, eigvecs = np.linalg.eig(A)
print(f"Eigenvalues: {eigvals}")
print(f"Eigenvectors:\n{eigvecs}")

# Solve linear system Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(f"\nSolution to Ax = b: {x}")
print(f"Verification: A @ x = {A @ x}")

# ========================================================================
# PART 2: PANDAS - DATA ANALYSIS
# ========================================================================

import pandas as pd

print("\n" + "=" * 50)
print("PANDAS TUTORIAL")
print("=" * 50)

# ---------- 2.1 SERIES AND DATAFRAMES ----------

print("\n2.1 Series and DataFrames")

# Series: 1D labeled array
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
print("Series:")
print(s)
print(f"Value at 'c': {s['c']}")

# DataFrame: 2D labeled data structure
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Paris', 'London', 'Tokyo'],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)

# DataFrame properties
print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# ---------- 2.2 DATA LOADING AND SAVING ----------

print("\n2.2 Data Loading and Saving")

# Create a sample CSV file
df.to_csv('sample_data.csv', index=False)
print("Saved DataFrame to 'sample_data.csv'")

# Read data
df_loaded = pd.read_csv('sample_data.csv')
print("\nLoaded DataFrame from CSV:")
print(df_loaded)

# Other data formats
# Excel: df.to_excel('file.xlsx', sheet_name='Sheet1')
# JSON: df.to_json('file.json')
# SQL: pd.read_sql('SELECT * FROM table', connection)

# ---------- 2.3 ACCESSING DATA ----------

print("\n2.3 Accessing Data")

# Select columns
print("Select 'Name' column:")
print(df['Name'])

# Select multiple columns
print("\nSelect 'Name' and 'Age' columns:")
print(df[['Name', 'Age']])

# Selecting by position (iloc)
print("\nFirst row:")
print(df.iloc[0])
print("\nFirst two rows, first two columns:")
print(df.iloc[:2, :2])

# Selecting by label (loc)
print("\nAccessing rows by labels:")
print(df.loc[2:3, ['Name', 'City']])

# Boolean indexing
print("\nRows where Age > 30:")
print(df[df['Age'] > 30])

# ---------- 2.4 DATA MANIPULATION ----------

print("\n2.4 Data Manipulation")

# Adding a column
df['Experience'] = [3, 5, 8, 12]
print("Added 'Experience' column:")
print(df)

# Applying functions to columns
df['Experience_Years'] = df['Experience'].apply(lambda x: f"{x} years")
print("\nApplied function to create 'Experience_Years':")
print(df)

# Sorting
print("\nSorted by Age (descending):")
print(df.sort_values('Age', ascending=False))

# Grouping
print("\nGrouping by City and calculating mean Age:")
print(df.groupby('City')['Age'].mean())

# ---------- 2.5 DATA CLEANING ----------

print("\n2.5 Data Cleaning")

# Create a DataFrame with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
print("DataFrame with missing values:")
print(df_missing)

# Check for missing values
print("\nMissing values count:")
print(df_missing.isnull().sum())

# Filling missing values
print("\nFilling missing values with 0:")
print(df_missing.fillna(0))

# Filling with method
print("\nFilling missing values with forward fill:")
print(df_missing.fillna(method='ffill'))

# Dropping rows with missing values
print("\nDropping rows with any missing values:")
print(df_missing.dropna())

# Dropping rows with all missing values
print("\nDropping rows with all missing values:")
print(df_missing.dropna(how='all'))

# ---------- 2.6 MERGING, JOINING, AND CONCATENATING ----------

print("\n2.6 Merging, Joining, and Concatenating")

# Create two DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Role': ['Engineer', 'Manager', 'Developer', 'Analyst']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 3, 5],
    'Department': ['HR', 'IT', 'Finance', 'Marketing'],
    'Salary': [50000, 60000, 70000, 80000]
})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Merge (like SQL join)
print("\nMerging on 'ID' (inner join):")
print(pd.merge(df1, df2, on='ID'))

print("\nMerging on 'ID' (left join):")
print(pd.merge(df1, df2, on='ID', how='left'))

# Concatenating
print("\nConcatenating DataFrames (vertical):")
print(pd.concat([df1, df1]))

# ---------- 2.7 PIVOTING AND RESHAPING ----------

print("\n2.7 Pivoting and Reshaping")

# Create a sample DataFrame
data = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 180]
}
sales_df = pd.DataFrame(data)
print("Sales data:")
print(sales_df)

# Pivot table
pivot = sales_df.pivot(index='Date', columns='Product', values='Sales')
print("\nPivot table:")
print(pivot)

# Melt (unpivot)
melted = pd.melt(pivot.reset_index(), id_vars=['Date'], 
                 var_name='Product', value_name='Sales')
print("\nMelted data:")
print(melted)

# ---------- 2.8 TIME SERIES ----------

print("\n2.8 Time Series")

# Create a time series
dates = pd.date_range('20230101', periods=6)
ts = pd.Series(np.random.randn(6), index=dates)
print("Time series:")
print(ts)

# Basic time series operations
print("\nResample to monthly:")
print(ts.resample('M').mean())

print("\nShift forward 2 periods:")
print(ts.shift(2))

print("\nRolling mean (window size 3):")
print(ts.rolling(window=3).mean())

# ========================================================================
# PART 3: MATPLOTLIB - DATA VISUALIZATION
# ========================================================================

import matplotlib.pyplot as plt

print("\n" + "=" * 50)
print("MATPLOTLIB TUTORIAL")
print("=" * 50)

# ---------- 3.1 BASIC PLOTTING ----------

print("\n3.1 Basic Plotting")

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.title('Basic Line Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.savefig('line_plot.png')
plt.close()
print("Created and saved 'line_plot.png'")

# Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Color')
plt.savefig('scatter_plot.png')
plt.close()
print("Created and saved 'scatter_plot.png'")

# Bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 55, 15]

plt.figure(figsize=(10, 6))
plt.bar(categories, values)
plt.title('Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.savefig('bar_plot.png')
plt.close()
print("Created and saved 'bar_plot.png'")

# Histogram
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('histogram.png')
plt.close()
print("Created and saved 'histogram.png'")

# ---------- 3.2 SUBPLOTS ----------

print("\n3.2 Subplots")

# Create a figure with multiple subplots
plt.figure(figsize=(12, 10))

# First subplot: Line plot
plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st position
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title('sin(x)')
plt.grid(True)

# Second subplot: Cos plot
plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd position
plt.plot(x, np.cos(x), 'g-')
plt.title('cos(x)')
plt.grid(True)

# Third subplot: Scatter
plt.subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd position
plt.scatter(np.random.rand(50), np.random.rand(50), c='red', alpha=0.5)
plt.title('Scatter Plot')

# Fourth subplot: Bar plot
plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th position
plt.bar(['A', 'B', 'C', 'D'], [10, 7, 5, 4])
plt.title('Bar Plot')

plt.tight_layout()  # Adjust spacing between subplots
plt.savefig('subplots.png')
plt.close()
print("Created and saved 'subplots.png'")

# Using object-oriented interface
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True)

axes[0, 1].plot(x, np.cos(x), 'g-')
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True)

axes[1, 0].scatter(np.random.rand(50), np.random.rand(50), c='red', alpha=0.5)
axes[1, 0].set_title('Scatter Plot')

axes[1, 1].bar(['A', 'B', 'C', 'D'], [10, 7, 5, 4])
axes[1, 1].set_title('Bar Plot')

plt.tight_layout()
plt.savefig('subplots_oop.png')
plt.close()
print("Created and saved 'subplots_oop.png'")

# ---------- 3.3 CUSTOMIZATION ----------

print("\n3.3 Customization")

# Create a styled line plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(12, 8))

# Plot with styling
plt.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
plt.plot(x, y2, 'r--', linewidth=2, label='cos(x)')

# Add titles and labels
plt.title('Sine and Cosine Functions', fontsize=18)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add a legend
plt.legend(fontsize=14)

# Customize the axes
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Add annotations
plt.annotate('Maximum', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Set axis limits
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)

plt.savefig('styled_plot.png')
plt.close()
print("Created and saved 'styled_plot.png'")

# ---------- 3.4 ADVANCED VISUALIZATIONS ----------

print("\n3.4 Advanced Visualizations")

# Contour plot
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

plt.figure(figsize=(12, 10))
contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour)
plt.title('Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('contour_plot.png')
plt.close()
print("Created and saved 'contour_plot.png'")

# 3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Surface Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(surface)
plt.savefig('3d_plot.png')
plt.close()
print("Created and saved '3d_plot.png'")

# ========================================================================
# PART 4: COMBINING NUMPY, PANDAS, AND MATPLOTLIB
# ========================================================================

print("\n" + "=" * 50)
print("INTEGRATING NUMPY, PANDAS, AND MATPLOTLIB")
print("=" * 50)

# ---------- 4.1 DATA PROCESSING AND VISUALIZATION WORKFLOW ----------

print("\n4.1 Data Processing and Visualization Workflow")

# Generate synthetic data
np.random.seed(42)
dates = pd.date_range('20220101', periods=100)
data = pd.DataFrame({
    'Date': dates,
    'Temperature': np.random.normal(20, 5, 100) + 3 * np.sin(np.arange(100) * 2 * np.pi / 50),
    'Humidity': np.random.normal(60, 10, 100),
    'WindSpeed': np.random.exponential(5, 100),
    'Rainfall': np.random.exponential(1, 100)
})

# Add a 'Season' column
data['Month'] = data['Date'].dt.month
data['Season'] = pd.cut(
    data['Month'],
    bins=[0, 3, 6, 9, 12],
    labels=['Winter', 'Spring', 'Summer', 'Fall'],
    include_lowest=True
)

print("Generated time series data:")
print(data.head())

# Data analysis
print("\nSummary statistics by season:")
seasonal_stats = data.groupby('Season').agg({
    'Temperature': ['mean', 'min', 'max', 'std'],
    'Humidity': ['mean', 'min', 'max', 'std'],
    'WindSpeed': ['mean', 'min', 'max', 'std'],
    'Rainfall': ['mean', 'sum']
})
print(seasonal_stats)

# Find correlations
correlations = data[['Temperature', 'Humidity', 'WindSpeed', 'Rainfall']].corr()
print("\nCorrelation matrix:")
print(correlations)

# Data visualization
plt.figure(figsize=(12, 10))

# Line plot of temperature over time
plt.subplot(2, 2, 1)
plt.plot(data['Date'], data['Temperature'], 'r-')
plt.title('Temperature Over Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)

# Box plot of temperature by season
plt.subplot(2, 2, 2)
data.boxplot(column='Temperature', by='Season', ax=plt.gca())
plt.title('Temperature by Season')
plt.ylabel('Temperature (°C)')
plt.suptitle('')  # Remove pandas-generated suptitle

# Scatter plot of temperature vs. humidity, colored by season
plt.subplot(2, 2, 3)
seasons = data['Season'].unique()
colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Fall': 'orange'}
for season in seasons:
    subset = data[data['Season'] == season]
    plt.scatter(subset['Temperature'], subset['Humidity'], 
                c=colors[season], label=season, alpha=0.7)
plt.title('Temperature vs. Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.legend()

# Histogram of rainfall
plt.subplot(2, 2, 4)
plt.hist(data['Rainfall'], bins=15, color='blue', alpha=0.7)
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('integrated_analysis.png')
plt.close()
print("Created and saved 'integrated_analysis.png'")

# ========================================================================
# PART 5: REAL-WORLD APPLICATIONS
# ========================================================================

print("\n" + "=" * 50)
print("REAL-WORLD APPLICATIONS")
print("=" * 50)

# ---------- 5.1 MACHINE LEARNING DATA PREPARATION ----------

print("\n5.1 Machine Learning Data Preparation")

# Create a synthetic dataset
np.random.seed(42)
n_samples = 200

# Generate features
X = np.random.randn(n_samples, 3)
feature_names = ['feature1', 'feature2', 'feature3']

# Generate target (y = 2*f1 - 3*f2 + 0.5*f3 + noise)
y = 2 * X[:, 0] - 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.5

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Machine learning dataset:")
print(df.head())

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['target'], test_size=0.3, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandardized features (first 3 rows):")
print(pd.DataFrame(X_train_scaled[:3], columns=feature_names))

# Train a simple model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\nModel coefficients:")
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize predictions vs. actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.title('Actual vs. Predicted')
plt.grid(True, alpha=0.3)
plt.savefig('ml_prediction.png')
plt.close()
print("Created and saved 'ml_prediction.png'")

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.bar(feature_names, np.abs(model.coef_))
plt.xlabel('Feature')
plt.ylabel('Absolute Coefficient')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()
print("Created and saved 'feature_importance.png'")

# ---------- 5.2 TIME SERIES FORECASTING ----------

print("\n5.2 Time Series Forecasting")

# Generate a synthetic time series
np.random.seed(42)
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
n = len(date_rng)

# Base trend
trend = np.linspace(0, 10, n)

# Seasonal component (yearly seasonality)
season_days = 365.25
seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / season_days)

# Weekly pattern
weekly = 2 * np.sin(2 * np.pi * np.arange(n) / 7)

# Random noise
noise = np.random.normal(0, 1, n)

# Combine components
signal = trend + seasonal + weekly + noise

# Create time series DataFrame
ts_data = pd.DataFrame(data={'date': date_rng, 'value': signal})
ts_data.set_index('date', inplace=True)

print("Time series data:")
print(ts_data.head())

# Add date features
ts_data['year'] = ts_data.index.year
ts_data['month'] = ts_data.index.month
ts_data['day'] = ts_data.index.day
ts_data['dayofweek'] = ts_data.index.dayofweek

# Moving averages
ts_data['MA7'] = ts_data['value'].rolling(window=7).mean()
ts_data['MA30'] = ts_data['value'].rolling(window=30).mean()
ts_data['MA90'] = ts_data['value'].rolling(window=90).mean()

# Resample to monthly data
monthly_data = ts_data['value'].resample('M').mean()

# Visualize original time series and moving averages
plt.figure(figsize=(12, 8))
plt.plot(ts_data.index, ts_data['value'], 'k-', alpha=0.4, label='Original')
plt.plot(ts_data.index, ts_data['MA7'], 'b-', label='7-day MA')
plt.plot(ts_data.index, ts_data['MA30'], 'g-', label='30-day MA')
plt.plot(ts_data.index, ts_data['MA90'], 'r-', label='90-day MA')
plt.title('Time Series with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('time_series_ma.png')
plt.close()
print("Created and saved 'time_series_ma.png'")

# Decompose the time series
from statsmodels.tsa.seasonal import seasonal_decompose

# Use a subset of data for better visualization
decomposition = seasonal_decompose(ts_data['value']['2021-01-01':'2021-12-31'], 
                                  model='additive', period=30)

# Plot decomposition
fig = plt.figure(figsize=(12, 10))
plt.subplot(411)
plt.plot(decomposition.observed)
plt.title('Observed')
plt.subplot(412)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.subplot(413)
plt.plot(decomposition.seasonal)
plt.title('Seasonal')
plt.subplot(414)
plt.plot(decomposition.resid)
plt.title('Residual')
plt.tight_layout()
plt.savefig('time_series_decomposition.png')
plt.close()
print("Created and saved 'time_series_decomposition.png'")

# ---------- 5.3 DATA DASHBOARDING ----------

print("\n5.3 Data Dashboarding")

# Generate sales data
np.random.seed(42)
n_months = 24
months = pd.date_range('2022-01-01', periods=n_months, freq='M')

products = ['Product A', 'Product B', 'Product C', 'Product D']
regions = ['North', 'South', 'East', 'West']

sales_data = []
for month in months:
    for product in products:
        for region in regions:
            # Base sales with seasonality and trend
            base_sales = 1000 + 100 * np.sin(month.month * np.pi / 6)
            
            # Product effect
            product_factor = {'Product A': 1.0, 'Product B': 1.2, 
                             'Product C': 0.8, 'Product D': 1.5}[product]
            
            # Region effect
            region_factor = {'North': 0.9, 'South': 1.1, 
                            'East': 1.2, 'West': 1.0}[region]
            
            # Time trend (increasing over time)
            trend = month.year - 2022 + month.month / 12
            trend_factor = 1 + trend * 0.05
            
            # Random noise
            noise = np.random.normal(1, 0.1)
            
            sales = int(base_sales * product_factor * region_factor * trend_factor * noise)
            
            sales_data.append({
                'Date': month,
                'Product': product,
                'Region': region,
                'Sales': sales
            })

sales_df = pd.DataFrame(sales_data)
print("Sales data:")
print(sales_df.head())

# Aggregate data for different views
monthly_sales = sales_df.groupby('Date')['Sales'].sum().reset_index()
product_sales = sales_df.groupby('Product')['Sales'].sum().reset_index()
region_sales = sales_df.groupby('Region')['Sales'].sum().reset_index()
product_region_sales = sales_df.groupby(['Product', 'Region'])['Sales'].sum().reset_index()

# Create a dashboard with multiple plots
plt.figure(figsize=(15, 12))

# Monthly Sales Trend
plt.subplot(221)
plt.plot(monthly_sales['Date'], monthly_sales['Sales'], 'b-', marker='o')
plt.title('Monthly Sales Trend')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Sales by Product
plt.subplot(222)
bars = plt.bar(product_sales['Product'], product_sales['Sales'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Sales by Product')
plt.ylabel('Sales')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{height:,}', ha='center', va='bottom')

# Sales by Region
plt.subplot(223)
plt.pie(region_sales['Sales'], labels=region_sales['Region'], autopct='%1.1f%%', 
        colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Sales by Region')

# Heatmap of Product-Region Sales
plt.subplot(224)
pivot_data = pd.pivot_table(product_region_sales, values='Sales',
                           index='Product', columns='Region')
im = plt.imshow(pivot_data, cmap='YlGnBu')
plt.title('Product-Region Sales Heatmap')
plt.ylabel('Product')
plt.xticks(range(len(regions)), regions)
plt.yticks(range(len(products)), products)

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Sales')

# Add value annotations
for i in range(len(products)):
    for j in range(len(regions)):
        plt.text(j, i, f'{pivot_data.iloc[i, j]:,.0f}', 
                 ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('sales_dashboard.png')
plt.close()
print("Created and saved 'sales_dashboard.png'")

# ========================================================================
# PART 6: MINI-PROJECT: STOCK MARKET ANALYSIS
# ========================================================================

print("\n" + "=" * 50)
print("MINI-PROJECT: STOCK MARKET ANALYSIS")
print("=" * 50)

# Generate synthetic stock data
np.random.seed(42)

def generate_stock_data(ticker, start_date, end_date, start_price, volatility):
    """Generate synthetic stock price data."""
    days = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(days)
    
    # Log returns (random walk with drift)
    drift = 0.0001  # Small upward drift
    returns = np.random.normal(loc=drift, scale=volatility, size=n_days)
    
    # Calculate price series
    prices = start_price * (1 + returns).cumprod()
    
    # Add volume
    volume = np.random.lognormal(mean=np.log(1000000), sigma=0.5, size=n_days)
    
    # Create DataFrame
    stock_data = pd.DataFrame({
        'Date': days,
        'Open': prices * (1 + np.random.normal(0, 0.002, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.004, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_days))),
        'Close': prices,
        'Volume': volume.astype(int)
    })
    
    # Ensure High is always >= Open, Close, Low
    stock_data['High'] = stock_data[['High', 'Open', 'Close']].max(axis=1)
    
    # Ensure Low is always <= Open, Close, High
    stock_data['Low'] = stock_data[['Low', 'Open', 'Close']].min(axis=1)
    
    # Add ticker
    stock_data['Ticker'] = ticker
    
    return stock_data

# Generate data for multiple stocks
start_date = '2020-01-01'
end_date = '2022-12-31'

stocks = {
    'AAPL': {'price': 75.0, 'volatility': 0.015},
    'MSFT': {'price': 150.0, 'volatility': 0.012},
    'GOOGL': {'price': 1300.0, 'volatility': 0.018},
    'AMZN': {'price': 1800.0, 'volatility': 0.020}
}

stock_dfs = []
for ticker, params in stocks.items():
    stock_df = generate_stock_data(ticker, start_date, end_date, 
                                  params['price'], params['volatility'])
    stock_dfs.append(stock_df)

# Combine all stock data
all_stocks = pd.concat(stock_dfs)
all_stocks.set_index('Date', inplace=True)
print("Generated stock data:")
print(all_stocks.head())

# ---------- 6.1 DATA PROCESSING ----------

print("\n6.1 Data Processing")

# Calculate daily returns
all_stocks['Daily_Return'] = all_stocks.groupby('Ticker')['Close'].pct_change() * 100

# Calculate moving averages
all_stocks['MA20'] = all_stocks.groupby('Ticker')['Close'].transform(
    lambda x: x.rolling(window=20).mean())
all_stocks['MA50'] = all_stocks.groupby('Ticker')['Close'].transform(
    lambda x: x.rolling(window=50).mean())
all_stocks['MA200'] = all_stocks.groupby('Ticker')['Close'].transform(
    lambda x: x.rolling(window=200).mean())

# Calculate trading signals (simple example: MA crossover)
all_stocks['Signal'] = 0
mask = all_stocks['MA20'] > all_stocks['MA50']
all_stocks.loc[mask, 'Signal'] = 1
all_stocks.loc[~mask, 'Signal'] = -1

# Calculate Bollinger Bands
all_stocks['MA20_std'] = all_stocks.groupby('Ticker')['Close'].transform(
    lambda x: x.rolling(window=20).std())
all_stocks['Upper_Band'] = all_stocks['MA20'] + (all_stocks['MA20_std'] * 2)
all_stocks['Lower_Band'] = all_stocks['MA20'] - (all_stocks['MA20_std'] * 2)

# Calculate trading volume indicators
all_stocks['Vol_MA20'] = all_stocks.groupby('Ticker')['Volume'].transform(
    lambda x: x.rolling(window=20).mean())
all_stocks['Volume_Ratio'] = all_stocks['Volume'] / all_stocks['Vol_MA20']

print("Processed stock data:")
print(all_stocks[['Ticker', 'Close', 'Daily_Return', 'MA20', 'Signal']].head())

# ---------- 6.2 VISUALIZATION AND ANALYSIS ----------

print("\n6.2 Visualization and Analysis")

# Function to plot stock data
def plot_stock_data(ticker):
    # Filter data for the specific ticker
    stock_data = all_stocks[all_stocks['Ticker'] == ticker].copy()
    
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{ticker} Stock Analysis", fontsize=16)
    
    # Plot 1: Price with Moving Averages
    axes[0, 0].plot(stock_data.index, stock_data['Close'], 'k-', label='Close')
    axes[0, 0].plot(stock_data.index, stock_data['MA20'], 'b-', label='MA20')
    axes[0, 0].plot(stock_data.index, stock_data['MA50'], 'g-', label='MA50')
    axes[0, 0].plot(stock_data.index, stock_data['MA200'], 'r-', label='MA200')
    axes[0, 0].set_title(f"{ticker} Price and Moving Averages")
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Bollinger Bands
    axes[0, 1].plot(stock_data.index, stock_data['Close'], 'k-', label='Close')
    axes[0, 1].plot(stock_data.index, stock_data['MA20'], 'b-', label='MA20')
    axes[0, 1].plot(stock_data.index, stock_data['Upper_Band'], 'g--', label='Upper Band')
    axes[0, 1].plot(stock_data.index, stock_data['Lower_Band'], 'r--', label='Lower Band')
    axes[0, 1].fill_between(stock_data.index, stock_data['Upper_Band'], stock_data['Lower_Band'], 
                          alpha=0.1, color='gray')
    axes[0, 1].set_title(f"{ticker} Bollinger Bands")
    axes[0, 1].set_ylabel("Price ($)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Daily Returns Histogram
    axes[1, 0].hist(stock_data['Daily_Return'].dropna(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_title(f"{ticker} Daily Returns Distribution")
    axes[1, 0].set_xlabel("Daily Return (%)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Volume
    axes[1, 1].bar(stock_data.index, stock_data['Volume'], alpha=0.5)
    axes[1, 1].plot(stock_data.index, stock_data['Vol_MA20'], 'r-', label='Volume MA20')
    axes[1, 1].set_title(f"{ticker} Trading Volume")
    axes[1, 1].set_ylabel("Volume")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{ticker}_analysis.png")
    plt.close()
    print(f"Created and saved '{ticker}_analysis.png'")

# Plot each stock
for ticker in stocks.keys():
    plot_stock_data(ticker)

# ---------- 6.3 PORTFOLIO ANALYSIS ----------

print("\n6.3 Portfolio Analysis")

# Create a pivot table for closing prices
pivot_close = all_stocks.pivot_table(values='Close', index=all_stocks.index, columns='Ticker')
print("Pivot table of stock prices:")
print(pivot_close.head())

# Calculate daily returns for the pivot
pivot_returns = pivot_close.pct_change().dropna()
print("\nDaily returns:")
print(pivot_returns.head())

# Calculate cumulative returns
cumulative_returns = (1 + pivot_returns).cumprod()
print("\nCumulative returns:")
print(cumulative_returns.tail())

# Calculate portfolio statistics
mean_returns = pivot_returns.mean() * 252  # Annualized
cov_matrix = pivot_returns.cov() * 252  # Annualized
print("\nAnnualized Mean Returns:")
print(mean_returns)
print("\nAnnualized Covariance Matrix:")
print(cov_matrix)

# Create an equal-weight portfolio
weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_return = np.sum(mean_returns * weights)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe_ratio = portfolio_return / portfolio_volatility  # Assuming risk-free rate = 0

print("\nEqual-weight Portfolio Metrics:")
print(f"Expected Return: {portfolio_return:.4f}")
print(f"Volatility: {portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# Calculate and plot the correlation matrix
correlation = pivot_returns.corr()
print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 8))
im = plt.imshow(correlation, cmap='coolwarm')
plt.colorbar(im)
plt.title('Stock Returns Correlation Matrix')
plt.xticks(range(len(correlation.columns)), correlation.columns)
plt.yticks(range(len(correlation.columns)), correlation.columns)

# Add correlation values
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                 ha='center', va='center', color='black')

plt.savefig('correlation_matrix.png')
plt.close()
print("Created and saved 'correlation_matrix.png'")

# Plot cumulative returns
plt.figure(figsize=(12, 8))
for ticker in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)

# Add an equal-weight portfolio
portfolio_cumulative = cumulative_returns.mean(axis=1)
plt.plot(cumulative_returns.index, portfolio_cumulative, 'k--', linewidth=3, label='Portfolio')

plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cumulative_returns.png')
plt.close()
print("Created and saved 'cumulative_returns.png'")

# ========================================================================
# PART 7: BEST PRACTICES AND PERFORMANCE TIPS
# ========================================================================

print("\n" + "=" * 50)
print("BEST PRACTICES AND PERFORMANCE TIPS")
print("=" * 50)

# ---------- 7.1 NUMPY BEST PRACTICES ----------

print("\n7.1 NumPy Best Practices")

print("""
NumPy Best Practices:
1. Use vectorized operations instead of loops.
   - BAD:  for i in range(len(array)): array[i] *= 2
   - GOOD: array *= 2

2. Use broadcasting instead of repetitive operations.
   - BAD:  for i in range(array.shape[0]): array[i] += scalar
   - GOOD: array += scalar

3. Use appropriate data types to save memory.
   - If you have integers that don't exceed 127, use np.int8 instead of np.int64
   - Example: array = np.array([1, 2, 3], dtype=np.int8)

4. Use fancy indexing and boolean masks for filtering.
   - BAD:  [x for x in array if x > 0]
   - GOOD: array[array > 0]

5. Use np.where for conditional assignments.
   - BAD:  for i in range(len(array)): array[i] = a if array[i] > 0 else b
   - GOOD: np.where(array > 0, a, b)

6. Preallocate arrays instead of growing them.
   - BAD:  result = []; for item in data: result.append(function(item))
   - GOOD: result = np.zeros(len(data)); for i, item in enumerate(data): result[i] = function(item)

7. Use specialized NumPy functions (e.g., np.sum) instead of Python's built-ins (e.g., sum).
   - BAD:  sum(array)
   - GOOD: np.sum(array)

8. Set axis parameter in aggregation functions to avoid unnecessary copies.
   - BAD:  np.mean(array)
   - GOOD: np.mean(array, axis=0)  # If you only need column means

9. Use views instead of copies when possible.
   - BAD:  new_array = array.copy()
   - GOOD: new_array = array.view()  # If you don't need to modify the original array

10. Use np.einsum for complex array operations.
    - For optimization of multiple array operations in a single step
""")

# ---------- 7.2 PANDAS BEST PRACTICES ----------

print("\n7.2 Pandas Best Practices")

print("""
Pandas Best Practices:
1. Use vectorized operations instead of iterating through rows.
   - BAD:  for i, row in df.iterrows(): df.at[i, 'new_col'] = calculation(row['col'])
   - GOOD: df['new_col'] = calculation(df['col'])

2. Use query() for filtering based on complex conditions.
   - BAD:  df[(df['A'] > 0) & (df['B'] < 10) & (df['C'] == 'x')]
   - GOOD: df.query('A > 0 and B < 10 and C == "x"')

3. Use method chaining for cleaner code.
   - BAD:  df2 = df.dropna(); df3 = df2.reset_index(); result = df3.groupby('col').sum()
   - GOOD: result = df.dropna().reset_index().groupby('col').sum()

4. Set categorical dtypes for string columns with low cardinality.
   - df['category'] = df['category'].astype('category')

5. Use appropriate dtypes to reduce memory usage.
   - df['small_int'] = df['small_int'].astype('int8')
   - df['date_column'] = pd.to_datetime(df['date_column'])

6. Use loc/iloc for selection instead of chained indexing.
   - BAD:  df['A'][df['B'] > 0] = 1
   - GOOD: df.loc[df['B'] > 0, 'A'] = 1

7. Use apply() with axis=1 only when necessary; prefer vectorized operations.
   - BAD:  df.apply(lambda row: complex_operation(row), axis=1)
   - GOOD: vectorized_complex_operation(df)

8. Set inplace=True to modify dataframes without creating copies.
   - BAD:  df = df.fillna(0)
   - GOOD: df.fillna(0, inplace=True)

9. Use merge-asof for merging time-series data with nearest matches.
   - pd.merge_asof(left, right, on='timestamp', direction='nearest')

10. Use get_dummies() for one-hot encoding categorical variables.
    - pd.get_dummies(df['category'], prefix='cat')

11. Use groupby with agg() for multiple aggregations in one pass.
    - df.groupby('group').agg({'A': 'mean', 'B': 'sum', 'C': ['min', 'max']})

12. Use swaplevel() and sort_index() to optimize MultiIndex operations.
    - df.swaplevel(0, 1).sort_index()
""")

# ---------- 7.3 MATPLOTLIB BEST PRACTICES ----------

print("\n7.3 Matplotlib Best Practices")

print("""
Matplotlib Best Practices:
1. Use the object-oriented interface (fig, ax) instead of plt.* functions.
   - BAD:  plt.plot(); plt.title(); plt.xlabel()
   - GOOD: fig, ax = plt.subplots(); ax.plot(); ax.set_title(); ax.set_xlabel()

2. Set style parameters at the beginning of your script.
   - plt.style.use('ggplot')  # or 'seaborn', 'fivethirtyeight', etc.

3. Use tight_layout() to avoid text overlap and optimize spacing.
   - plt.tight_layout()

4. Use constrained_layout for better spacing in complex figures.
   - fig, ax = plt.subplots(constrained_layout=True)

5. Save high-resolution figures with appropriate DPI.
   - plt.savefig('figure.png', dpi=300)

6. Use meaningful colormaps for different types of data.
   - Sequential: 'viridis', 'plasma', 'cividis'
   - Diverging: 'coolwarm', 'RdBu'
   - Qualitative: 'tab10', 'tab20'

7. Customize figures for different use cases (presentations vs publications).
   - For presentations: larger fonts, bolder lines, fewer details
   - For publications: precise annotations, more details, optimized for print

8. Set reasonable figure sizes to avoid distortion.
   - fig, ax = plt.subplots(figsize=(10, 6))  # 5:3 aspect ratio

9. Use GridSpec for complex subplot arrangements.
   - from matplotlib.gridspec import GridSpec

10. Close figures when done to free memory.
    - plt.close() or plt.close('all')

11. Use plt.rcParams to set global defaults.
    - plt.rcParams['font.size'] = 12
    - plt.rcParams['figure.figsize'] = (10, 6)

12. Add appropriate labels, legends, and titles for clarity.
    - Every plot should clearly communicate what it represents
""")

# ---------- 7.4 COMMON MISTAKES TO AVOID ----------

print("\n7.4 Common Mistakes to Avoid")

print("""
Common Mistakes to Avoid:
1. Modifying data during iteration.
   - This can lead to unpredictable results and errors.

2. Ignoring missing data in analysis.
   - Always check df.isnull().sum() before analysis.

3. Not handling outliers appropriately.
   - Outliers can significantly distort statistical analyses.

4. Using inefficient loops instead of vectorized operations.
   - Vectorized operations in NumPy and Pandas are much faster.

5. Unnecessary copying of large datasets.
   - Use views or references when possible.

6. Not checking data types before operations.
   - Mixed data types can lead to unexpected results.

7. Ignoring index alignment in Pandas operations.
   - Ensure indices match when combining DataFrames.

8. Using float equality comparisons (e.g., a == 0.1).
   - Due to floating-point precision issues, use np.isclose() instead.

9. Forgetting to handle categorical and string data properly.
   - Convert to appropriate types before analysis.

10. Not setting seeds for reproducibility.
    - np.random.seed(42) before any random operations.

11. Using inappropriate visualization techniques for the data type.
    - Match the visualization to the data (e.g., bar charts for categorical data).

12. Not closing resources (files, plots, etc.).
    - Always close resources to avoid memory leaks.

13. Chained indexing in Pandas.
    - df['col']['condition'] can lead to unexpected behavior; use df.loc instead.

14. Not considering performance implications for large datasets.
    - Profile code and use appropriate optimizations.
""")

# ---------- 7.5 OPTIMIZATION STRATEGIES ----------

print("\n7.5 Optimization Strategies")

print("""
Optimization Strategies:
1. Use appropriate data types to reduce memory usage.
   - int8/int16 for small integers, float32 instead of float64 when precision allows

2. Minimize copying of large arrays and DataFrames.
   - Use inplace=True when available
   - Use views instead of copies when possible

3. Profile your code to identify bottlenecks.
   - Use %timeit in Jupyter Notebooks
   - Use the cProfile module for function profiling

4. Vectorize operations instead of using loops.
   - NumPy and Pandas operations are highly optimized

5. Use boolean indexing for filtering.
   - df[df['column'] > value] instead of iterating

6. Use query() for complex filtering conditions in pandas.
   - df.query('A > 0 & B < 10') can be faster than df[(df['A'] > 0) & (df['B'] < 10)]

7. Utilize parallelization for independent operations.
   - Dask for large datasets beyond memory
   - concurrent.futures for parallel processing

8. Use specialized libraries when appropriate.
   - numba for JIT compilation
   - scipy for scientific computing functions
   - scikit-learn for machine learning

9. Preallocate arrays of the correct size.
   - np.zeros(shape) instead of growing arrays dynamically

10. Use appropriate aggregation methods.
    - Set axis parameter in numpy aggregations
    - Use optimized pandas aggregation methods

11. Batch process very large datasets.
    - Process data in chunks to avoid memory issues

12. Use appropriate file formats for large datasets.
    - HDF5, Parquet, or Feather instead of CSV
""")

# ========================================================================
# PART 8: CONCLUSION
# ========================================================================

print("\n" + "=" * 50)
print("CONCLUSION")
print("=" * 50)

print("""
This tutorial has covered the essential features of NumPy, Pandas, and Matplotlib,
providing a strong foundation for data manipulation, analysis, and visualization in Python.

Key takeaways:
- **NumPy**: Efficient numerical computing with arrays, mathematical operations, and performance optimization.
- **Pandas**: Powerful data manipulation using Series and DataFrames for real-world datasets.
- **Matplotlib**: Customizable data visualization for insights and trend analysis.

By mastering these libraries, you can:
✔ Handle large datasets efficiently.
✔ Clean, transform, and analyze data for AI-driven applications.
✔ Visualize trends, anomalies, and patterns in an intuitive way.
✔ Automate data processing for reports, dashboards, and decision-making.

### Next Steps:
1. Experiment with **real-world datasets** to solidify your understanding.
2. Combine these tools with **machine learning frameworks** (Scikit-Learn, TensorFlow, PyTorch).
3. Implement automation scripts for **business intelligence, web scraping, and AI dashboards**.
4. Expand into **interactive visualizations** using libraries like **Seaborn or Plotly**.
5. Keep practicing and build **JafarDigital’s AI-powered automation systems**!

Thank you for following this tutorial. 🚀 Happy coding!
""")

