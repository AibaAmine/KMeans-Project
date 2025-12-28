import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# PART 1 : Data Exploration

# Set style for better visualizations
sns.set(style="whitegrid")

# 1. Load the Dataset


column_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
     "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Load the data
df = pd.read_csv("heart.csv", header=0, names=column_names)  # If no header in CSV
# Or if the CSV has headers: df = pd.read_csv('heart.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Display shape
print("\nShape of the dataset:", df.shape)

# Display column names and data types
print("\nColumn names and data types:")
print(df.dtypes)

# 2. Target Removal (Unsupervised learning - K-Means does not use labels)
# The 'target' column indicates presence/absence of heart disease
df_unsupervised = df.drop("target", axis=1)

print("\nAfter removing target column - new shape:", df_unsupervised.shape)

# 3. Basic Statistics for numerical features
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

print("\nBasic statistics for numerical features:")
stats = df_unsupervised[numerical_features].agg(["mean", "std", "min", "max"])
print(stats)

# 4. Data Cleaning
# Check for missing values (in this dataset, missing are marked as '?' in 'ca' and 'thal')
df_unsupervised = df_unsupervised.replace("?", np.nan)

print("\nMissing values per column:")
print(df_unsupervised.isnull().sum())

# Remove rows with missing values (only a few in this dataset)
df_clean = df_unsupervised.dropna()

print("\nShape after removing missing values:", df_clean.shape)

# Check for and remove duplicates
duplicates = df_clean.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print("Shape after removing duplicates:", df_clean.shape)

# Remove irrelevant columns (no patient ID in this dataset; all features are relevant)
# Note: 'ca' and 'thal' are kept as they are clinically relevant for clustering

# 5. Visualization
# Histograms for feature distributions
df_clean[numerical_features].hist(bins=20, figsize=(14, 10), layout=(2, 3))
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Histograms/KDE for categorical features
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    sns.countplot(x=col, data=df_clean, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Box plots to identify outliers and spread (focus on numerical)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[numerical_features])
plt.title("Box Plots of Numerical Features")
plt.show()

# Scatter plots for relationships between selected numerical features
# Example: age vs thalach, colored by sex for insight
plt.figure(figsize=(10, 6))
sns.scatterplot(x="age", y="thalach", hue="sex", data=df_clean, palette="deep")
plt.title("Scatter Plot: Age vs Maximum Heart Rate (colored by sex)")
plt.show()

# Additional scatter: trestbps vs chol
plt.figure(figsize=(10, 6))
sns.scatterplot(x="trestbps", y="chol", hue="cp", data=df_clean, palette="viridis")
plt.title(
    "Scatter Plot: Resting Blood Pressure vs Cholesterol (colored by chest pain type)"
)
plt.show()

# 6. Preprocessing for Clustering
# Identify categorical and numerical features
categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]  # ca and thal are ordinal/categorical
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Apply One Hot Encoding to categorical variables
# Use ColumnTransformer for clean pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),  # Keep numerical as is for now
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
    ]
)

# Fit and transform the data
df_encoded = preprocessor.fit_transform(df_clean)

# Get feature names after encoding
ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
    categorical_features
)
all_feature_names = numerical_features + list(ohe_feature_names)

df_preprocessed = pd.DataFrame(df_encoded, columns=all_feature_names)

print("\nShape after One Hot Encoding:", df_preprocessed.shape)
print("Encoded feature names sample:", all_feature_names[:10])

# Select features for K-Means (all encoded features are used)
features_for_clustering = df_preprocessed.columns  # All columns


# Normalize/Standardize numerical features (all columns now include original numerical + encoded)
# Standardization (zero mean, unit variance) is recommended for K-Means
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_preprocessed)

df_final = pd.DataFrame(df_scaled, columns=features_for_clustering)

print("\nFinal preprocessed data for K-Means clustering - shape:", df_final.shape)
print(df_final.head())

# The df_final DataFrame is now ready for K-Means clustering (e.g., from sklearn.cluster import KMeans)
# Example: kmeans = KMeans(n_clusters=2, random_state=42); clusters = kmeans.fit_predict(df_final)

# PART 2 : Model Training
## Part 2.1
## Splitting the data: 80% Training, 20% Test
X_train, X_test = train_test_split(df_final, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# Part 2.2
# We store the  error for each k here
error_values = []
k_range = range(2, 11)

print("Running Elbow Method...")

for k in k_range:
    # Initialize KMeans with k clusters
    # random_state=42 ensures consistent results
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Fit on the TRAINING data (X_train)
    kmeans.fit(X_train)

    # Append the inertia (error) to our list
    error_values.append(kmeans.inertia_)

# --- Plotting the Elbow Curve ---
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_values, marker="o", linestyle="--")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)
plt.savefig("elbow_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# Part 2.3: Train Final K-Means Model with Optimal K

optimal_k = 3

# Train the final model
final_kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10,
    max_iter=300,  # Maximum iterations for convergence
)

# Fit on training data
final_kmeans.fit(X_train)


# Part 2.4: Obtain cluster labels for both training and test data

# For training data
train_labels = final_kmeans.predict(X_train)

# For test data
test_labels = final_kmeans.predict(X_test) 

# Display results
print(f"Training data cluster labels: {train_labels}")
print(f"Test data cluster labels: {test_labels}")


# Part 2.5: Display the cluster centers

print(f"\n{'='*60}")
print(f"Part 2.5: Cluster Centers")
print(f"{'='*60}\n")

# Get the cluster centers from the trained model
cluster_centers = final_kmeans.cluster_centers_

print(f"Number of clusters: {optimal_k}")
print(f"Shape of cluster centers: {cluster_centers.shape}")

# Display each cluster center
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i} Center:")
    print(center)
    print()


