import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# ============================================================
# PART 1: DATA EXPLORATION
# ============================================================

print("=" * 60)
print("PART 1: DATA EXPLORATION")
print("=" * 60)

# 1.1: Load the Dataset
# Define column names for the Heart Disease dataset
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

# Load the dataset from CSV file
df = pd.read_csv("heart.csv", header=0, names=column_names)

print("\n1.1: Dataset Overview")
print("First 5 rows:")
print(df.head())
print(f"\nShape: {df.shape}")
print("\nData types:")
print(df.dtypes)

# 1.2: Remove target column
# K-Means is unsupervised, so we remove the target label
df_unsupervised = df.drop("target", axis=1)
print(f"\n1.2: After removing target column: {df_unsupervised.shape}")

# 1.3: Basic Statistics
# Separate features into numerical and categorical for appropriate processing
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Calculate basic statistics for numerical features
print("\n1.3: Basic statistics:")
print(df_unsupervised[numerical_features].agg(["mean", "std", "min", "max"]))

# 1.4: Handle missing values
# Replace '?' markers with NaN for proper detection
df_unsupervised = df_unsupervised.replace("?", np.nan)
print(f"\n1.4: Missing values per column:")
print(df_unsupervised.isnull().sum())

# Remove rows with missing values
df_clean = df_unsupervised.dropna()
print(f"Shape after removing missing values: {df_clean.shape}")

# 1.5: Remove duplicates
# Check for and remove duplicate rows to avoid bias in clustering
duplicates = df_clean.duplicated().sum()
print(f"\n1.5: Number of duplicate rows: {duplicates}")
if duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"Shape after removing duplicates: {df_clean.shape}")

# 1.6: Visualizations
# Visualize data distributions to understand patterns
print("\n1.6: Generating visualizations...")

# Histograms for numerical features to see distributions
df_clean[numerical_features].hist(bins=20, figsize=(14, 10), layout=(2, 3))
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Count plots for categorical features
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    sns.countplot(x=col, data=df_clean, ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.show()

# Box plots to identify outliers and data spread
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[numerical_features])
plt.title("Box Plots of Numerical Features")
plt.show()

# Scatter plot: Age vs Maximum Heart Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x="age", y="thalach", hue="sex", data=df_clean, palette="deep")
plt.title("Age vs Heart Rate by Sex")
plt.show()

# Scatter plot: Blood Pressure vs Cholesterol
plt.figure(figsize=(10, 6))
sns.scatterplot(x="trestbps", y="chol", hue="cp", data=df_clean, palette="viridis")
plt.title("Blood Pressure vs Cholesterol by Chest Pain Type")
plt.show()

# 1.7: One Hot Encoding
# Convert categorical variables to numerical using One Hot Encoding
# K-Means requires all features to be numerical
print("\n1.7: Applying One Hot Encoding...")
preprocessor = ColumnTransformer(
    [
        ("num", "passthrough", numerical_features),  # Keep numerical features as is
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_features,
        ),  # Encode categorical
    ]
)

# Apply the transformations
df_encoded = preprocessor.fit_transform(df_clean)
ohe_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
    categorical_features
)
all_features = numerical_features + list(ohe_features)
df_preprocessed = pd.DataFrame(df_encoded, columns=all_features)

print(f"Shape after encoding: {df_preprocessed.shape}")

# 1.8: Standardization
# Scale features to have mean=0 and std=1
# This is important for K-Means as it uses Euclidean distance
print("\n1.8: Standardizing features...")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_preprocessed)
df_final = pd.DataFrame(df_scaled, columns=all_features)

print(f"Final preprocessed shape: {df_final.shape}")
print(df_final.head())


# ============================================================
# PART 2: MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("PART 2: MODEL TRAINING")
print("=" * 60)

# 2.1: Split data
# Split into 80% training and 20% test sets
# This allows us to evaluate if clusters generalize to unseen data
X_train, X_test = train_test_split(df_final, test_size=0.2, random_state=42)
print(f"\n2.1: Data split")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 2.2: Elbow Method
# Find optimal number of clusters by plotting inertia for different k values
# The "elbow" point indicates where adding more clusters gives diminishing returns
print("\n2.2: Running Elbow Method...")
error_values = []
k_range = range(2, 11)  # Test k from 2 to 10

for k in k_range:
    # Train K-Means with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    # Store inertia (sum of squared distances to nearest cluster center)
    error_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_values, marker="o", linestyle="--")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)
plt.savefig("elbow_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# 2.3: Train final model
# Based on the elbow curve, we select k=3 as optimal
optimal_k = 3
print(f"\n2.3: Training final model with k={optimal_k}")

# Initialize and train K-Means with optimal k
# n_init=10: Run algorithm 10 times with different initializations
# max_iter=300: Maximum iterations for convergence
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
final_kmeans.fit(X_train)
print("Model trained successfully")

# 2.4: Get cluster labels
# Assign each data point to its nearest cluster
train_labels = final_kmeans.predict(X_train)  # Labels for training data
test_labels = final_kmeans.predict(X_test)  # Labels for test data

print(f"\n2.4: Cluster labels obtained")
print(f"Training labels: {train_labels}")
print(f"Test labels: {test_labels}")

# 2.5: Display cluster centers
# Extract the centroid (mean position) of each cluster
# These represent the "average" patient profile in each cluster
cluster_centers = final_kmeans.cluster_centers_

print(f"\n2.5: Cluster Centers")
print(f"Number of clusters: {optimal_k}")
print(f"Shape: {cluster_centers.shape}")  # (3, 22) - 3 clusters, 22 features each
for i, center in enumerate(cluster_centers):
    print(f"\nCluster {i}:")
    print(center)


# ============================================================
# PART 3: EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PART 3: EVALUATION")
print("=" * 60)

# 3.1: Calculate metrics
# Evaluate clustering quality using three metrics:
# - Inertia: Lower is better (compactness of clusters)
# - Silhouette Score: Higher is better (range -1 to 1, cluster separation)
# - Davies-Bouldin Index: Lower is better (cluster similarity)

train_inertia = final_kmeans.inertia_  # Already calculated during training
test_inertia = -final_kmeans.score(X_test)  # Compute for test set
train_silhouette = silhouette_score(X_train, train_labels)
test_silhouette = silhouette_score(X_test, test_labels)
train_db = davies_bouldin_score(X_train, train_labels)
test_db = davies_bouldin_score(X_test, test_labels)

print(f"\n3.1: Clustering Metrics")
print(f"\nTraining Set:")
print(f"  Inertia: {train_inertia:.2f}")
print(f"  Silhouette Score: {train_silhouette:.4f}")
print(f"  Davies-Bouldin Index: {train_db:.4f}")

print(f"\nTest Set:")
print(f"  Inertia: {test_inertia:.2f}")
print(f"  Silhouette Score: {test_silhouette:.4f}")
print(f"  Davies-Bouldin Index: {test_db:.4f}")

# 3.2: Compare metrics for different k values
# Validate our choice of k=3 by comparing metrics across multiple k values
print(f"\n3.2: Comparing metrics for k=2 to k=10")
print(f"\n{'k':<5} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Inertia':<15}")
print("-" * 60)

silhouette_scores = []
db_scores = []
inertia_scores = []
k_comparison_range = range(2, 11)

# Train models with different k values and calculate metrics
for k in k_comparison_range:
    temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    temp_kmeans.fit(X_train)
    labels = temp_kmeans.labels_

    # Calculate all three metrics
    sil = silhouette_score(X_train, labels)
    db = davies_bouldin_score(X_train, labels)
    inertia = temp_kmeans.inertia_

    # Store for plotting
    db_scores.append(db)
    inertia_scores.append(inertia)

    print(f"{k:<5} {sil:<15.4f} {db:<20.4f} {inertia:<15.2f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Silhouette Score (Higher is better)
ax1.plot(k_comparison_range, silhouette_scores, marker="o", color="blue", linewidth=2)
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette Score")
ax1.set_title("Silhouette Score vs k\n(Higher is Better)")
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
ax1.legend()

# Plot 2: Davies-Bouldin Index (Lower is better)
ax2.plot(k_comparison_range, db_scores, marker="s", color="red", linewidth=2)
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Davies-Bouldin Index")
ax2.set_title("Davies-Bouldin Index vs k\n(Lower is Better)")
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
ax2.legend()

# Plot 3: Inertia (for reference)
ax3.plot(k_comparison_range, inertia_scores, marker="^", color="green", linewidth=2)
ax3.set_xlabel("Number of clusters (k)")
ax3.set_ylabel("Inertia")
ax3.set_title("Inertia vs k\n(Lower is Better)")
ax3.grid(True, alpha=0.3)
ax3.axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
ax3.legend()

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# 3.4: Visualize clustering results
# Use PCA to reduce 22 dimensions to 2D for visualization
# PCA finds the directions of maximum variance in the data
print(f"\n3.4: Visualizing clustering with PCA...")

pca = PCA(n_components=2)  # Reduce to 2 principal components
X_train_pca = pca.fit_transform(X_train)  # Transform training data
X_test_pca = pca.transform(X_test)  # Transform test data
cluster_centers_pca = pca.transform(cluster_centers)  # Transform centroids

print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Create side-by-side plots for training and test sets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot training set clusters
scatter1 = ax1.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=train_labels,  # Color by cluster assignment
    cmap="viridis",
    alpha=0.6,
    s=50,
    edgecolors="k",
    linewidth=0.5,
)
# Plot centroids as red X markers
ax1.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    c="red",
    marker="X",
    s=300,
    edgecolors="black",
    linewidth=2,
    label="Centroids",
)
ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax1.set_title("Training Set Clustering")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label="Cluster")

# Plot test set clusters
scatter2 = ax2.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=test_labels,  # Color by cluster assignment
    cmap="viridis",
    alpha=0.6,
    s=50,
    edgecolors="k",
    linewidth=0.5,
)
# Plot centroids as red X markers
ax2.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    c="red",
    marker="X",
    s=300,
    edgecolors="black",
    linewidth=2,
    label="Centroids",
)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax2.set_title("Test Set Clustering")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label="Cluster")

# Save and display the visualization
plt.tight_layout()
plt.savefig("clustering_visualization.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nClustering visualization complete!")
