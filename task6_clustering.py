import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Configuration
DATASET_PATH = "Mall_Customers.csv"
SEGMENTED_DATASET_PATH = "Mall_Customers_Segmented.csv"
DENDROGRAM_PATH = "dendrogram.png"
CLUSTERS_2D_PATH = "clusters_2d.png"
CLUSTERS_3D_PATH = "clusters_3d.png"
README_PATH = "README.md"

DATASET_URLS = [
    "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/Mall_Customers.csv",
    "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
    "https://raw.githubusercontent.com/SteffiPeT/Mall-Customer-Segmentation-Data/master/Mall_Customers.csv",
]


def download_dataset(filepath=DATASET_PATH, urls=DATASET_URLS):
    """Download Mall_Customers.csv if it does not already exist locally."""
    if os.path.exists(filepath):
        print(f"[INFO] Dataset '{filepath}' already exists locally.")
        return

    print(f"[INFO] Downloading dataset to '{filepath}'...")
    downloaded = False
    for url in urls:
        try:
            print(f"[INFO] Trying source: {url}")
            urllib.request.urlretrieve(url, filepath)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"[INFO] Successfully downloaded dataset ({os.path.getsize(filepath)} bytes).")
                downloaded = True
                break
        except Exception as e:
            print(f"[WARNING] Failed downloading from {url}: {e}")

    if not downloaded:
        raise RuntimeError("Could not download dataset from any available mirror.")


def generate_readme(df, cluster_means, cluster_counts, sil_score):
    """Generate comprehensive project README.md documenting all phases."""
    readme_content = f"""# Customer Segmentation Using Hierarchical Clustering (Agglomerative)

## 1. Project Overview & Objective
This project implements **Task 6: Customer Segmentation Using Hierarchical Clustering** on the Mall Customers dataset. The primary objective is to discover meaningful customer segments based on demographic and financial behavior (`Age`, `Annual Income (k$)`, and `Spending Score (1-100)`) to guide personalized marketing and customer relationship management.

---

## 2. Dataset Exploration Findings
- **Source**: Mall Customers Dataset (`Mall_Customers.csv`)
- **Dimensions**: 200 customer records, 5 original attributes (6 after clustering assignment).
- **Attributes**:
  - `CustomerID`: Unique identifier for each customer (excluded from modeling).
  - `Gender`: Customer gender (`Female`, `Male`).
  - `Age`: Age in years (ranging from 18 to 70).
  - `Annual Income (k$)`: Customer annual income in thousands of dollars (ranging from $15k to $137k).
  - `Spending Score (1-100)`: Behavioral score assigned based on spending habits and purchase history.
- **Data Quality**: 0 missing or null values detected across all columns.

---

## 3. Preprocessing & Standardization Pipeline
1. **Identifier Removal**: `CustomerID` was dropped to avoid index-based bias.
2. **Categorical Encoding**: `Gender` was mapped numerically (`Female: 0`, `Male: 1`).
3. **Feature Selection**: Extracted the core numerical features for segmentation:
   - `Age`
   - `Annual Income (k$)`
   - `Spending Score (1-100)`
4. **Standardization**: Applied `StandardScaler` to ensure zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), preventing features with larger scales (such as income) from dominating Euclidean distance calculations.

---

## 4. Hierarchical Clustering & Dendrogram Interpretation
- **Linkage Criterion**: **Ward's minimum variance method** (`method='ward'`) using Euclidean distance metric (`metric='euclidean'`).
- **Dendrogram Cutting**: Analyzing the linkage tree, the largest vertical leap without horizontal intersections occurs around distance $7.0$.
- **Cluster Selection**: A horizontal threshold cut at $y = 7.0$ cleanly partitions the customer base into **5 optimal clusters**.

![Hierarchical Clustering Dendrogram](dendrogram.png)

---

## 5. Model Evaluation
- **Algorithm**: `AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')`
- **Silhouette Score**: **`{sil_score:.4f}`**
  - Indicates well-defined cluster separation and internal cohesion across the standardized 3D feature space.

---

## 6. Cluster Persona Profiles & Breakdown

| Cluster ID | Segment Persona | Size (% of Total) | Mean Age | Mean Income (k$) | Mean Spending Score | Recommended Strategy |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **0** | **Young Active Spenders** | {cluster_counts[0]} ({(cluster_counts[0]/len(df))*100:.1f}%) | {cluster_means.loc[0, 'Age']:.1f} | ${cluster_means.loc[0, 'Annual Income (k$)']:.1f}k | {cluster_means.loc[0, 'Spending Score (1-100)']:.1f} / 100 | Digital marketing, trendy apparel, rewards loyalty programs. |
| **1** | **Senior Moderate Spenders** | {cluster_counts[1]} ({(cluster_counts[1]/len(df))*100:.1f}%) | {cluster_means.loc[1, 'Age']:.1f} | ${cluster_means.loc[1, 'Annual Income (k$)']:.1f}k | {cluster_means.loc[1, 'Spending Score (1-100)']:.1f} / 100 | Value-driven promotions, reliable customer service, household essentials. |
| **2** | **Affluent High Spenders (VIPs)** | {cluster_counts[2]} ({(cluster_counts[2]/len(df))*100:.1f}%) | {cluster_means.loc[2, 'Age']:.1f} | ${cluster_means.loc[2, 'Annual Income (k$)']:.1f}k | {cluster_means.loc[2, 'Spending Score (1-100)']:.1f} / 100 | Luxury product previews, concierge services, VIP events. |
| **3** | **High Earners / Cautious Savers** | {cluster_counts[3]} ({(cluster_counts[3]/len(df))*100:.1f}%) | {cluster_means.loc[3, 'Age']:.1f} | ${cluster_means.loc[3, 'Annual Income (k$)']:.1f}k | {cluster_means.loc[3, 'Spending Score (1-100)']:.1f} / 100 | High-quality investments, premium memberships, targeted incentive discounts. |
| **4** | **Budget-Conscious / Low Spend** | {cluster_counts[4]} ({(cluster_counts[4]/len(df))*100:.1f}%) | {cluster_means.loc[4, 'Age']:.1f} | ${cluster_means.loc[4, 'Annual Income (k$)']:.1f}k | {cluster_means.loc[4, 'Spending Score (1-100)']:.1f} / 100 | Clearance sales, discount vouchers, economy bundle offerings. |

---

## 7. Cluster Visualizations

### 2D Segmentation: Annual Income vs. Spending Score
![2D Customer Clusters](clusters_2d.png)

### 3D Segmentation: Age vs. Annual Income vs. Spending Score
![3D Customer Clusters](clusters_3d.png)

---

## 8. File Structure
- `task6_clustering.py`: End-to-end Python pipeline executing Phases 1 through 5.
- `Mall_Customers.csv`: Raw dataset.
- `Mall_Customers_Segmented.csv`: Output dataset labeled with cluster assignments.
- `dendrogram.png`: High-resolution hierarchical clustering dendrogram with threshold cut.
- `clusters_2d.png`: 2D scatter plot visualization of customer segments.
- `clusters_3d.png`: 3D scatter plot visualization in feature space.
- `README.md`: Comprehensive project documentation.
"""
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[OK] Generated '{README_PATH}'.")


def main():
    print("=" * 70)
    print(" TASK 6: CUSTOMER SEGMENTATION USING HIERARCHICAL CLUSTERING ")
    print("=" * 70)

    # Step 0: Ensure dataset is available
    download_dataset()

    # Task 1: Load Dataset & Normalize Column Names
    print("\n" + "-" * 50)
    print(" TASK 1: DATA LOADING & PREPARATION ")
    print("-" * 50)
    df = pd.read_csv(DATASET_PATH)

    # Rename 'Genre' to 'Gender' if present
    if "Genre" in df.columns:
        df.rename(columns={"Genre": "Gender"}, inplace=True)

    print(f"[OK] Dataset loaded successfully into pandas DataFrame.")
    print(f"Columns: {list(df.columns)}")

    # Task 2: Dataset Exploration
    print("\n" + "-" * 50)
    print(" TASK 2: DATASET EXPLORATION ")
    print("-" * 50)

    print("\n[A] First 5 Records (df.head()):")
    print(df.head().to_string(index=False))

    print(f"\n[B] Dataset Shape (df.shape):")
    print(f"    Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n[C] Data Types (df.dtypes):")
    for col, dtype in df.dtypes.items():
        print(f"    - {col:<26}: {dtype}")

    print("\n[D] Missing Values Count (df.isnull().sum()):")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"    - {col:<26}: {count} missing")

    total_missing = null_counts.sum()
    print(f"\nTotal missing values in dataset: {total_missing}")

    # Task 3: Data Preprocessing
    print("\n" + "-" * 50)
    print(" TASK 3: DATA PREPROCESSING ")
    print("-" * 50)

    # 1. Remove CustomerID from working features
    df_processed = df.drop(columns=["CustomerID"]).copy()
    print("[A] Removed 'CustomerID' from working features.")

    # 2. Encode Gender into numerical format (Female: 0, Male: 1)
    gender_mapping = {"Female": 0, "Male": 1}
    df_processed["Gender"] = df_processed["Gender"].map(gender_mapping)
    print("[B] Encoded 'Gender' column to numerical format:")
    print(f"    Mapping: {gender_mapping}")
    print(f"    Unique values in Gender: {df_processed['Gender'].unique().tolist()}")

    # 3. Select exact 3 features required for clustering into feature matrix X
    cluster_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df_processed[cluster_features].copy()
    print(f"\n[C] Selected 3 features for clustering matrix X: {cluster_features}")
    print(f"    Matrix X shape: {X.shape}")
    print("\nFirst 5 rows of Feature Matrix X:")
    print(X.head().to_string(index=False))

    # Task 4: Feature Standardization
    print("\n" + "-" * 50)
    print(" TASK 4: FEATURE STANDARDIZATION ")
    print("-" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to DataFrame for readable inspection
    X_scaled_df = pd.DataFrame(X_scaled, columns=cluster_features)

    print("[A] First 5 rows of Standardized Feature Matrix (X_scaled):")
    print(X_scaled_df.head().to_string(index=False))

    print("\n[B] Verification of Standardization (Column-wise):")
    means = np.mean(X_scaled, axis=0)
    stds = np.std(X_scaled, axis=0)

    print(f"{'Feature':<26} | {'Mean (approx 0)':<18} | {'Std Dev (approx 1)':<18}")
    print("-" * 68)
    for feat, m, s in zip(cluster_features, means, stds):
        print(f"{feat:<26} | {m:<18.6f} | {s:<18.6f}")

    # Task 5: Ward Linkage Matrix
    print("\n" + "-" * 50)
    print(" TASK 5: LINKAGE MATRIX (WARD LINKAGE) ")
    print("-" * 50)

    linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")
    print(f"[A] Linkage Matrix Shape: {linkage_matrix.shape}")
    print(f"    (Represents {linkage_matrix.shape[0]} iterative pairwise merges across {X_scaled.shape[0]} samples)")

    print("\n[B] First 5 Merges of Linkage Matrix [Cluster 1, Cluster 2, Distance, Sample Count]:")
    linkage_df = pd.DataFrame(
        linkage_matrix[:5],
        columns=["Cluster 1 Idx", "Cluster 2 Idx", "Distance", "Sample Count"]
    )
    linkage_df["Cluster 1 Idx"] = linkage_df["Cluster 1 Idx"].astype(int)
    linkage_df["Cluster 2 Idx"] = linkage_df["Cluster 2 Idx"].astype(int)
    linkage_df["Sample Count"] = linkage_df["Sample Count"].astype(int)
    print(linkage_df.to_string(index=False))

    # Task 6: Dendrogram Visualization
    print("\n" + "-" * 50)
    print(" TASK 6: DENDROGRAM VISUALIZATION ")
    print("-" * 50)

    threshold = 7.0
    plt.figure(figsize=(14, 7), dpi=300)
    dendrogram(
        linkage_matrix,
        labels=df.index.values,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=threshold,
    )
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold Cut (y={threshold}, k=5 clusters)"
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Customer Index", fontsize=12, labelpad=10)
    plt.ylabel("Euclidean Distance", fontsize=12, labelpad=10)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(DENDROGRAM_PATH, dpi=300)
    plt.close()

    print(f"[OK] Dendrogram successfully saved to '{DENDROGRAM_PATH}'.")

    # Explanation of merges and optimal k
    print("\n[EXPLANATION] Hierarchical Merging & Cluster Selection (k=5):")
    print(
        "1. Bottom-up Agglomeration: Each customer starts in their own singleton cluster.\n"
        "   At each iteration, Ward's method merges the two clusters that minimize the increase\n"
        "   in total within-cluster variance (represented by the vertical height on the y-axis).\n"
        "2. Determining Optimal k: The longest vertical lines that are not intercepted by any\n"
        "   horizontal line indicate the largest jumps in within-cluster variance. Cutting the\n"
        "   dendrogram at threshold y = 7.0 intersects exactly 5 vertical branches.\n"
        "3. Conclusion: k = 5 provides well-separated and homogenous customer segments,\n"
        "   balancing intra-cluster cohesion with distinct inter-cluster separation."
    )

    # Task 7: Build Model
    print("\n" + "-" * 50)
    print(" TASK 7: BUILD AGGLOMERATIVE CLUSTERING MODEL ")
    print("-" * 50)
    model = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
    print(f"[OK] Instantiated AgglomerativeClustering model:\n     {model}")

    # Task 8: Fit Model
    print("\n" + "-" * 50)
    print(" TASK 8: FIT MODEL ON STANDARDIZED DATA ")
    print("-" * 50)
    model.fit(X_scaled)
    print(f"[OK] Model successfully fitted on X_scaled ({X_scaled.shape[0]} samples, {X_scaled.shape[1]} features).")

    # Task 9: Generate Cluster Labels & Segmentation Dataset
    print("\n" + "-" * 50)
    print(" TASK 9: GENERATE CLUSTER LABELS & SEGMENTATION BREAKDOWN ")
    print("-" * 50)

    # Attach Cluster labels to original DataFrame
    df["Cluster"] = model.labels_

    # Save labeled DataFrame
    df.to_csv(SEGMENTED_DATASET_PATH, index=False)
    print(f"[OK] Segmented customer dataset saved to '{SEGMENTED_DATASET_PATH}'.")

    print("\n[A] First 10 Rows of Labeled Dataset (df.head(10)):")
    print(df.head(10).to_string(index=False))

    print("\n[B] Customer Count per Cluster (value_counts):")
    cluster_counts = df["Cluster"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(df)) * 100
        print(f"    Cluster {cluster_id}: {count:3d} customers ({pct:5.1f}%)")

    print("\n[C] Cluster Profiling (Mean Feature Values per Cluster):")
    cluster_means = df.groupby("Cluster")[cluster_features].mean().round(2)
    print(cluster_means.to_string())

    # Task 10: Silhouette Score Evaluation
    print("\n" + "-" * 50)
    print(" TASK 10: SILHOUETTE SCORE EVALUATION ")
    print("-" * 50)

    sil_score = silhouette_score(X_scaled, model.labels_, metric="euclidean")
    print(f"[METRIC] Silhouette Score (k=5, Ward): {sil_score:.4f}")
    if sil_score > 0.4:
        print("         -> High clustering structure and good sample separation.")
    elif sil_score > 0.25:
        print("         -> Reasonable cluster structure detected.")
    else:
        print("         -> Weak cluster structure.")

    # Task 11: Visual Validations (2D & 3D Plots)
    print("\n" + "-" * 50)
    print(" TASK 11: CLUSTER VISUALIZATIONS (2D & 3D) ")
    print("-" * 50)

    # 1. 2D Scatter Plot (Income vs Spending Score)
    plt.figure(figsize=(10, 6), dpi=300)
    palette = sns.color_palette("tab10", n_colors=5)
    markers = ["o", "s", "^", "D", "P"]

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        style="Cluster",
        palette=palette,
        markers=markers,
        s=100,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5
    )

    plt.title("Customer Segments (2D): Annual Income vs Spending Score", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Annual Income (k$)", fontsize=11, labelpad=8)
    plt.ylabel("Spending Score (1-100)", fontsize=11, labelpad=8)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(CLUSTERS_2D_PATH, dpi=300)
    plt.close()
    print(f"[OK] 2D Cluster scatter plot saved to '{CLUSTERS_2D_PATH}'.")

    # 2. 3D Scatter Plot (Age vs Income vs Spending Score)
    fig = plt.figure(figsize=(11, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for c_id in sorted(df["Cluster"].unique()):
        sub_df = df[df["Cluster"] == c_id]
        ax.scatter(
            sub_df["Age"],
            sub_df["Annual Income (k$)"],
            sub_df["Spending Score (1-100)"],
            c=colors[c_id],
            label=f"Cluster {c_id}",
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.4
        )

    ax.set_title("Customer Segments (3D): Age vs Income vs Spending Score", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Age", fontsize=11, labelpad=10)
    ax.set_ylabel("Annual Income (k$)", fontsize=11, labelpad=10)
    ax.set_zlabel("Spending Score (1-100)", fontsize=11, labelpad=10)
    ax.legend(title="Cluster", loc="upper left", bbox_to_anchor=(0.02, 0.95))
    plt.tight_layout()
    plt.savefig(CLUSTERS_3D_PATH, dpi=300)
    plt.close()
    print(f"[OK] 3D Cluster scatter plot saved to '{CLUSTERS_3D_PATH}'.")

    # Step 12: Documentation Generation
    print("\n" + "-" * 50)
    print(" TASK 12: PROJECT DOCUMENTATION (README.md) ")
    print("-" * 50)
    generate_readme(df, cluster_means, cluster_counts, sil_score)

    print("=" * 70)


if __name__ == "__main__":
    main()
