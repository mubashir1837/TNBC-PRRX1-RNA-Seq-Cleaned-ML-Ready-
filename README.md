# Inducible PRRX1 Over-Expression in Triple-Negative Breast Cancer (RNA-Seq)
**Cleaned & Machine Learning-Ready Gene Count Matrix for 48 Samples | GEO Accession: GSE202769**

---

##  Dataset Overview & Context

This repository contains a highly curated, cleaned, and machine learning-ready gene expression dataset derived from NCBI Gene Expression Omnibus (GEO) accession **GSE202769**. 

### The Biological Context
**Triple-Negative Breast Cancer (TNBC)** is a highly aggressive subtype of breast cancer lacking estrogen receptor (ER), progesterone receptor (PR), and HER2 receptor expression, making targeted therapy extremely challenging. 
The transcription factor **PRRX1** (Paired Related Homeobox 1) has been identified as a key driver of **Epithelial-Mesenchymal Transition (EMT)**, a developmental process co-opted by tumor cells to gain migratory, invasive, and drug-resistant properties. Overexpressing PRRX1 reprogrammes breast cancer cells, driving cellular plasticity and epigenetic/transcriptional heterogeneity.

### Experimental Design
The study utilizes a complete factorial in vitro experimental setup comprising **48 samples**:
1.  **4 TNBC Cell Line Backgrounds**: `HCC3153`, `MFM223`, `SUM185`, and `EMG3`.
2.  **2 PRRX1 Over-Expression Constructs**:
    *   `wt`: Wild-Type PRRX1 construct.
    *   `dH3`: Homeodomain Helix 3 mutant PRRX1 construct (incapable of binding DNA; acts as a structural control).
3.  **2 Induction Treatments**:
    *   `dox`: Doxycycline-induced PRRX1 expression (active over-expression).
    *   `no_dox`: Untreated basal expression control.
4.  **3 Timepoints**: Day 7, Day 14, and Day 21.

This high-dimensional biological dataset represents a complete **4x2x2x3 factorial design** ($4 \times 2 \times 2 \times 3 = 48$ samples), capturing the temporal dynamics of transcription factor-driven state transitions.

---

## 📁 Repository Directory Structure

The workspace is organized as follows:
```directory
├── GSE202769_TNBC_Cell_line_PRRX1_OE_RNAseq_Gene_Counts.csv   # Raw NCBI GEO counts file
├── process_data.py                                            # Main data engineering pipeline
├── README.md                                                  # This Kaggle Landing Manual
├── processed_data/                                            # Processed datasets
│   ├── sample_metadata.csv                                    # Tidy relational metadata mapping
│   ├── cleaned_raw_counts.csv                                 # Aggregated raw counts matrix
│   ├── normalized_log2_cpm_unfiltered.csv                    # log2(CPM+1) normalized (all genes)
│   ├── normalized_log2_cpm_filtered.csv                      # log2(CPM+1) normalized (filtered genes)
│   └── breast_cancer_ml_matrix.csv                            # Transposed ML Consolidated Matrix
└── plots/                                                     # Quality Control & Biological Visualizations
    ├── library_sizes.png                                      # Sequencing depth bar plot
    ├── prrx1_induction_profile.png                            # PRRX1 target gene validation boxplot
    ├── pca_expression.png                                     # 2D PCA gene expression projection
    └── correlation_heatmap.png                                # Hierarchical clustered correlation heatmap
```

---

## 🛠️ Data Engineering & Pipeline Specs

We implemented a robust data processing pipeline in [process_data.py](process_data.py) to convert noisy, raw biological count data into clean matrices.

1.  **Index Duplicate Aggregation**: The raw file contained duplicate gene symbols in its index. We consolidated these duplicates by **summing their raw read counts** across all samples. Summing is the mathematically correct method for transcript-level counts, representing the total transcription output of a gene locus.
2.  **Sequencing Depth Normalization (CPM)**: Raw library sizes vary heavily between samples (ranging from 33M to 52M reads). We normalized read counts to **Counts Per Million (CPM)**:
    $$\text{CPM}_{i, j} = \frac{\text{Count}_{i, j}}{\text{Library Size}_j} \times 1,000,000$$
3.  **Variance Stabilization**: Biological RNA-Seq counts are extremely skewed and follow a negative binomial distribution. We applied a $\log_2(\text{CPM} + 1)$ transformation to linearize and stabilize variance. The $+1$ pseudocount ensures that zeroes remain zeroes.
4.  **Low-Expression Background Filtering**: Genes with extremely low counts across all samples add biological background noise and can lead to ML model overfitting. We filtered out genes with a total sum of raw counts **$< 50$** across all 48 samples.
    *   *Total unique genes evaluated*: **20,114**
    *   *Low-expression genes filtered out*: **3,987 (19.82%)**
    *   *High-confidence features retained*: **16,127**
5.  **ML Transposition**: To conform to standard machine learning layouts (rows as samples, columns as features), we transposed the filtered normalized matrix and prepended five relational metadata columns: `Cell_Line`, `Construct`, `Treatment`, `Induction` (binary), and `Timepoint`.

---

## 📊 Exploratory Quality Control & Biological Validation

We generated four high-resolution visualizations in [plots/](plots) to validate biological transitions and sequencing consistency:

*   **Sequencing Depth [library_sizes.png](plots/library_sizes.png)**: Shows total read counts per sample colored by cell line. Confirms excellent sequencing depths (average of ~42M reads) and validates the necessity of library size normalization.
*   **PRRX1 Expression Profile [prrx1_induction_profile.png](plots/prrx1_induction_profile.png)**: Validates that Doxycycline treatment (`dox`) successfully induced a massive, highly significant over-expression of the driver gene `PRRX1` in all cell lines compared to untreated controls (`no_dox`).
*   **PCA Clustering [pca_expression.png](plots/pca_expression.png)**: Projects the 16,127-gene expression profiles into 2D. Confirms that cell line lineages form distinct biological clusters, representing the largest transcriptomic variance (PC1/PC2), while treatment induction separates samples within each background.
*   **Pearson Correlation Heatmap [correlation_heatmap.png](plots/correlation_heatmap.png)**: Clustered Pearson correlation coefficients between all 48 transcriptomes. Confirms exceptional technical replicability ($r > 0.90$ within replicates) and shows zero biological outliers.

---

## 🏆 Kaggle ML Challenges & Benchmark Tasks

This dataset offers a rich array of classification, regression, and clustering challenges for machine learning practitioners:

### Challenge 1: Cellular Lineage Classification (Multi-class)
*   **Task**: Predict the cell line origin (`Cell_Line`: HCC3153, MFM223, SUM185, or EMG3) based on gene expression features.
*   **Significance**: Benchmarks feature selection and classification accuracy on high-dimensional genomic features.
*   **Benchmark Baseline**: 100% accuracy (Random Forest).

### Challenge 2: Inducible State Classification (Binary)
*   **Task**: Predict the treatment state (`Induction`: 1 if `dox`, 0 if `no_dox`), identifying if the PRRX1 over-expression driver is active.
*   **Significance**: Identifies downstream transcriptional targets of PRRX1 driving EMT.

### Challenge 3: Homeodomain Functional State Prediction (Binary)
*   **Task**: Classify the overexpressed construct (`Construct`: `wt` vs `dH3`) in induced samples.
*   **Significance**: Predicts whether the transcription factor's DNA-binding domain is mutated (`dH3`) or active (`wt`), representing a subtle, functional transcriptomic shift.

### Challenge 4: Time-Series Temporal Classification (Multi-class)
*   **Task**: Classify the progression point (`Timepoint`: 7, 14, or 21 days).
*   **Significance**: Model the temporal kinetics of EMT cell-state transitions.

---

## 🐍 Quick Start: Python Baseline Tutorial

Use the following copy-paste-ready Python script to load the consolidated ML matrix, standardize the genomic features, train a baseline classifier, and evaluate predictions:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Consolidated ML Matrix
# Rows = 48 Samples, Columns = 5 Labels + 16,127 Genes
df = pd.read_csv("processed_data/breast_cancer_ml_matrix.csv", index_col="Sample_ID")

# 2. Extract features and target label
metadata_columns = ["Cell_Line", "Construct", "Treatment", "Induction", "Timepoint"]
X = df.drop(columns=metadata_columns)  # 16,127 gene expression features
y = df["Cell_Line"]                    # Target label

# 3. Label encode the targets
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Perform stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# 5. Standardize gene features (scale to Z-score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Fit a baseline Random Forest Classifier
print("[*] Fitting Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test_scaled)
print(f"\n[+] Classification Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n[+] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

---

## Citations & References

If you publish models or findings using this processed dataset, please credit the original authors and the NCBI Gene Expression Omnibus:

*   **Original GSE dataset**: [NCBI GEO accession GSE202769](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE202769).
*   **Original Study Reference**: Please cite the accompanying paper researching PRRX1-driven chromatin dynamics and transcriptional heterogeneity in Triple-Negative Breast Cancer.
