#!/usr/bin/env python3
"""
Breast Cancer RNA-seq Gene Counts Processing and Machine Learning Preparation Pipeline.
GEO Accession: GSE202769

This script cleans raw gene count data, aggregates duplicates, filters out low-expression noise,
normalizes counts to CPM (Counts Per Million), log2-transforms expression values, structures a
machine learning-ready transposed matrix, and generates high-resolution biological QC figures.
"""

import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- CONFIGURATION & PATHS ---
RAW_FILE = "GSE202769_TNBC_Cell_line_PRRX1_OE_RNAseq_Gene_Counts.csv"
OUTPUT_DIR = "processed_data"
PLOT_DIR = "plots"

# Low expression filter thresholds
# Keep genes with a total sum of at least 50 counts across all 48 samples
MIN_TOTAL_COUNTS = 50 

def setup_directories():
    """Create processed data and plot directories if they do not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"[*] Directories configured: '{OUTPUT_DIR}' and '{PLOT_DIR}'")

def log_step(name, start_time):
    """Log elapsed time for a processing step."""
    elapsed = time.time() - start_time
    print(f"[+] Completed: {name} | Time taken: {elapsed:.2f} seconds\n" + "-"*60)

def load_raw_data():
    """Load the raw gene counts CSV file and perform initial inspection."""
    print(f"[*] Loading raw dataset from '{RAW_FILE}'...")
    start_time = time.time()
    
    # Read CSV, setting the first column (empty header) as the index (gene symbols)
    df = pd.read_csv(RAW_FILE, index_col=0)
    
    # Clean the index name
    df.index.name = "Gene_Symbol"
    
    print(f"    - Dimensions: {df.shape[0]} genes (rows) x {df.shape[1]} samples (columns)")
    print(f"    - Missing values (Nulls) in raw file: {df.isnull().sum().sum()}")
    print(f"    - Duplicate Gene Symbols in raw index: {df.index.duplicated().sum()}")
    
    log_step("Data Loading & Inspection", start_time)
    return df

def clean_and_aggregate(df):
    """
    Handle duplicates and missing values.
    Aggregates duplicate gene symbols by summing their raw read counts (biologically correct method).
    """
    print("[*] Beginning Data Cleaning and Aggregation...")
    start_time = time.time()
    
    # 1. Fill missing values (if any) with 0
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"    - Filling {null_count} missing value(s) with 0.")
        df = df.fillna(0)
    
    # 2. Aggregating index duplicates by summing counts
    duplicated_genes = df.index.duplicated().sum()
    if duplicated_genes > 0:
        print(f"    - Found {duplicated_genes} duplicated Gene Symbol(s).")
        print("    - Aggregating duplicate genes by summing raw counts...")
        
        # Group by the index (Gene_Symbol) and sum the values
        df_clean = df.groupby(df.index).sum()
        
        # Ensure counts remain integer types
        df_clean = df_clean.astype(int)
        
        print(f"    - Dimensions after duplicate aggregation: {df_clean.shape[0]} unique genes x {df_clean.shape[1]} samples")
    else:
        df_clean = df.copy()
        print("    - No duplicates found. Index is clean.")
    
    # Save the cleaned raw counts
    raw_out_path = os.path.join(OUTPUT_DIR, "cleaned_raw_counts.csv")
    df_clean.to_csv(raw_out_path)
    print(f"    - Cleaned raw count matrix saved to: '{raw_out_path}'")
    
    log_step("Aggregation & Cleaning", start_time)
    return df_clean

def normalize_cpm(df):
    """Normalize raw counts to Counts Per Million (CPM) to adjust for differences in library size."""
    print("[*] Normalizing Raw Counts to Counts Per Million (CPM)...")
    start_time = time.time()
    
    # Calculate library size (sum of counts for each column/sample)
    lib_sizes = df.sum(axis=0)
    
    # Perform CPM normalization: (count / library_size) * 1,000,000
    df_cpm = df.divide(lib_sizes, axis=1) * 1e6
    
    log_step("CPM Normalization", start_time)
    return df_cpm, lib_sizes

def log_transform(df_cpm):
    """Apply log2(CPM + 1) transformation for variance stabilization."""
    print("[*] Applying Log2(CPM + 1) Transformation...")
    start_time = time.time()
    
    # log2(expression + 1)
    df_log = np.log2(df_cpm + 1)
    
    # Save the unfiltered normalized log2-cpm matrix
    unfiltered_out = os.path.join(OUTPUT_DIR, "normalized_log2_cpm_unfiltered.csv")
    df_log.to_csv(unfiltered_out)
    print(f"    - Unfiltered normalized expression matrix saved to: '{unfiltered_out}'")
    
    log_step("Log2 Transformation", start_time)
    return df_log

def filter_low_expression(df_raw, df_log):
    """
    Filters out genes with low total raw counts across all samples to reduce noise.
    Threshold: Keep genes with a total sum of counts >= MIN_TOTAL_COUNTS.
    """
    print(f"[*] Filtering Low-Expression Genes (Total counts >= {MIN_TOTAL_COUNTS})...")
    start_time = time.time()
    
    # Calculate sum of raw counts for each gene
    total_raw_counts = df_raw.sum(axis=1)
    
    # Create a boolean mask for genes meeting the threshold
    keep_mask = total_raw_counts >= MIN_TOTAL_COUNTS
    
    filtered_genes_count = len(keep_mask) - keep_mask.sum()
    print(f"    - Total genes evaluated: {len(keep_mask)}")
    print(f"    - Genes filtered out: {filtered_genes_count} ({filtered_genes_count / len(keep_mask) * 100:.2f}%)")
    print(f"    - Genes retained: {keep_mask.sum()}")
    
    # Apply mask to the log-transformed normalized data
    df_log_filtered = df_log[keep_mask]
    
    # Save the filtered normalized log2-cpm matrix
    filtered_out = os.path.join(OUTPUT_DIR, "normalized_log2_cpm_filtered.csv")
    df_log_filtered.to_csv(filtered_out)
    print(f"    - Filtered normalized expression matrix saved to: '{filtered_out}'")
    
    log_step("Noise Filtering", start_time)
    return df_log_filtered

def parse_sample_metadata(sample_name):
    """
    Parse experimental variables from sample columns.
    Example columns: 
    - HCC3153_wt_dox_day_7_RNA
    - HCC3153_wt_no_dox_day_7_RNA
    """
    parts = sample_name.split("_")
    cell_line = parts[0]
    construct = parts[1]
    
    # Identify treatment: check if 'no' is followed by 'dox'
    if "no" in parts and "dox" in parts:
        treatment = "no_dox"
        induction = 0
    elif "dox" in parts:
        treatment = "dox"
        induction = 1
    else:
        treatment = "unknown"
        induction = 0
    
    # Search for numeric timepoint in parts
    day_idx = parts.index("day") if "day" in parts else -1
    if day_idx != -1 and day_idx + 1 < len(parts):
        timepoint = int(parts[day_idx + 1])
    else:
        # Fallback regex search
        match = re.search(r'day_(\d+)', sample_name)
        timepoint = int(match.group(1)) if match else 0
        
    return {
        "Cell_Line": cell_line,
        "Construct": construct,
        "Treatment": treatment,
        "Induction": induction,
        "Timepoint": timepoint
    }

def build_ml_matrix(df_log_filtered, lib_sizes):
    """
    Transposes normalized expression data and prepends experimental metadata columns
    to build a machine learning-ready transposed matrix. Also exports a standalone metadata file.
    """
    print("[*] Building Machine Learning-Ready Transposed Dataset...")
    start_time = time.time()
    
    # 1. Transpose expression matrix: Rows (Samples) x Columns (Genes)
    df_ml = df_log_filtered.T
    df_ml.index.name = "Sample_ID"
    
    # 2. Extract and parse metadata for all samples
    metadata_list = []
    for sample_id in df_ml.index:
        meta = parse_sample_metadata(sample_id)
        meta["Sample_ID"] = sample_id
        meta["Library_Size"] = int(lib_sizes[sample_id])
        metadata_list.append(meta)
        
    df_meta = pd.DataFrame(metadata_list)
    # Reorder columns to place Sample_ID first
    df_meta = df_meta[["Sample_ID", "Cell_Line", "Construct", "Treatment", "Induction", "Timepoint", "Library_Size"]]
    
    # Save standalone metadata CSV
    meta_out = os.path.join(OUTPUT_DIR, "sample_metadata.csv")
    df_meta.to_csv(meta_out, index=False)
    print(f"    - Standalone relational sample metadata saved to: '{meta_out}'")
    
    # 3. Prepend metadata fields to transposed gene expression matrix
    # Merge metadata and expression on Sample_ID
    df_meta_indexed = df_meta.set_index("Sample_ID")
    # Dropping Library Size from the ML training matrix to avoid leakage
    df_meta_ml_fields = df_meta_indexed.drop(columns=["Library_Size"])
    
    df_ml_final = pd.concat([df_meta_ml_fields, df_ml], axis=1)
    
    # Save the consolidated ML matrix
    ml_out = os.path.join(OUTPUT_DIR, "breast_cancer_ml_matrix.csv")
    df_ml_final.to_csv(ml_out)
    print(f"    - ML-Ready consolidated dataset saved to: '{ml_out}'")
    print(f"      Matrix shape: {df_ml_final.shape[0]} samples x {df_ml_final.shape[1]} features (5 labels + {df_ml.shape[1]} genes)")
    
    log_step("ML Matrix Construction", start_time)
    return df_ml_final, df_meta

# --- QUALITY CONTROL & VISUALIZATION ---

def plot_library_sizes(df_meta):
    """Plot sequencing depth (total read counts) across all 48 samples."""
    print("[*] Generating Sequencing Library Size Distribution...")
    plt.figure(figsize=(14, 6))
    
    # Sort metadata by Cell Line and Treatment for logical grouping
    df_sorted = df_meta.sort_values(by=["Cell_Line", "Treatment"])
    
    # Use curated HSL-tailored palette (vibrant and professional)
    palette = sns.color_palette("husl", len(df_sorted["Cell_Line"].unique()))
    
    # Map cell lines to colors
    cell_line_colors = dict(zip(df_sorted["Cell_Line"].unique(), palette))
    colors = [cell_line_colors[cl] for cl in df_sorted["Cell_Line"]]
    
    # Plot bars
    bars = plt.bar(df_sorted["Sample_ID"], df_sorted["Library_Size"] / 1e6, color=colors, edgecolor='black', alpha=0.85)
    
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Read Counts (Millions)", fontsize=12, fontweight='bold')
    plt.title("Sequencing Library Size (Depth) across Breast Cancer RNA-Seq Samples", fontsize=14, fontweight='bold', pad=15)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cell_line_colors[cl], edgecolor='black', label=cl) for cl in cell_line_colors]
    plt.legend(handles=legend_elements, title="Cell Line Background", title_fontsize='11', fontsize='10', loc='upper right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(PLOT_DIR, "library_sizes.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    - Plot saved: '{out_path}'")

def plot_prrx1_induction(df_ml_final):
    """
    Biological validation: Plot the expression profile of the target gene PRRX1
    to verify that doxycycline successfully induced over-expression (OE) in WT construct.
    """
    print("[*] Generating Target Gene (PRRX1) Over-Expression Verification Profile...")
    
    # Check if PRRX1 is in the columns
    prrx1_symbol = "PRRX1"
    if prrx1_symbol not in df_ml_final.columns:
        # Search case-insensitively
        matches = [col for col in df_ml_final.columns if col.upper() == "PRRX1"]
        if matches:
            prrx1_symbol = matches[0]
        else:
            print("    [!] Warning: PRRX1 gene symbol not found in processed matrix. Skipping induction plot.")
            return

    # Extract relevant columns
    df_prrx1 = df_ml_final[["Cell_Line", "Construct", "Treatment", prrx1_symbol]].copy()
    df_prrx1[prrx1_symbol] = df_prrx1[prrx1_symbol].astype(float)
    
    # Create subplots comparing Wild-Type PRRX1 construct to dH3 Mutant construct
    plt.figure(figsize=(12, 6))
    
    # Use elegant colors: 'no_dox' (basal control) as HSL grey, 'dox' (induction) as sleek red/orange
    hue_colors = {"no_dox": "#4A607A", "dox": "#D95D39"}
    
    sns.boxplot(
        data=df_prrx1, 
        x="Cell_Line", 
        y=prrx1_symbol, 
        hue="Treatment", 
        palette=hue_colors,
        linewidth=1.5,
        fliersize=4
    )
    
    # Overlay stripplot for sample-level clarity
    sns.stripplot(
        data=df_prrx1,
        x="Cell_Line",
        y=prrx1_symbol,
        hue="Treatment",
        palette=hue_colors,
        dodge=True,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.7,
        legend=False
    )
    
    # Add construct labels at the top
    plt.ylabel("Expression log2(CPM + 1)", fontsize=12, fontweight='bold')
    plt.xlabel("Cell Line Background", fontsize=12, fontweight='bold')
    plt.title(f"Target Gene Verification: Inducible {prrx1_symbol} Expression under Doxycycline", fontsize=14, fontweight='bold', pad=15)
    plt.legend(title="Dox Inducible Treatment", title_fontsize='11', fontsize='10')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(PLOT_DIR, "prrx1_induction_profile.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    - Plot saved: '{out_path}'")

def plot_pca(df_ml_final):
    """
    Perform Principal Component Analysis (PCA) on gene expression features
    to verify sample clustering by Cell Line background and Dox treatment.
    """
    print("[*] Performing PCA and generating 2D projection...")
    
    # Separate features (genes) from metadata
    meta_cols = ["Cell_Line", "Construct", "Treatment", "Induction", "Timepoint"]
    expression_matrix = df_ml_final.drop(columns=meta_cols)
    
    # Run PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(expression_matrix)
    
    # Create PCA DataFrame
    df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    df_pca["Cell_Line"] = df_ml_final["Cell_Line"].values
    df_pca["Treatment"] = df_ml_final["Treatment"].values
    df_pca["Construct"] = df_ml_final["Construct"].values
    
    # Calculate explained variance ratio
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Plot setup
    plt.figure(figsize=(10, 8))
    
    # Clean, vibrant HSL tailered markers and colors
    markers = {"no_dox": "o", "dox": "X"}
    cell_line_colors = {"HCC3153": "#D95D39", "MFM223": "#1A936F", "SUM185": "#2E4057", "EMG3": "#F18F01"}
    
    # Scatter plot
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Cell_Line",
        style="Treatment",
        palette=cell_line_colors,
        markers=markers,
        s=150,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.85
    )
    
    # Labels and design
    plt.xlabel(f"PC1 ({var_exp[0]:.2f}% Explained Variance)", fontsize=12, fontweight='bold')
    plt.ylabel(f"PC2 ({var_exp[1]:.2f}% Explained Variance)", fontsize=12, fontweight='bold')
    plt.title("Principal Component Analysis (PCA) of TNBC Cell Lines (log2 CPM+1 Retained Genes)", fontsize=14, fontweight='bold', pad=15)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=11, title="Biological Factors", title_fontsize=12)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(PLOT_DIR, "pca_expression.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    - Plot saved: '{out_path}'")

def plot_sample_correlation(df_ml_final):
    """
    Compute sample-to-sample Pearson correlation and plot a clustered heatmap
    to visualize transcriptome similarities and find potential biological outliers.
    """
    print("[*] Computing Pearson correlation matrix and plotting clustered heatmap...")
    
    # Separate features from metadata
    meta_cols = ["Cell_Line", "Construct", "Treatment", "Induction", "Timepoint"]
    expression_matrix = df_ml_final.drop(columns=meta_cols)
    
    # Compute correlation matrix between rows (samples)
    # Transposing back to get sample-by-sample correlation
    corr_matrix = expression_matrix.T.corr(method="pearson")
    
    # Generate clustered heatmap (clustermap)
    plt.figure(figsize=(12, 10))
    
    # Curate annotation colors for Cell Line mapping at the heatmap axes
    cell_lines = df_ml_final["Cell_Line"]
    cell_line_colors = {"HCC3153": "#D95D39", "MFM223": "#1A936F", "SUM185": "#2E4057", "EMG3": "#F18F01"}
    row_colors = cell_lines.map(cell_line_colors)
    
    g = sns.clustermap(
        corr_matrix,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="coolwarm",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.1,
        figsize=(12, 12),
        cbar_kws={'label': 'Pearson Correlation (r)'}
    )
    
    # Adjust plot titles and labels
    g.fig.suptitle("Sample-to-Sample Pearson Correlation Clustered Heatmap", fontsize=16, fontweight='bold', y=1.02)
    
    # Create cell line color legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cell_line_colors[cl], edgecolor='black', label=cl) for cl in cell_line_colors]
    g.ax_heatmap.legend(handles=legend_elements, title="Cell Line Background", title_fontsize='11', fontsize='10', loc='lower left', bbox_to_anchor=(1.2, 0))
    
    out_path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    g.savefig(out_path, dpi=300)
    plt.close()
    print(f"    - Plot saved: '{out_path}'")

def main():
    """Execute the entire bioinformatics processing pipeline."""
    overall_start = time.time()
    print("="*60)
    print("      BREAST CANCER RNA-SEQ GENE COUNTS PROCESSING PIPELINE")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # 1. Load data
    df_raw = load_raw_data()
    
    # 2. Clean duplicates and missing values
    df_clean = clean_and_aggregate(df_raw)
    
    # 3. CPM Library normalization
    df_cpm, lib_sizes = normalize_cpm(df_clean)
    
    # 4. Log2 transformation
    df_log = log_transform(df_cpm)
    
    # 5. Low-expression noise filtering
    df_log_filtered = filter_low_expression(df_clean, df_log)
    
    # 6. ML Matrix building & Sample Metadata generation
    df_ml_final, df_meta = build_ml_matrix(df_log_filtered, lib_sizes)
    
    # 7. Generate visualizations
    print("\n[*] Starting Exploratory Data Analysis & Visualization...")
    plot_library_sizes(df_meta)
    plot_prrx1_induction(df_ml_final)
    plot_pca(df_ml_final)
    plot_sample_correlation(df_ml_final)
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Pipeline executed successfully!")
    print(f"Total time elapsed: {time.time() - overall_start:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()
