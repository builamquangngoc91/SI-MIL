# PathExpert Feature Extraction Pipeline Analysis

## Overview

The `PathExpert_feature_extraction.sh` script runs a comprehensive 9-step pipeline to extract handcrafted pathological features from Whole Slide Images (WSIs). This pipeline processes medical imaging data through cell segmentation, patch extraction, and multiple feature extraction methods to create a rich feature set for machine learning models.

## Prerequisites

Before running the script, ensure the following prerequisites are met:

1. **Environment**: Conda environment named `simil` should be activated
2. **HoVer-Net Output**: Cell segmentation and classification results must exist in `Hovernet_output/json/`
3. **Input Data**: WSI files should be in the `slides/` directory
4. **Directory Structure**: The `data_sample` folder should be properly structured

## Pipeline Workflow

### Step 1: Cell Property Extraction (`extract_properties.py`)

**Purpose**: Extract morphological and statistical properties of individual cells from HoVer-Net segmentation results.

**Input**:

- WSI slides from `data_sample/slides/`
- HoVer-Net JSON output from `data_sample/Hovernet_output/json/`

**Processing**:

- Reads HoVer-Net segmentation masks and classification results
- Extracts cell-level morphological features (area, perimeter, eccentricity, etc.)
- Calculates texture features using Gray-Level Co-occurrence Matrix (GLCM)
- Computes statistical moments (skewness, kurtosis) for cell regions
- Processes cells from different tissue magnifications (40X)

**Output**:

- Cell property files saved to `data_sample/cell_property/`
- Each WSI gets its own `.pickle` file containing cell properties

**Workers**: 10 parallel processes for faster processing

---

### Step 2: Patch Extraction (`deepzoom_tiler_organ.py`)

**Purpose**: Extract image patches from WSIs at specific magnifications suitable for feature extraction.

**Input**:

- WSI files from `data_sample/slides/`

**Processing**:

- Uses DeepZoom technology for efficient multi-resolution image processing
- Extracts patches at 5X magnification with size 224x224 pixels
- Applies tissue detection to avoid background regions
- Creates hierarchical patch structure for each WSI
- Implements overlap handling and boundary conditions

**Output**:

- Image patches saved to `data_sample/patches/[WSI_NAME]/`
- Each patch saved as individual `.jpg` file

**Workers**: 10 parallel processes

---

### Step 3: Patch Dictionary Construction (`patch_dict_list.py`)

**Purpose**: Create indexing structures for efficient patch access during feature extraction.

**Input**:

- Patch directory from `data_sample/patches/`

**Processing**:

- Scans all patch directories and creates comprehensive lists
- Builds dictionary mapping WSI names to patch index ranges
- Creates sequential indexing system for all patches across all WSIs
- Generates both list and dictionary structures for flexible access

**Output**:

- `all_list.pickle`: Sequential list of all patch file paths
- `all_dict.pickle`: Dictionary mapping WSI names to patch indices

---

### Step 4: Cell Statistics Feature Extraction (`extract_cell_statistics_features.py`)

**Purpose**: Extract patch-level statistical features based on cell populations and properties.

**Input**:

- WSI slides, cell properties, and patch indices

**Processing**:

- For each patch, identifies corresponding cells from 40X magnification
- Calculates cell type distributions (neoplastic, inflammatory, connective, dead, non-neoplastic, unknown)
- Computes statistical aggregations: mean, std, min, max, percentiles
- Extracts morphological statistics across different cell types
- Handles coordinate transformation between patch and cell coordinate systems

**Output**:

- Statistical features saved to `data_sample/features/cell_statistics/`
- Each WSI gets feature file containing patch-wise cell statistics

**Workers**: 10 parallel processes

---

### Step 5: Social Network Analysis Features (`extract_sna_features.py`)

**Purpose**: Extract graph-based features representing spatial relationships between cells.

**Input**:

- WSI slides, cell properties, and patch indices

**Processing**:

- Constructs spatial graphs connecting nearby cells
- Implements multiple graph construction strategies (K-NN, radius-based)
- Calculates graph-theoretic features: centrality measures, clustering coefficients
- Computes inter-cell type interaction patterns
- Analyzes spatial organization and cell community structures
- Based on "Cells are Actors" methodology

**Output**:

- SNA features saved to `data_sample/features/sna_statistics/`
- Graph-based spatial relationship features for each patch

**Workers**: 10 parallel processes

---

### Step 6: ATHENA Spatial Features (`extract_athena_spatial_features.py`)

**Purpose**: Extract spatial heterogeneity features using ATHENA (Analysis of Tumor Heterogeneity in pathology images) framework.

**Input**:

- WSI slides, cell properties, and patch indices

**Processing**:

- Implements ATHENA's spatial analysis algorithms
- Calculates spatial entropy and organization measures
- Extracts heterogeneity indices at multiple spatial scales
- Computes tissue architecture features
- Analyzes spatial patterns and clustering behaviors
- Measures local and global spatial statistics

**Output**:

- ATHENA features saved to `data_sample/features/athena_statistics/`
- Spatial heterogeneity measures for each patch

**Workers**: 10 parallel processes

---

### Step 7: Tissue Feature Extraction (`extract_tissue_features.py`)

**Purpose**: Extract tissue-level morphological and compositional features.

**Input**:

- WSI slides, HoVer-Net JSON output, and patch indices

**Processing**:

- Analyzes tissue composition and structure
- Extracts color and texture features from tissue regions
- Calculates tissue density and distribution patterns
- Applies background filtering (threshold=220)
- Computes tissue-specific statistical measures
- Integrates with HoVer-Net cell type information

**Output**:

- Tissue features saved to `data_sample/features/tissue_statistics/`
- Tissue-level morphological and compositional features

**Workers**: 10 parallel processes
**Background Threshold**: 220 (for tissue/background separation)

---

### Step 8: Feature Combination (`club_features.py`)

**Purpose**: Combine all extracted features into unified feature vectors for each patch.

**Input**:

- All feature directories from previous steps
- Column name definitions from pickle files
- Patch dictionary for indexing

**Processing**:

- Loads features from all extraction modules:
  - Cell statistics features
  - Social Network Analysis features
  - ATHENA spatial features
  - Tissue features
- Aligns features across different modalities using patch indices
- Handles missing data and feature normalization
- Creates unified feature matrix with consistent indexing
- Removes specified cell types (configurable via `--remove_cell_type`)

**Output**:

- Combined feature file: `cell_athena_sna.csv`
- Unified feature matrix ready for machine learning

**Configuration**:

- `--remove_cell_type "none"`: No cell types removed (can be changed to remove specific types)

---

### Step 9: Data Filtering and Normalization (`data_filtering.py`)

**Purpose**: Apply final filtering, normalization, and prepare train/test splits.

**Input**:

- Combined features from `cell_athena_sna.csv`
- Train/test split definition from `train_test_dict.json`
- Patch dictionary for sample identification

**Processing**:

- Applies patch filtering based on tissue content and quality heuristics
- Removes patches with insufficient tissue or poor quality
- Implements binning-based feature normalization
- Creates separate feature sets for training and testing
- Applies statistical normalization techniques
- Handles class balancing and sample selection
- Configurable neoplastic cell filtering

**Output**:

- `train_dict.pickle` / `train_list.pickle`: Training samples and features
- `test_dict.pickle` / `test_list.pickle`: Testing samples and features
- `trainfeat_deep.pth` / `testfeat_deep.pth`: PyTorch tensors for deep learning
- `binned_hcf.csv`: Normalized feature matrix

**Configuration**:

- `--bins 10`: Number of bins for feature normalization
- `--norm_feat "bin"`: Binning-based normalization method
- `--remove_noneoplastic "False"`: Keep non-neoplastic cells

## Output Directory Structure

After successful execution, the following structure is created:

```
data_sample/
├── cell_property/           # Step 1 output
│   └── [WSI_NAME].pickle
├── patches/                 # Step 2 output
│   ├── all_dict.pickle     # Step 3 output
│   ├── all_list.pickle     # Step 3 output
│   └── [WSI_NAME]/
│       └── *.jpg
└── features/
    ├── cell_statistics/     # Step 4 output
    ├── sna_statistics/      # Step 5 output
    ├── athena_statistics/   # Step 6 output
    ├── tissue_statistics/   # Step 7 output
    ├── cell_athena_sna.csv # Step 8 output
    ├── train_dict.pickle    # Step 9 output
    ├── test_dict.pickle     # Step 9 output
    ├── train_list.pickle    # Step 9 output
    ├── test_list.pickle     # Step 9 output
    ├── trainfeat_deep.pth   # Step 9 output
    ├── testfeat_deep.pth    # Step 9 output
    └── binned_hcf.csv       # Step 9 output
```

## Performance Considerations

- **Total Processing Time**: Varies significantly based on:

  - Number and size of WSI files
  - Available computational resources
  - Complexity of tissue samples

- **Memory Usage**: High memory requirements due to:

  - Large WSI file processing
  - Multiple parallel workers
  - Feature matrix storage

- **Disk Space**: Substantial storage needed for:
  - Original WSI files (typically GBs each)
  - Extracted patches (thousands per WSI)
  - Feature files and intermediate results

## Error Handling and Monitoring

- Each step includes progress tracking using `tqdm`
- Multiprocessing errors are logged
- Missing input files cause graceful failures
- Intermediate outputs allow pipeline resumption from failed steps

## Technical Dependencies

- **OpenSlide**: WSI file reading and processing
- **HoVer-Net**: Cell segmentation and classification
- **scikit-image**: Image processing and feature extraction
- **NetworkX**: Graph analysis for SNA features
- **NumPy/Pandas**: Numerical computations and data handling
- **PyTorch**: Deep learning tensor operations

## Customization Options

The pipeline can be customized by modifying:

1. **Worker counts**: Adjust `--workers` parameter for each step
2. **Cell type filtering**: Modify `--remove_cell_type` in step 8
3. **Normalization method**: Change `--norm_feat` in step 9
4. **Background threshold**: Adjust tissue detection sensitivity
5. **Patch size and magnification**: Modify extraction parameters

This comprehensive pipeline transforms raw WSI data into rich, multi-modal feature representations suitable for downstream machine learning tasks in computational pathology.
