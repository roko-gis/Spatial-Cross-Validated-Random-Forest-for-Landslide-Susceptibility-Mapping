# Spatial Cross-Validated Random Forest for Landslide Susceptibility Mapping

Research software for landslide susceptibility mapping using Random Forest classification, spatial cross-validation, probability calibration, spatial diagnostics and GIS-based environmental predictors.

Version: 1.0.0
QGIS: 3.x with Python/Processing support
Developed and tested with: QGIS 3.44.13
Author: Rosen Iliev
Affiliation: Space Research and Technology Institute, Bulgarian Academy of Sciences (SRTI-BAS), Remote Sensing and GIS Department
Contact: ilievrosen@space.bas.bg
---

## Funding and Acknowledgements

This software was developed within the framework of the COST Action CA24144 – ANTICIPATE: Extended-range Multi-hazard Predictions and Early Warnings and was nationally co-funded by the project “Space and Ground Technologies for Studying Cascading Multi-hazard Events – Models, Applications, Precursors (SPARC)”, supported under Co-funding Contract No. KP-06-COST/23, dated 18 August 2026, concluded between the Space Research and Technology Institute at the Bulgarian Academy of Sciences (SRTI-BAS) and the Bulgarian National Science Fund (BNSF) within the framework of BNSF-1.009 – “Procedure for Providing National Co-funding for the Participation of Bulgarian Research Teams in Established Actions in Science and Technology.”

The SPARC project provides national co-funding for the participation of Bulgarian research teams in COST Action CA24144 – ANTICIPATE, supported by COST (European Cooperation in Science and Technology).

For more information, please visit the official COST Action CA24144 page and the ANTICIPATE website.

## Overview

This repository contains a Python-based **QGIS Processing algorithm** for landslide susceptibility modelling using Random Forest classification and spatially structured model validation.

The workflow integrates:

* point-based landslide inventory processing;
* automatic positive and background/pseudo-absence sample generation;
* raster alignment and optional GDAL resampling;
* categorical predictor detection and one-hot encoding;
* missing-value handling;
* mutual information, VIF and spatial autocorrelation diagnostics;
* optional iterative VIF-based predictor removal;
* Random Forest classification;
* spatial block cross-validation;
* optional spatially aware probability calibration;
* probabilistic model evaluation;
* block-bootstrap confidence intervals for AUC;
* optional independent group-based hold-out testing;
* spatial residual analysis;
* raw and Gaussian-smoothed susceptibility mapping;
* probability ambiguity mapping;
* relative five-class susceptibility mapping;
* optional SHAP-based model interpretation;
* serialization of the final model and model metadata.

The method is intended for **research and comparative susceptibility assessment**. It is not an operational landslide warning, hazard or risk-assessment system.

---

## Key Features

### Spatial Cross-Validation

Model evaluation uses `StratifiedGroupKFold` with spatial groups derived from raster coordinates.

Each modelling observation is assigned to a spatial block according to its raster row and column position. Observations within the same spatial block are therefore assigned to the same validation group.

This reduces the risk of overly optimistic performance estimates caused by spatial dependence between neighbouring observations.

The default configuration uses:

* 5 spatial folds;
* 1000 m spatial block size;
* shuffled group assignment with a fixed Random Forest/CV seed.

The number of folds and block size can be modified through the QGIS Processing interface.

---

### Distance-Constrained Background Sampling

Non-landslide/background samples are selected from valid reference-raster cells located outside a user-defined exclusion distance from mapped landslides.

The user can specify:

* minimum distance from landslides in metres;
* background/positive sampling ratio;
* additional exclusion buffer;
* maximum number of background samples.

The default background/positive ratio is `2.0`.

The implementation limits the number of background samples to **50,000 cells**.

The actual number of selected background cells may be lower when fewer valid candidate cells are available.

The distance is converted from metres to raster pixels using the reference raster pixel size.

---

### Raster Alignment and Resampling

All predictor rasters are checked against the reference raster for:

* coordinate reference system;
* raster origin;
* pixel size;
* raster dimensions.

Misaligned predictors can automatically be reprojected/resampled to the reference grid using GDAL `Warp`.

The resampling method is selected according to the detected predictor type:

* **Nearest neighbour** for detected categorical rasters;
* **Bilinear** resampling for continuous predictors.

This functionality is enabled by default.

If resampling is disabled, misaligned rasters are handled using a legacy crop/pad procedure and a warning is generated.

---

### Categorical Predictor Encoding

The algorithm attempts to identify categorical predictors using their layer names and raster values.

A predictor is considered for one-hot encoding when:

* its name matches a recognised categorical keyword;
* its values are integer-like;
* it contains between 2 and 50 unique classes.

Each detected class is converted to a separate binary predictor.

Categorical detection is heuristic and should therefore be verified by the user.

---

### Missing-Value Handling

Predictor NoData values are converted to `NaN`.

The optional **Impute missing predictor values** parameter controls whether samples containing missing predictor values are retained in the initial modelling dataset.

When this option is disabled, observations containing at least one missing predictor are removed from the training/evaluation sample set.

---

### Probability Calibration

Random Forest predictions are calibrated using isotonic regression through scikit-learn's `CalibratedClassifierCV`.

Two calibration modes are available.

#### Standard Calibration

The default mode uses the standard internal cross-validation mechanism of `CalibratedClassifierCV`.

#### Spatially Aware Calibration

An optional **spatially aware probability calibration** mode uses spatial groups for the internal calibration cross-validation.

This is intended to provide a more conservative calibration procedure in the presence of spatial dependence.

---

### Predictor Diagnostics

The workflow calculates several predictor diagnostics using the modelling sample data.

#### Mutual Information

`mutual_info_classif` is used to estimate the relationship between each predictor and the binary landslide/non-landslide response.

The report includes:

* mutual information score;
* relative MI contribution in percent;
* predictor ranking.

#### Variance Inflation Factor

VIF is calculated using linear regression between each predictor and the remaining predictors.

A user-defined VIF threshold can be applied to iteratively remove predictors with excessive multicollinearity.

The default threshold is:

```
10.0
```
The current implementation removes the predictor with the highest VIF and recalculates VIF values iteratively until the threshold is satisfied or only one predictor remains.

#### Predictor Moran's I

Moran's I is calculated for each predictor to provide a diagnostic of spatial autocorrelation.

The analysis uses a k-nearest-neighbour spatial weighting structure and permutation-based significance testing.

---

### Spatial Residual Diagnostics

For each spatial cross-validation fold, Moran's I is calculated for the prediction residuals:

```
residual = observed class - predicted probability
```

The report includes:

* Moran's I;
* permutation-based two-sided p-value.

This provides an additional diagnostic of remaining spatial structure in model residuals.

A significant residual spatial autocorrelation may indicate that spatial structure remains unexplained by the predictors and modelling strategy.

---

### Optional Hold-Out Test Set

An optional independent test fraction can be specified.

When enabled, `GroupShuffleSplit` is first used to separate spatial groups into:

```
Training data
Test data
```

Spatial cross-validation is then performed only on the training data.

The final model is fitted on the training set and evaluated once on the held-out test set.

The test-set report includes:

* ROC-AUC;
* PR-AUC / Average Precision;
* Brier score;
* sensitivity;
* specificity;
* Matthews correlation coefficient.

The default test fraction is `0.20`.

A value of `0` disables the hold-out test set.

---

### Block-Bootstrap Confidence Interval

The AUC confidence interval is calculated from the pooled out-of-fold predictions produced during spatial cross-validation.

The current implementation uses **spatial blocks as bootstrap units**.

The default configuration uses:

```
200 bootstrap iterations
```

For each bootstrap iteration, spatial groups are sampled with replacement and all observations belonging to the selected groups are included in the bootstrap sample.

The resulting 95% confidence interval is based on the 2.5th and 97.5th percentiles of the bootstrap AUC distribution.

---

### SHAP Model Interpretation

SHAP analysis is optional.

If the `shap` Python package is available, mean absolute SHAP values can be calculated for a subset of the training data.

The implementation:

* samples up to 500 observations;
* accesses the Random Forest classifiers contained within the calibrated model;
* calculates Tree SHAP values;
* averages SHAP contributions across the calibrated classifiers;
* reports mean absolute SHAP values by predictor.

SHAP computation can be computationally expensive and is therefore disabled by default.

---

### Pre-Trained Models

An existing serialized `.pkl` model can be supplied through the optional **Pre-trained model** parameter.

When a valid pre-trained model is supplied:

* model training is skipped;
* spatial cross-validation is skipped;
* the supplied model is used for raster prediction;
* susceptibility maps are generated;
* the supplied model is copied to the output directory;
* model metadata and outputs are generated.

The user should ensure that the predictor configuration is compatible with the pre-trained model.

---

## Methodological Workflow

```
Landslide inventory
        │
        ▼
Positive sample extraction
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Reference raster                 Predictor rasters
        │                              │
        │                       Alignment check
        │                              │
        │                       Optional GDAL Warp
        │                              │
        │                       Categorical detection
        │                              │
        │                       One-hot encoding
        │                              │
        └──────────────┬───────────────┘
                       ▼
                Sample extraction
                       │
             Background sampling
                       │
                       ▼
                Missing-value handling
                       │
                       ▼
          MI / VIF / Moran's I diagnostics
                       │
                       ▼
              Optional VIF removal
                       │
                       ▼
              Optional hold-out split
                       │
                       ▼
              Spatial block definition
                       │
                       ▼
          StratifiedGroupKFold validation
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Model performance     Residual Moran's I
             │
             ▼
      Block-bootstrap AUC CI
             │
             ▼
     Final calibrated RF model
             │
             ▼
       Raster-wide prediction
             │
       ┌─────┼───────────┐
       ▼     ▼           ▼
      Raw  Smoothed   Ambiguity
      probability      map
             │
             ▼
     Five-class susceptibility
             │
             ├──► GeoTIFF outputs
             ├──► Serialized model
             ├──► Metadata JSON
             └──► CSV diagnostic report
```

---

## Input Data

### Landslide Inventory

The current implementation expects a **point-based landslide inventory**.

Each point represents a known landslide location within the area covered by the reference raster.

Multiple inventory points falling within the same raster cell are treated as one modelling observation.

Only point geometries are accepted.

Polygon or line inventories must therefore be converted to representative points before processing.

Inventory points falling outside the valid area of the reference raster are ignored.

The inventory coordinates should be compatible with the reference raster coordinate reference system.

---

### Reference Raster

The reference raster defines:

* analysis extent;
* raster dimensions;
* pixel size;
* spatial grid;
* coordinate reference system;
* valid analysis cells.

Its first raster band is used to identify valid and NoData cells.

The reference raster should therefore cover the complete modelling area.

---

### Predictor Rasters

Predictors are supplied as raster layers and may represent:

* terrain variables;
* elevation;
* slope;
* aspect;
* curvature;
* geology;
* lithology;
* soil properties;
* land cover;
* hydrological variables;
* climatic variables;
* other environmental conditioning factors.

Predictors should have appropriate spatial coverage and meaningful NoData definitions.

The algorithm can automatically align predictors to the reference grid when the resampling option is enabled.

---

## Sample Generation

### Positive Samples

Positive samples are extracted from mapped landslide locations.

Only points located within valid reference-raster cells are retained.

Multiple landslide points mapping to the same raster cell are collapsed into a single positive observation.

---

### Background Samples

Background samples are randomly selected from valid reference-raster cells outside the exclusion distance around positive landslide cells.

The exclusion zone is based on raster-cell distance.

Parameters include:

| Parameter                  | Default |
| -------------------------- | ------: |
| Minimum distance           |    75 m |
| Background/positive ratio  |     2.0 |
| Additional buffer          |     0 m |
| Maximum background samples |  50,000 |

If an additional buffer is supplied, it is added to the minimum exclusion distance.

The number of background samples is limited by:

* requested background/positive ratio;
* number of available candidate cells;
* maximum background sample limit.

At least 10 background candidates must be available after applying the exclusion distance.

---

## Predictor Preparation

Each predictor raster is checked against the reference raster.

The following properties are compared:

* CRS;
* origin;
* pixel size;
* raster dimensions.

When resampling is enabled, a misaligned raster is reprojected and resampled to the reference grid.

Continuous predictors use bilinear resampling.

Detected categorical predictors use nearest-neighbour resampling.

Categorical predictors containing between 2 and 50 integer-like classes can subsequently be one-hot encoded.

NoData values equal to the configured NoData value are converted to `NaN`.

---

## Random Forest Model

The classifier is implemented using scikit-learn's `RandomForestClassifier`.

The default configuration is:

| Parameter                |              Default |
| ------------------------ | -------------------: |
| Number of trees          |                  250 |
| Maximum depth            |                   10 |
| Minimum samples per leaf |                    4 |
| Class weighting          | `balanced_subsample` |
| Number of parallel jobs  |                 `-1` |
| Random seed              |                   42 |

The Random Forest is combined with:

```
SimpleImputer(strategy="median")
        │
        ▼
CalibratedClassifierCV
        │
        ▼
RandomForestClassifier
```

The calibration method is isotonic regression.

---

## Spatial Group Definition

Spatial groups are derived from raster coordinates.

The user specifies a spatial block size in metres. This is converted into raster pixels using the reference raster pixel size.

A group identifier is generated from:

```
row // block_pixels
column // block_pixels
```

The resulting spatial blocks are used by `StratifiedGroupKFold`.

The block size should be selected according to the spatial scale of the landslide processes and predictor dependence being investigated.

A larger block size generally provides a more stringent test of spatial transferability but may reduce the number of available spatial groups.

---

## Spatial Cross-Validation

Spatial model evaluation uses:

```python
StratifiedGroupKFold
```

with shuffled spatial groups and the configured random seed.

The default number of folds is:

```
5
```

The algorithm requires at least as many unique spatial groups as requested CV folds.

For each fold:

1. spatial groups are separated into training and validation sets;
2. a calibrated Random Forest is fitted using the training data;
3. predictions are generated for the spatially separated validation data;
4. performance metrics are calculated;
5. residual Moran's I is calculated.

The resulting predictions are pooled to calculate the overall block-bootstrap confidence interval for AUC.

---

## Probability Calibration

The Random Forest classifier is wrapped in `CalibratedClassifierCV` using isotonic regression.

Two modes are available:

### Standard Calibration

Standard internal calibration cross-validation is used.

This is the default because it is computationally less demanding.

### Spatially Aware Calibration

When enabled, spatial groups are supplied to the calibration cross-validation.

This mode attempts to reduce spatial leakage during probability calibration itself.

Because the additional spatial calibration can fail for datasets with insufficient or unsuitable group/class distributions, the implementation automatically falls back to standard calibration if necessary.

Spatial calibration should therefore be regarded as an optional conservative modelling configuration rather than a guarantee of statistical independence.

---

## Model Evaluation

Performance is calculated from validation predictions generated during spatial cross-validation.

### ROC-AUC

Measures discrimination between landslide and background observations across classification thresholds.

Higher values indicate better discrimination.

### PR-AUC / Average Precision

Measures ranking performance using the precision-recall relationship and is particularly informative under class imbalance.

### Brier Score

Measures probabilistic prediction error.

Lower values indicate better probabilistic accuracy.

### Sensitivity

The proportion of positive landslide observations correctly classified at a probability threshold of `0.5`.

### Specificity

The proportion of background observations correctly classified at a probability threshold of `0.5`.

### Matthews Correlation Coefficient

MCC provides a balanced summary of binary classification performance.

### Moran's I

Moran's I is calculated for spatial residuals to diagnose remaining spatial autocorrelation.

The report also provides a permutation-based p-value.

---

## Held-Out Test Set

When the test fraction is greater than zero, spatial groups are first separated using:

```python
GroupShuffleSplit
```

The resulting training data are used for spatial cross-validation and final model fitting.

The held-out groups are not used during training.

The final model is then evaluated against the test observations.

The default test fraction is:

```
0.20
```

Set the parameter to `0` to disable the hold-out test.

---

## Final Susceptibility Mapping

After spatial cross-validation, a final calibrated Random Forest model is fitted using the available training data.

Raster prediction is performed in chunks to limit memory usage.

The default prediction chunk size is:

```
50,000 cells
```

The workflow generates four raster products:

1. raw calibrated probability;
2. Gaussian-smoothed susceptibility;
3. probability ambiguity;
4. five-class susceptibility.

---

## Gaussian Smoothing

The raw probability surface can be spatially smoothed using a Gaussian filter.

The default smoothing parameter is:

```
sigma = 1.0 pixel
```

The implementation uses **normalized Gaussian smoothing**.

The valid-cell mask is smoothed together with the prediction surface so that NoData cells do not act as zero-probability observations at valid-cell boundaries.

The resulting smoothed surface is therefore a spatially modified model output.

**Smoothed susceptibility values should not automatically be interpreted as directly calibrated probabilities.**

---

## Probability Ambiguity

An ambiguity surface is calculated as:

```
ambiguity = 1 - 2 × |probability - 0.5|
```

Therefore:

* values close to `0` correspond to predictions close to 0 or 1;
* values close to `1` correspond to predictions close to 0.5.

The ambiguity output is a **probability-derived diagnostic proxy**, not a formal confidence interval or statistical uncertainty estimate.

---

## Susceptibility Classes

The continuous susceptibility surface is classified into five relative classes:

1. **Very Low**
2. **Low**
3. **Moderate**
4. **High**
5. **Very High**

Three classification methods are available.

### Percentile

Default method.

The class boundaries are calculated from the 20th, 40th, 60th and 80th percentiles of the valid smoothed susceptibility values.

### Fixed Thresholds

Fixed probability thresholds are used:

```
0.2
0.4
0.6
0.8
```

### Jenks Natural Breaks

The Fisher-Jenks natural-breaks optimization is applied to the valid susceptibility values.

For computational efficiency, a maximum of 10,000 values is used to calculate the Jenks breaks.

The resulting five classes are:

```
1 = Very Low
2 = Low
3 = Moderate
4 = High
5 = Very High
```

The classes represent **relative susceptibility within the study area** and should not be interpreted as absolute probability categories.

---

## Output Products

The output filenames include the user-defined prefix, software version and execution timestamp.

| Output                  | File pattern                                    | Description                                  |
| ----------------------- | ----------------------------------------------- | -------------------------------------------- |
| Raw probability         | `{PREFIX}_Probability_Raw_v1.0.0_*.tif`         | Raw calibrated model prediction              |
| Smoothed susceptibility | `{PREFIX}_Susceptibility_Smoothed_v1.0.0_*.tif` | Normalized Gaussian-smoothed prediction      |
| Probability ambiguity   | `{PREFIX}_Probability_Ambiguity_v1.0.0_*.tif`   | Probability-derived ambiguity proxy          |
| Susceptibility classes  | `{PREFIX}_Susceptibility_Classes_v1.0.0_*.tif`  | Five relative susceptibility classes         |
| Serialized model        | `{PREFIX}_model_v1.0.0_*.pkl`                   | Serialized final calibrated model            |
| Model metadata          | `{PREFIX}_model_metadata_v1.0.0_*.json`         | Model configuration and environment metadata |
| CSV report              | `{PREFIX}_summary_v1.0.0_*.csv`                 | Performance and predictor diagnostics        |

All GeoTIFF outputs preserve the reference raster's geotransform and projection.

Continuous outputs use floating-point GeoTIFF format with:

```
NoData = -9999
```

The susceptibility class raster uses:

```
0 = NoData
1 = Very Low
2 = Low
3 = Moderate
4 = High
5 = Very High
```

---

## CSV Report

The CSV report contains several sections.

### Data Summary

Includes:

* software version;
* output directory;
* number of original predictors;
* number of predictors used in the final model;
* total samples;
* positive samples;
* background samples;
* number of spatial blocks;
* pixel size;
* minimum sampling distance.

### Overall CV Performance

Includes mean, standard deviation, minimum and maximum for:

* AUC;
* PR-AUC;
* Brier score;
* sensitivity;
* specificity;
* MCC;
* Moran's I;
* Moran's I p-value.

### AUC Confidence Interval

The report contains the 95% block-bootstrap confidence interval for AUC.

### Spatial CV Results

Individual fold results are reported for all successfully evaluated folds.

### Held-Out Test Set

When enabled, independent test-set metrics are included.

### Feature Diagnostics

For each predictor, the report contains:

* rank;
* feature name;
* MI contribution;
* VIF;
* Moran's I;
* Moran's I p-value;
* whether the predictor was used in the final model;
* removal reason, if applicable.

### SHAP Importance

When SHAP analysis is enabled and successful, mean absolute SHAP importance values are also included.

---

## Model Metadata

A JSON metadata file is generated for each execution.

It records:

* software version;
* execution timestamp;
* CRS;
* raster information;
* predictors used in the model;
* Random Forest parameters;
* spatial CV configuration;
* background sampling configuration;
* test-set configuration;
* classification method;
* VIF threshold;
* NoData handling;
* predictor resampling setting;
* spatial calibration setting;
* calibration method;
* Python version;
* QGIS version;
* GDAL version;
* NumPy version;
* SciPy version;
* scikit-learn version;
* SHAP availability;
* random seed;
* spatial block size;
* minimum sampling distance;
* additional buffer distance.

This metadata is intended to support reproducibility and documentation of individual modelling runs.

---

## Reproducibility

The model configuration uses a default random seed of:

```
42
```

However, reproducibility depends on more than the Random Forest seed.

Important factors include:

* input datasets;
* landslide inventory;
* predictor preprocessing;
* raster alignment;
* raster resampling;
* QGIS/Python environment;
* scientific package versions;
* Random Forest parameters;
* spatial block size;
* number of CV folds;
* background/positive sampling ratio;
* sampling distance;
* test-set fraction;
* VIF threshold;
* calibration mode;
* smoothing parameter;
* classification method.

The generated JSON metadata and CSV report should therefore be retained together with the model and output rasters.

The trained model is serialized using `joblib`.

For scientific publication and archival reproducibility, the corresponding software release should be associated with a Zenodo DOI.

---

## QGIS Processing Interface

The algorithm extends:

```python
QgsProcessingAlgorithm
```

and runs through the **QGIS Processing Toolbox**.

**Processing group:**

```
Landslide Susceptibility
```

**Algorithm:**

```
Spatial CV Random Forest – Landslide Susceptibility (v1.0.0)
```

**Algorithm ID:**

```
spatial_rf_landslide_susceptibility
```

The algorithm can also be executed from the QGIS Python Console:

```python
import processing

processing.run(
    "spatial_rf_landslide_susceptibility",
    {
        # algorithm parameters
    }
)
```

---

## Processing Parameters

### Basic Parameters

| Parameter             |      Default |
| --------------------- | -----------: |
| Landslide inventory   |     Required |
| Base/reference raster |     Required |
| Predictor rasters     | At least one |
| Minimum distance      |         75 m |
| Spatial block size    |       1000 m |
| Output prefix         |        `LSM` |
| Output folder         |     Required |

### Random Forest Parameters

| Parameter                | Default |
| ------------------------ | ------: |
| Number of trees          |     250 |
| Maximum tree depth       |      10 |
| Minimum samples per leaf |       4 |

### Validation Parameters

| Parameter           |  Default |
| ------------------- | -------: |
| CV folds            |        5 |
| Test set fraction   |     0.20 |
| Spatial calibration | Disabled |

### Sampling Parameters

| Parameter                  | Default |
| -------------------------- | ------: |
| Background/positive ratio  |     2.0 |
| Additional buffer          |     0 m |
| Maximum background samples |  50,000 |

### Mapping Parameters

| Parameter                |    Default |
| ------------------------ | ---------: |
| Gaussian smoothing sigma |        1.0 |
| Classification method    | Percentile |

### Predictor Diagnostics

| Parameter                       | Default |
| ------------------------------- | ------: |
| VIF threshold                   |    10.0 |
| Impute missing predictor values | Enabled |
| Resample predictors             | Enabled |

### Optional Analysis

| Parameter         |  Default |
| ----------------- | -------: |
| Compute SHAP      | Disabled |
| Pre-trained model |     None |

---

## Installation

The workflow is designed for **QGIS 3.x** with Python and Processing support and was developed and tested with **QGIS 3.44.13**.

### Dependencies

Required Python packages include:

* NumPy
* SciPy
* scikit-learn
* joblib

Optional:

* `shap`

The following components are normally provided by the QGIS installation:

* GDAL;
* PyQGIS;
* PyQt;
* QGIS Processing framework.

Additional installation information can be provided in:

* `INSTALLATION.md`

---

## Limitations

The main limitations of the current implementation include:

* point-based landslide inventory assumption;
* polygon inventories require preprocessing;
* dependence on correctly prepared predictor rasters;
* categorical-variable detection is partly heuristic;
* raster resampling can affect predictor values;
* background sampling represents pseudo-absence/background rather than confirmed non-landslide observations;
* raster-cell distance is used for background exclusion;
* spatial blocking reduces but does not eliminate spatial dependence;
* the choice of spatial block size can substantially affect estimated model performance;
* spatial calibration may fail for datasets with insufficient spatial groups or unsuitable class distributions;
* isotonic calibration can overfit when calibration data are limited;
* VIF is based on linear relationships and may not capture nonlinear dependence;
* automatic VIF removal can remove scientifically relevant correlated predictors;
* Moran's I is a diagnostic and does not itself correct spatial autocorrelation;
* Gaussian smoothing modifies the raw model predictions;
* smoothed susceptibility should not automatically be interpreted as calibrated probability;
* the ambiguity raster is a diagnostic proxy rather than formal statistical uncertainty;
* susceptibility classes are relative rather than absolute;
* Jenks classification is based on the distribution of predictions within the study area;
* fixed probability classes should not be interpreted as empirically validated hazard thresholds without independent calibration;
* the maximum number of background samples is 50,000;
* SHAP computation depends on the installed SHAP version and model compatibility;
* predictions from pre-trained models require compatible predictor structure and preprocessing;
* model serialization can depend on the versions of Python/scientific libraries used.

The outputs should therefore be interpreted in the context of the input inventory, predictor quality, sampling design, spatial validation strategy and study-area characteristics.

---

## Scientific Interpretation

The resulting susceptibility surface represents the modelled spatial propensity for landslide occurrence given the supplied inventory and environmental predictors.

The output should be interpreted as a **susceptibility assessment**, not as:

* a deterministic prediction of future landslides;
* a landslide hazard probability in the strict probabilistic sense;
* a risk map;
* an operational early-warning product.

In particular, the five susceptibility classes are relative categories and depend on the selected classification method.

Independent validation, sensitivity analysis and domain-specific expert interpretation are recommended before using the results for scientific conclusions or decision support.

---

## Software Citation

When using this software in scientific research, cite the specific software version, repository and associated DOI.

For version-specific citation and archival information, use the **Zenodo DOI associated with the repository release**.

---

## Scientific References

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Proceedings of the 31st International Conference on Neural Information Processing Systems
4768 - 4777

Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. Methods in Ecology and Evolution, 12, 1620–1633. https://doi.org/10.1111/2041-210X.13650

Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1–2), 17–23. https://doi.org/10.1093/biomet/37.1-2.17

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In Proceedings of the 22nd International Conference on Machine Learning (ICML '05) (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors. Quality & Quantity, 41, 673–690. https://doi.org/10.1007/s11135-006-9018-6

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

---

## License

**MIT License**

Copyright (c) 2026 Rosen Iliev

See the `LICENSE` file for the complete license text.
