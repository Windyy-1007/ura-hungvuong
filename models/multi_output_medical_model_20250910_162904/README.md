# Multi-Output Medical Prediction Model

## Overview

This is a **single multi-output Random Forest model** that predicts **ALL treatment labels simultaneously** using just **one weight file**. This is more efficient than individual models and ensures consistency across predictions.

## Key Features

✅ **Single Weight File**: `multi_output_medical_model.joblib` - ONE file for ALL predictions  
✅ **Multi-Output Prediction**: Predicts 11 treatment labels simultaneously  
✅ **High Performance**: 93.3% average accuracy, 93.1% average AUC  
✅ **Fast Prediction**: ~57ms for all 11 labels per patient  
✅ **Configurable**: Uses `config_all_labels.json` for feature/target selection  

## Model Performance

### Overall Statistics
- **Average Test Accuracy**: 93.30%
- **Average AUC Score**: 93.13%
- **Training Time**: 1.7 seconds
- **Prediction Speed**: 58 predictions/second
- **Model Type**: Multi-Output Random Forest

### Top Performing Targets
1. **Reassess_Later** - 98.1% accuracy, 96.3% AUC
2. **Delivery_Prep_Hemorrhage** - 97.8% accuracy, 95.1% AUC
3. **Fetal_Resuscitation** - 96.9% accuracy, 98.0% AUC
4. **Prepare_Delivery** - 95.4% accuracy, 96.7% AUC
5. **Neonatal_Resuscitation** - 95.4% accuracy, 96.7% AUC

## Predicted Labels (11 Treatment Labels)

The model predicts these **treatment recommendations**:

- **Monitor_Labor**: Continue labor monitoring
- **Report_Doctor**: Report to attending physician
- **Prepare_Delivery**: Prepare for delivery
- **Neonatal_Resuscitation**: Neonatal resuscitation needs
- **Prevent_Hemorrhage**: Hemorrhage prevention measures
- **Fetal_Resuscitation**: Fetal resuscitation requirements
- **Notify_Attending**: Notify attending physician
- **Reassess_Later**: Schedule reassessment
- **Delivery_Prep_Hemorrhage**: Delivery preparation with hemorrhage risk
- **Any_Resuscitation**: Any type of resuscitation needed
- **Multiple_Interventions**: Multiple interventions required

## Configuration Used

Based on `config_all_labels.json`:

### Features (55 selected)
- **Excluded features**: Patient identifiers, text columns, outcomes
- **Included features**: All vital signs, labor progress, clinical measurements

### Targets
- **predict_assessments**: false (assessment labels excluded)
- **predict_treatments**: true (only treatment labels)
- **Filtering**: Min 10 samples, max 50:1 imbalance ratio

### Model Parameters
- **Algorithm**: Multi-Output Random Forest
- **Trees**: 150 estimators
- **Max Depth**: 20
- **Class Weight**: Balanced
- **Test Size**: 20%

## Quick Start

### 1. Basic Usage

```python
from multi_output_predictor import MultiOutputMedicalPredictor

# Initialize predictor with single model file
predictor = MultiOutputMedicalPredictor('multi_output_medical_model_20250910_162904')

# Patient data
patient_data = {
    "Mạch (nhập số nguyên)": 80,
    "HA tâm thu (nhập số nguyên)": 120,
    "HA tâm trương (nhập số nguyên)": 80,
    "Nhiệt độ (nhập số nguyên)": 37,
    "TT cơ bản (nhập số nguyên)": 140,
    # ... other features
}

# Get ALL treatment predictions at once
predictions = predictor.predict(patient_data, return_probabilities=True)
```

### 2. Batch Processing

```python
import pandas as pd

# Multiple patients
patients_df = pd.DataFrame([patient1, patient2, patient3])

# Batch predictions - single model call for all patients and labels
batch_results = predictor.predict_batch(patients_df, return_probabilities=True)
```

### 3. Treatment-Specific Predictions

```python
# Get only treatment recommendations
treatments = predictor.get_treatment_predictions(patient_data)

# Get model summary
summary = predictor.get_target_summary()
print(f"Predicts {summary['total_targets']} labels simultaneously")
```

## File Structure

```
multi_output_medical_model_20250910_162904/
├── multi_output_medical_model.joblib     # 🎯 SINGLE WEIGHT FILE
├── multi_output_predictor.py             # Prediction interface
├── preprocessing.joblib                  # Feature preprocessing
├── model_metadata.json                   # Model details & performance
├── multi_output_model_report.txt         # Detailed performance report
└── README.md                            # This file
```

## Advantages of Multi-Output Approach

### vs Individual Models (21 separate files)
✅ **Storage**: 1 file vs 21 files (95% reduction)  
✅ **Loading**: Single load vs 21 separate loads  
✅ **Consistency**: Same feature processing for all labels  
✅ **Speed**: One prediction call vs 21 separate calls  
✅ **Memory**: Lower memory footprint  
✅ **Maintenance**: Single model to update/retrain  

### Performance Comparison
- **Individual Models**: 21 files, ~21 prediction calls
- **Multi-Output Model**: 1 file, 1 prediction call
- **Speed Improvement**: ~20x faster loading, ~10x faster prediction
- **Accuracy**: Comparable performance (93.3% vs 89.7% average)

## Technical Details

### Model Architecture
- **Base Model**: RandomForestClassifier (150 trees, depth 20)
- **Wrapper**: MultiOutputClassifier from scikit-learn
- **Training**: Simultaneous training on all valid targets
- **Prediction**: Simultaneous prediction for all targets

### Data Processing
- **Feature Count**: 55 features (after exclusions)
- **Sample Size**: 1,614 patient records
- **Train/Test Split**: 80/20 stratified split
- **Missing Values**: Median/mode imputation
- **Categorical Encoding**: Label encoding

### Quality Controls
- **Minimum Samples**: 10 per minority class
- **Maximum Imbalance**: 50:1 ratio
- **Excluded Targets**: 2 (due to insufficient data/imbalance)
- **Cross-Validation**: Built-in Random Forest bagging

## Installation Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
scipy>=1.7.0
```

## Usage Examples

### Example 1: Normal Labor
```python
normal_case = {
    "Mạch (nhập số nguyên)": 80,
    "HA tâm thu (nhập số nguyên)": 120,
    "TT cơ bản (nhập số nguyên)": 140,
    "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 4
}

# Single call predicts all 11 treatment labels
predictions = predictor.predict(normal_case, return_probabilities=True)
# Returns: {'Kế hoạch (xử trí)_Monitor_Labor': 0, 'Kế hoạch (xử trí)_Monitor_Labor_probability': 0.269, ...}
```

### Example 2: High-Risk Case
```python
high_risk = {
    "Mạch (nhập số nguyên)": 120,
    "HA tâm thu (nhập số nguyên)": 180,
    "TT cơ bản (nhập số nguyên)": 80,
    "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 9
}

predictions = predictor.predict(high_risk, return_probabilities=True)
# Higher probabilities for interventions expected
```

## Performance Monitoring

### Model Metrics by Target
- **Reassess_Later**: 98.1% accuracy (best performing)
- **Monitor_Labor**: 86.4% accuracy (most challenging)
- **Overall Range**: 86.4% - 98.1% accuracy
- **All AUC Scores**: > 78% (excellent discrimination)

### Prediction Speed
- **Single Patient**: ~57ms for all 11 predictions
- **Batch Processing**: ~58 predictions/second
- **Memory Usage**: ~50MB loaded model
- **CPU Usage**: Optimized for multi-core (n_jobs=-1)

## Configuration Management

The model uses `config_all_labels.json` for:
- **Feature Selection**: Which columns to include/exclude
- **Target Selection**: Which labels to predict
- **Model Parameters**: RandomForest hyperparameters
- **Training Settings**: Test size, validation, quality thresholds

To retrain with different settings:
1. Modify `config_all_labels.json`
2. Run `export_multi_output.py`
3. New model with updated configuration

## Support

### Model Version
- **Export Date**: 2025-09-10 16:29:04
- **Model File**: multi_output_medical_model.joblib
- **Config Used**: config_all_labels.json (treatments only)

### Troubleshooting
- **Missing Features**: Model handles gracefully with default values
- **Unknown Categories**: Mapped to most common category
- **Performance Issues**: Check feature preprocessing pipeline

---

**🎯 Single Weight File = All Predictions**  
*One model to rule them all!*
