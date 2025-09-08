# Medical Prediction Models - Export Package

## Overview

This package contains trained Random Forest models for predicting medical assessments and treatment recommendations in obstetric care. The models were trained on a comprehensive dataset of medical records and can predict **21 different medical labels** with high accuracy.

## Package Contents

### Trained Models (21 files)
- **Assessment Models (10)**: Predict medical assessments and patient conditions
- **Treatment Models (11)**: Recommend appropriate treatment plans and interventions

### Key Files
- `medical_predictor.py` - **Main prediction interface**
- `models_metadata.json` - Comprehensive model metadata
- `models_summary_report.txt` - Detailed performance report
- `preprocessing.joblib` - Data preprocessing components
- `test_predictions.py` - Example usage script

## Model Performance

### Overall Statistics
- **Average Test Accuracy**: 89.71% ± 7.12%
- **Average AUC Score**: 89.22% ± 4.91%
- **Best Model Accuracy**: 98.14% (Reassess_Later)
- **Success Rate**: 21/32 labels (65.6%)

### Top Performing Models
1. **Reassess_Later** - 98.14% accuracy, 99.01% AUC
2. **Contractions_Complete** - 97.83% accuracy, 87.59% AUC
3. **CTG_II_Guidance** - 97.83% accuracy, 90.22% AUC
4. **Patient_Pain** - 97.21% accuracy, 91.23% AUC
5. **Position_Unfavorable** - 96.59% accuracy, 97.98% AUC

## Quick Start

### 1. Basic Usage

```python
from medical_predictor import MedicalPredictor

# Initialize predictor
predictor = MedicalPredictor('path/to/medical_models_20250908_210133')

# Sample patient data
patient_data = {
    "Mạch (nhập số nguyên)": 80,
    "HA tâm thu (nhập số nguyên)": 120,
    "HA tâm trương (nhập số nguyên)": 80,
    "Nhiệt độ (nhập số nguyên)": 37,
    "TT cơ bản (nhập số nguyên)": 140,
    "Các cơn co TC/10 phút (điền số nguyên)": 3,
    "Thời gian của các cơn co TC (điền số nguyên)": 40,
    "Cổ TC (KH: X)": 6,
    "Độ lọt (KH: O)": -2,
    "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 5
}

# Make predictions
predictions = predictor.predict(patient_data, return_probabilities=True)
```

### 2. Batch Processing

```python
import pandas as pd

# Load multiple patients
patients_df = pd.DataFrame([patient_data1, patient_data2, ...])

# Batch predictions
batch_results = predictor.predict_batch(patients_df, return_probabilities=True)
```

### 3. Specialized Predictions

```python
# Get only assessment predictions
assessments = predictor.get_assessment_predictions(patient_data)

# Get only treatment recommendations
treatments = predictor.get_treatment_predictions(patient_data)
```

## Available Predictions

### Assessment Labels (10 models)
- **CTG_Group_I**: Cardiotocography Group I classification
- **CTG_Group_II**: Cardiotocography Group II classification
- **Guidance_Push**: Guidance for pushing during delivery
- **Patient_Stable**: Patient stability assessment
- **Position_Unfavorable**: Unfavorable fetal position
- **Contractions_Complete**: Contraction completion status
- **Patient_Pain**: Patient pain assessment
- **CTG_II_Position_Unfavorable**: CTG Group II with unfavorable position
- **CTG_II_Guidance**: CTG Group II guidance requirements
- **CTG_Abnormal**: Abnormal CTG patterns

### Treatment Labels (11 models)
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

## Input Features

The models expect the following key input features:

### Vital Signs
- Heart rate (Mạch)
- Blood pressure systolic/diastolic (HA tâm thu/tâm trương)
- Temperature (Nhiệt độ)

### Labor Progress
- Fetal heart rate baseline (TT cơ bản)
- Contractions per 10 minutes (Các cơn co TC/10 phút)
- Contraction duration (Thời gian của các cơn co TC)
- Cervical dilation (Cổ TC)
- Fetal station (Độ lọt)

### Patient Assessment
- Pain score (VAS scale)
- CTG patterns
- Various clinical indicators

## Model Details

### Training Parameters
- **Algorithm**: Random Forest Classifier
- **Number of trees**: 150
- **Max depth**: 20
- **Class weighting**: Balanced
- **Cross-validation**: 5-fold
- **Train/test split**: 80/20

### Data Preprocessing
- **Missing value handling**: Median imputation for numerical, mode for categorical
- **Categorical encoding**: Label encoding for string variables
- **Feature selection**: 45 features used across all models
- **Sample size**: 1,614 patient records

### Quality Controls
- **Minimum minority class samples**: 10
- **Maximum class imbalance**: 50:1
- **Stratified sampling**: Maintains class distribution in train/test splits

## File Structure

```
medical_models_20250908_210133/
├── medical_predictor.py              # Main prediction interface
├── models_metadata.json              # Model metadata and performance metrics
├── models_summary_report.txt         # Detailed performance report
├── preprocessing.joblib               # Preprocessing components
├── test_predictions.py               # Example usage script
└── rf_model_*.joblib                 # Individual trained models (21 files)
```

## Installation Requirements

```python
# Required packages
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

## Performance Notes

### High-Performing Models (>95% accuracy)
- Reassess_Later (98.14%)
- Contractions_Complete (97.83%)
- CTG_II_Guidance (97.83%)
- Patient_Pain (97.21%)
- Position_Unfavorable (96.59%)

### Models with Class Imbalance Challenges
Some labels were excluded due to extreme class imbalance or insufficient samples:
- CTG_Group_III (only one class present)
- Multipara (83.9:1 imbalance)
- Various rare conditions with <10 samples

## Usage Examples

### Example 1: Normal Labor Case
```python
normal_case = {
    "Mạch (nhập số nguyên)": 80,
    "HA tâm thu (nhập số nguyên)": 120,
    "HA tâm trương (nhập số nguyên)": 80,
    "TT cơ bản (nhập số nguyên)": 140,
    "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 4
}

predictions = predictor.predict(normal_case)
# Expected: Low intervention requirements
```

### Example 2: High-Risk Case
```python
high_risk_case = {
    "Mạch (nhập số nguyên)": 110,
    "HA tâm thu (nhập số nguyên)": 160,
    "HA tâm trương (nhập số nguyên)": 100,
    "TT cơ bản (nhập số nguyên)": 100,
    "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 8
}

predictions = predictor.predict(high_risk_case, return_probabilities=True)
# Expected: Higher intervention probabilities
```

## Support and Maintenance

### Model Versioning
- **Export Date**: 2025-09-08 21:01:33
- **Model Version**: medical_models_20250908_210133
- **Data Version**: dataset_long_format_normalized_labeled.csv

### Performance Monitoring
- Regular validation recommended on new data
- Monitor prediction distributions for dataset drift
- Retrain models if performance degrades

### Error Handling
- Models handle missing features gracefully
- Unknown categorical values mapped to known categories
- Comprehensive error logging in prediction interface

## Technical Notes

### Memory Requirements
- Total package size: ~50MB
- RAM usage during prediction: ~100MB
- Loading time: ~2-3 seconds

### Prediction Speed
- Single prediction: <10ms
- Batch of 100 patients: <100ms
- Suitable for real-time clinical applications

### Thread Safety
- Models are thread-safe for read operations
- Can be used in multi-threaded web applications
- No state maintained between predictions

## Contact and Attribution

This model package was generated using the URA (University Research Assistant) medical prediction framework. For questions or issues, please refer to the project documentation or contact the development team.

---

*Generated on: September 8, 2025*  
*Package Version: medical_models_20250908_210133*  
*Total Models: 21 trained Random Forest classifiers*
