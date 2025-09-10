# URA Medical Prediction System

## 🎯 Goal of the Repository

This repository contains a **machine learning system for medical treatment prediction** based on patient vital signs and clinical measurements. The system is designed to assist healthcare professionals by predicting appropriate treatment recommendations for obstetric and neonatal care.

### Key Objectives:
- **Predict treatment recommendations** (`Kế hoạch xử trí`) based on patient clinical data
- **Support clinical decision-making** with high-accuracy ML predictions
- **Provide real-time predictions** for patient care scenarios
- **Ensure production-ready performance** (98.3% accuracy achieved)

## 🏗️ Repository Structure

```
ura-hungvuong/
├── main.py                           # 🚀 Main entry point for all operations
├── quick_evaluate.py                 # ⚡ Quick model evaluation
├── config_all_labels.json           # ⚙️ Configuration file
├── EVALUATION_SUMMARY.md            # 📊 Model performance summary
├── README.md                        # 📖 This file
│
├── data/                            # 📁 Dataset and results
│   ├── dataset_long_format_normalized_labeled.csv  # Main dataset
│   ├── evaluation_report_*.csv      # Evaluation results
│   └── *_predictions.csv           # Prediction outputs
│
├── models/                          # 🤖 Trained models
│   └── multi_output_medical_model_*/ # Latest trained model
│       ├── multi_output_medical_model.joblib  # 🎯 Main weight file
│       ├── multi_output_predictor.py          # Prediction interface
│       ├── preprocessing.joblib               # Feature preprocessing
│       ├── model_metadata.json               # Model details
│       └── README.md                          # Model documentation
│
└── src/                             # 💻 Source code
    ├── classifiers/                 # ML models and training
    │   ├── export_multi_output.py   # 🏭 Model training & export
    │   ├── evaluate.py              # 📊 Model evaluation
    │   └── models/                  # Model implementations
    │       ├── random_forest.py     # Random Forest classifier
    │       └── batch_training.py    # Batch training utilities
    │
    └── dataloader/                  # 📥 Data processing
        ├── loader.py                # Data loading utilities
        ├── preprocess.py            # Data preprocessing
        └── ults/                    # Data utilities
            ├── labeler.py           # Label processing
            ├── normalize.py         # Data normalization
            └── getHeaders.py        # Column management
```

## 🎯 Model Configuration

The system is configured to predict **treatment labels only** (`Kế hoạch xử trí`) while excluding specific features:

### ✅ Target Predictions (Treatment Labels)
- Monitor_Labor - Continue labor monitoring
- Report_Doctor - Report to attending physician  
- Prepare_Delivery - Prepare for delivery
- Neonatal_Resuscitation - Neonatal resuscitation needs
- Prevent_Hemorrhage - Hemorrhage prevention measures
- Fetal_Resuscitation - Fetal resuscitation requirements
- Notify_Attending - Notify attending physician
- Reassess_Later - Schedule reassessment
- Delivery_Prep_Hemorrhage - Delivery preparation with hemorrhage risk
- Any_Resuscitation - Any type of resuscitation needed
- Multiple_Interventions - Multiple interventions required

### ❌ Excluded Features
```json
"exclude_features": [
  "Dấu thời gian",           # Timestamp
  "Mã bệnh nhân",            # Patient ID
  "Họ và tên",               # Patient name
  "Nhận định và đánh giá",   # Assessment text
  "Kế hoạch (xử trí)",       # Treatment text
  "Sanh",                    # Birth outcomes
  "Giờ sanh (chỉ điền khi không phải sanh mổ)",  # Birth time
  "Lý do mổ (nếu có):",      # Surgery reason
  "Giới tính em bé",         # Baby gender
  "Apgar (nhập phân số số nguyên X/Y) (ví dụ: 1/5)",  # Apgar score
  "Cân nặng"                 # Birth weight
]
```

## 🚀 Quick Start Commands

### 1. System Status Check
```bash
python main.py
```
*Shows system status, available models, and required files*

### 2. Train New Model
```bash
python main.py --train
```
*Trains a new multi-output model using config_all_labels.json*

### 3. Quick Model Evaluation  
```bash
python main.py --quick-eval
# OR
python quick_evaluate.py
```
*Quick performance summary with top/bottom performing labels*

### 4. Detailed Model Evaluation
```bash
python main.py --evaluate
```
*Comprehensive evaluation with detailed reports saved to data/*

### 5. Make Predictions on New Data
```bash
python main.py --predict patient_data.csv
```
*Generates predictions for new patient data*

### 6. Custom Configuration
```bash
python main.py --train --config custom_config.json
python main.py --evaluate --model models/specific_model --config custom_config.json
```

## 📊 Model Performance

Current model achieves **production-ready performance**:

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **98.3%** |
| **Overall Precision** | **98.4%** |
| **Overall Recall** | **98.3%** |
| **Overall F1-Score** | **98.3%** |
| **Overall AUC** | **99.4%** |

### 🏆 Top Performing Labels
1. **Notify_Attending** - 100.0% accuracy
2. **Fetal_Resuscitation** - 99.7% accuracy  
3. **Reassess_Later** - 99.7% accuracy
4. **Delivery_Prep_Hemorrhage** - 99.7% accuracy
5. **Prevent_Hemorrhage** - 98.5% accuracy

## 🛠️ Development Workflow

### Training a New Model
1. **Prepare data**: Ensure `data/dataset_long_format_normalized_labeled.csv` exists
2. **Configure**: Modify `config_all_labels.json` if needed
3. **Train**: Run `python main.py --train`
4. **Evaluate**: Run `python main.py --evaluate`
5. **Test**: Use `python main.py --predict test_data.csv`

### Model Evaluation Workflow
1. **Quick check**: `python quick_evaluate.py`
2. **Detailed analysis**: `python main.py --evaluate`  
3. **Review reports**: Check `data/evaluation_report_*.csv`
4. **Performance analysis**: See `EVALUATION_SUMMARY.md`

### Making Predictions
1. **Prepare CSV**: Patient data with same columns as training data
2. **Predict**: `python main.py --predict new_patients.csv`
3. **Review results**: Check `new_patients_predictions.csv`

## 📁 Key Files Description

### Configuration Files
- **`config_all_labels.json`** - Main configuration controlling features, targets, and model parameters

### Training & Evaluation
- **`main.py`** - Central command interface for all operations
- **`src/classifiers/export_multi_output.py`** - Multi-output model training and export
- **`src/classifiers/evaluate.py`** - Comprehensive model evaluation system
- **`quick_evaluate.py`** - Fast evaluation summary

### Data Processing
- **`src/dataloader/loader.py`** - Data loading utilities
- **`src/dataloader/preprocess.py`** - Data preprocessing functions  
- **`src/dataloader/ults/`** - Data transformation utilities

### Model Artifacts
- **`models/multi_output_medical_model_*/`** - Complete trained model package
- **`multi_output_medical_model.joblib`** - Main weight file (single file for all predictions)
- **`multi_output_predictor.py`** - Prediction interface
- **`preprocessing.joblib`** - Feature preprocessing pipeline

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Multi-Output Random Forest (150 trees, depth 20)
- **Features**: 55 clinical features (after exclusions)
- **Targets**: 11 treatment labels (simultaneous prediction)
- **Training**: 1,614 patient records, 80/20 train/test split

### Data Pipeline
1. **Loading**: CSV data with 79 columns
2. **Feature Selection**: Apply exclusion rules from config
3. **Target Filtering**: Only treatment labels (`Kế hoạch xử trí`)
4. **Preprocessing**: Missing value imputation, categorical encoding
5. **Quality Control**: Minimum sample and imbalance thresholds
6. **Training**: Multi-output Random Forest with balanced classes

### Performance Characteristics
- **Single Weight File**: One model predicts all 11 labels
- **Fast Prediction**: ~57ms for all labels per patient
- **Memory Efficient**: ~50MB loaded model
- **High Accuracy**: 98.3% average across all labels
- **Production Ready**: Suitable for clinical decision support

## 📈 Usage Examples

### Example 1: Train and Evaluate
```bash
# Train new model
python main.py --train

# Quick evaluation
python main.py --quick-eval

# Detailed evaluation with reports
python main.py --evaluate
```

### Example 2: Make Predictions
```bash
# Predict on new data
python main.py --predict new_patients.csv

# Results saved to new_patients_predictions.csv
```

### Example 3: Custom Workflow
```bash
# Check system status
python main.py

# Train with specific config
python main.py --train --config custom_config.json

# Evaluate specific model
python main.py --evaluate --model models/specific_model
```

## 📚 Additional Resources

- **`EVALUATION_SUMMARY.md`** - Detailed model performance analysis
- **`models/*/README.md`** - Model-specific documentation  
- **`data/evaluation_report_*.txt`** - Detailed evaluation reports
- **Configuration reference** - See `config_all_labels.json` for all options

## 🎯 Key Features

✅ **Single Weight File** - One model file predicts all treatment labels  
✅ **Configuration-Driven** - Easy to modify features and targets  
✅ **Production Ready** - 98.3% accuracy suitable for clinical use  
✅ **Fast Predictions** - Real-time prediction capabilities  
✅ **Comprehensive Evaluation** - Detailed performance metrics  
✅ **Easy to Use** - Simple command-line interface  
✅ **Well Documented** - Complete documentation and examples  

## 🚀 Getting Started

1. **Clone/setup repository**
2. **Check system status**: `python main.py`
3. **Train your first model**: `python main.py --train`
4. **Evaluate performance**: `python main.py --quick-eval`
5. **Make predictions**: `python main.py --predict your_data.csv`

The system is designed to be production-ready for medical treatment prediction with minimal setup required!
