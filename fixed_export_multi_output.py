"""
Fixed Multi-Output Medical Prediction Model Export Script
=========================================================

This script trains a SINGLE Random Forest model that can predict ALL treatment labels
simultaneously using multi-output classification with FIXED preprocessing that properly
handles numeric vs categorical features.

Key Fix: Only applies label encoding to truly categorical columns, preserves numeric values.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import json
from pathlib import Path
import sys
import warnings
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, multilabel_confusion_matrix
from scipy import sparse

# Add src to path for imports
src_path = str(Path(__file__).parent.parent)
sys.path.insert(0, src_path)

warnings.filterwarnings('ignore')

class FixedMultiOutputMedicalExporter:
    """
    Exports a single multi-output trained model for all medical prediction labels.
    FIXED VERSION: Properly distinguishes numeric vs categorical features.
    """
    
    def __init__(self, config_path, data_path, models_output_dir):
        """
        Initialize the multi-output model exporter.
        """
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)
        self.models_dir = Path(models_output_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self.load_config()
        
        # Storage for trained model and metadata
        self.multi_output_model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.model_metadata = {}
        self.training_stats = {}
        
        # Define which columns should be treated as categorical vs numeric
        self.categorical_features = [
            'Tiền căn bệnh lý',
            'Yếu tố nguy cơ', 
            'CTG',
            'Nước ối (V: ối vỡ/Vg: Vàng)',
            'Kiểu thế',
            'Bướu HT',
            'Chồng khớp',
            'Nước tiểu',
            'Thuốc gì?',
            'Nếu 10: 10X hay 10R? (Không phải 10 xin bỏ qua)',
        ]
        
        print(f"FIXED Multi-Output Medical Model Exporter initialized")
        print(f"Will preserve numeric features and only encode categorical features")
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            raise
    
    def load_and_prepare_data(self):
        """Load the dataset and identify target columns and features based on config."""
        try:
            print(f"\n📂 Loading dataset from: {self.data_path}")
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"✅ Dataset loaded successfully. Shape: {self.df.shape}")
            
            # Identify target columns based on config
            all_target_columns = [col for col in self.df.columns 
                                if ('Nhận định và đánh giá_' in col or 'Kế hoạch (xử trí)_' in col)]
            
            print(f"📊 Found {len(all_target_columns)} potential target columns")
            
            # Apply target filters if specified in config
            excluded_targets = self.config.get('excluded_targets', [])
            included_targets = self.config.get('included_targets', [])
            
            self.target_columns = []
            for col in all_target_columns:
                # Check if this target should be included
                if excluded_targets and any(exc in col for exc in excluded_targets):
                    print(f"⏭️  Excluding target: {col}")
                    continue
                
                if included_targets and not any(inc in col for inc in included_targets):
                    print(f"⏭️  Excluding target (not in included list): {col}")
                    continue
                
                self.target_columns.append(col)
            
            print(f"🎯 Using {len(self.target_columns)} target columns")
            for target in self.target_columns[:5]:  # Show first 5
                print(f"   - {target}")
            if len(self.target_columns) > 5:
                print(f"   ... and {len(self.target_columns) - 5} more")
            
            # Remove rows where all targets are NaN
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=self.target_columns, how='all')
            rows_after_dropna = len(self.df)
            print(f"📉 Removed {initial_rows - rows_after_dropna} rows with all NaN targets")
            
            # Get feature columns (exclude targets and configured excluded features)
            excluded_features = self.config.get('excluded_features', [])
            self.feature_columns = [col for col in self.df.columns 
                                  if col not in self.target_columns 
                                  and col not in excluded_features]
            
            print(f"🔧 Using {len(self.feature_columns)} feature columns")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def preprocess_features(self, X, fit_encoders=True):
        """
        FIXED preprocessing that properly handles numeric vs categorical features.
        
        Parameters:
        X (pd.DataFrame): Input features
        fit_encoders (bool): Whether to fit new encoders or use existing ones
        
        Returns:
        pd.DataFrame: Preprocessed features
        """
        X = X.copy()
        
        print(f"🔧 Preprocessing {X.shape[0]} samples with {X.shape[1]} features")
        
        # Step 1: Handle missing values
        for col in X.columns:
            if X[col].dtype in ['object', 'string'] or col in self.categorical_features:
                # For categorical columns, fill with mode or 'Unknown'
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
            else:
                # For numerical columns, fill with median
                X[col].fillna(X[col].median(), inplace=True)
        
        # Step 2: Identify truly categorical columns
        # A column is categorical if:
        # 1. It's in our predefined categorical_features list, OR
        # 2. It has object/string dtype, OR  
        # 3. It has very few unique values relative to total samples (< 20% unique)
        
        truly_categorical_cols = []
        numeric_cols = []
        
        for col in X.columns:
            is_categorical = False
            reason = "unknown"
            
            # Check if explicitly defined as categorical
            if col in self.categorical_features:
                is_categorical = True
                reason = "predefined categorical"
            # Check if object/string type
            elif X[col].dtype in ['object', 'string']:
                is_categorical = True  
                reason = "object/string dtype"
            # Check if few unique values (but not binary 0/1)
            elif len(X[col].unique()) <= max(20, len(X) * 0.05):
                unique_vals = set(X[col].unique())
                if not (unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) == 2):
                    is_categorical = True
                    reason = f"few unique values ({len(X[col].unique())})"
            
            if is_categorical:
                truly_categorical_cols.append(col)
                if fit_encoders:
                    print(f"   🏷️  Categorical: {col} ({reason})")
            else:
                numeric_cols.append(col)
                if fit_encoders:
                    print(f"   🔢 Numeric: {col}")
        
        print(f"📊 Identified {len(truly_categorical_cols)} categorical and {len(numeric_cols)} numeric features")
        
        # Step 3: Apply label encoding ONLY to truly categorical columns
        for col in truly_categorical_cols:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle new categories during prediction
                    known_labels = set(self.label_encoders[col].classes_)
                    X[col] = X[col].astype(str)
                    unknown_mask = ~X[col].isin(known_labels)
                    if unknown_mask.any():
                        print(f"   ⚠️  Found {unknown_mask.sum()} unknown values in {col}, using default")
                        X.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                    X[col] = self.label_encoders[col].transform(X[col])
            else:
                # Use existing encoders
                if col in self.label_encoders:
                    known_labels = set(self.label_encoders[col].classes_)
                    X[col] = X[col].astype(str)
                    unknown_mask = ~X[col].isin(known_labels)
                    if unknown_mask.any():
                        print(f"   ⚠️  Found {unknown_mask.sum()} unknown values in {col}, using default")
                        X.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Step 4: Ensure numeric columns remain numeric
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isna().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        print(f"✅ Preprocessing complete. Shape: {X.shape}")
        return X
    
    def train_multi_output_model(self):
        """Train a single multi-output Random Forest model for all labels."""
        
        print(f"\n🚀 Training Multi-Output Random Forest Model")
        print("=" * 60)
        
        # Prepare features and targets
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_columns].copy()
        
        print(f"📊 Dataset overview:")
        print(f"   Features shape: {X.shape}")
        print(f"   Targets shape: {y.shape}")
        
        # Check target distributions
        print(f"\n🎯 Target distributions:")
        for i, col in enumerate(self.target_columns):
            if col in y.columns:
                pos_count = (y[col] == 1).sum()
                total_count = len(y[col].dropna())
                pos_ratio = pos_count / total_count * 100 if total_count > 0 else 0
                print(f"   {col.replace('Kế hoạch (xử trí)_', ''):<25} {pos_count:>4}/{total_count:<4} ({pos_ratio:.1f}%)")
        
        # Fill NaN values in targets (0 = no intervention needed)
        y = y.fillna(0)
        
        # Convert targets to integers
        y = y.astype(int)
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit_encoders=True)
        
        print(f"\n✅ Preprocessing complete:")
        print(f"   Features shape: {X_processed.shape}")
        print(f"   Applied encoders to {len(self.label_encoders)} categorical features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"\n📚 Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        
        # Train multi-output Random Forest
        print(f"\n🌲 Training Multi-Output Random Forest...")
        start_time = time.time()
        
        base_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.multi_output_model = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.multi_output_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        print(f"\n📈 Evaluating model...")
        y_pred_train = self.multi_output_model.predict(X_train)
        y_pred_test = self.multi_output_model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"   Training accuracy: {train_accuracy:.4f}")
        print(f"   Testing accuracy: {test_accuracy:.4f}")
        
        # Individual target performance
        print(f"\n🎯 Individual target performance:")
        for i, target in enumerate(self.target_columns):
            target_name = target.replace('Kế hoạch (xử trí)_', '').replace('Nhận định và đánh giá_', '')
            try:
                train_acc = accuracy_score(y_train.iloc[:, i], y_pred_train[:, i])
                test_acc = accuracy_score(y_test.iloc[:, i], y_pred_test[:, i])
                print(f"   {target_name:<25} Train: {train_acc:.3f}, Test: {test_acc:.3f}")
            except Exception as e:
                print(f"   {target_name:<25} Evaluation error: {str(e)[:50]}")
        
        # Store training statistics
        self.training_stats = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X_processed.shape[1],
            'n_targets': len(self.target_columns)
        }
        
        # Store model metadata
        self.model_metadata = {
            'model_type': 'MultiOutputRandomForest',
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'categorical_features': list(self.label_encoders.keys()),
            'timestamp': datetime.now().isoformat(),
            'config_used': self.config
        }
        
        return True
    
    def export_model(self):
        """Export the trained multi-output model and preprocessing components."""
        
        if self.multi_output_model is None:
            print("❌ No trained model to export. Train model first.")
            return False
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.models_dir / f"fixed_multi_output_medical_model_{timestamp}"
        export_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Exporting model to: {export_dir}")
        
        try:
            # Export main model
            model_path = export_dir / "multi_output_medical_model.joblib"
            joblib.dump(self.multi_output_model, model_path)
            print(f"✅ Multi-output model saved: {model_path.name}")
            
            # Export preprocessing components
            preprocessing_data = {
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'target_names': [col.replace('Kế hoạch (xử trí)_', '').replace('Nhận định và đánh giá_', '') 
                               for col in self.target_columns],
                'categorical_features': list(self.label_encoders.keys())
            }
            
            preprocessing_path = export_dir / "preprocessing.joblib"
            joblib.dump(preprocessing_data, preprocessing_path)
            print(f"✅ Preprocessing data saved: {preprocessing_path.name}")
            
            # Export metadata
            metadata_path = export_dir / "model_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_metadata, f, indent=2, ensure_ascii=False)
            print(f"✅ Metadata saved: {metadata_path.name}")
            
            # Export training statistics
            stats_path = export_dir / "training_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_stats, f, indent=2)
            print(f"✅ Training stats saved: {stats_path.name}")
            
            # Create a simple prediction interface file
            predictor_code = f'''"""
Fixed Multi-Output Medical Predictor
===================================

This file provides a simple interface to use the trained multi-output model.
Fixed version that properly handles numeric vs categorical features.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class FixedMultiOutputMedicalPredictor:
    """
    Fixed predictor that properly handles numeric vs categorical features.
    """
    
    def __init__(self, model_dir):
        """Load the trained model and preprocessing components."""
        self.model_dir = Path(model_dir)
        
        # Load model and preprocessing
        self.model = joblib.load(self.model_dir / "multi_output_medical_model.joblib")
        self.preprocessing = joblib.load(self.model_dir / "preprocessing.joblib")
        
        # Extract components
        self.label_encoders = self.preprocessing['label_encoders']
        self.feature_columns = self.preprocessing['feature_columns'] 
        self.target_names = self.preprocessing['target_names']
        self.categorical_features = self.preprocessing['categorical_features']
        
        print(f"✅ Fixed Multi-Output Medical Predictor loaded")
        print(f"   Model expects {{len(self.feature_columns)}} features")
        print(f"   Predicts {{len(self.target_names)}} targets")
        print(f"   Categorical features: {{len(self.categorical_features)}}")
    
    def preprocess_input(self, patient_data):
        """
        FIXED preprocessing that preserves numeric values.
        """
        # Convert to DataFrame if needed
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Add missing feature columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Select only required features
        df = df[self.feature_columns]
        
        # Handle missing values
        for col in df.columns:
            if col in self.categorical_features or df[col].dtype in ['object', 'string']:
                df[col].fillna('Unknown', inplace=True)
            else:
                df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
        
        # Apply label encoding ONLY to categorical features
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                encoder = self.label_encoders[col]
                df[col] = df[col].astype(str)
                
                # Handle unknown categories
                known_labels = set(encoder.classes_)
                unknown_mask = ~df[col].isin(known_labels)
                if unknown_mask.any():
                    df.loc[unknown_mask, col] = encoder.classes_[0]
                
                df[col] = encoder.transform(df[col])
        
        # Ensure numeric columns remain numeric
        for col in df.columns:
            if col not in self.categorical_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    df[col].fillna(0, inplace=True)
        
        return df
    
    def predict(self, patient_data):
        """Make predictions for patient data."""
        # Preprocess input
        processed_data = self.preprocess_input(patient_data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        # Format results
        results = []
        for i in range(len(processed_data)):
            result = {{}}
            for j, target_name in enumerate(self.target_names):
                pred = predictions[i][j]
                # Get probability of positive class
                if hasattr(probabilities[j][i], '__len__') and len(probabilities[j][i]) > 1:
                    prob = probabilities[j][i][1]
                else:
                    prob = 0.0
                
                result[target_name] = {{
                    'prediction': int(pred),
                    'probability': float(prob)
                }}
            results.append(result)
        
        return results[0] if len(results) == 1 else results

# Example usage
if __name__ == "__main__":
    # Load predictor
    predictor = FixedMultiOutputMedicalPredictor("{export_dir.name}")
    
    # Example prediction
    test_patient = {{
        "Năm sinh": 1990,
        "Mạch (nhập số nguyên)": 120.0,
        "HA tâm thu (nhập số nguyên)": 140.0,
        "HA tâm trương (nhập số nguyên)": 90.0,
        "TT cơ bản (nhập số nguyên)": 120.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 7.0
    }}
    
    results = predictor.predict(test_patient)
    
    print("\\n🔮 Prediction Results:")
    for target, result in results.items():
        status = "🔴" if result['prediction'] == 1 else "⚪"
        print(f"   {{status}} {{target:<25}} {{result['prediction']}} ({{result['probability']:.3f}})")
'''
            
            predictor_path = export_dir / "fixed_multi_output_predictor.py"
            with open(predictor_path, 'w', encoding='utf-8') as f:
                f.write(predictor_code)
            print(f"✅ Fixed predictor interface saved: {predictor_path.name}")
            
            print(f"\n🎉 FIXED MODEL EXPORT COMPLETE!")
            print(f"📁 Export directory: {export_dir}")
            print(f"🔧 Key improvements:")
            print(f"   - Preserves numeric feature values")
            print(f"   - Only encodes truly categorical features")
            print(f"   - Should produce meaningful predictions")
            
            return str(export_dir)
            
        except Exception as e:
            print(f"❌ Error exporting model: {str(e)}")
            return False

def main():
    """Main function to train and export the fixed multi-output model."""
    
    print("🔧 FIXED Multi-Output Medical Model Trainer")
    print("=" * 60)
    print("This version properly handles numeric vs categorical features!")
    
    # Configuration
    config_path = "config_all_labels.json"
    data_path = "data/dataset_long_format_normalized_labeled.csv" 
    output_dir = "models"
    
    # Initialize exporter
    exporter = FixedMultiOutputMedicalExporter(config_path, data_path, output_dir)
    
    # Load and prepare data
    if not exporter.load_and_prepare_data():
        print("❌ Failed to load data")
        return
    
    # Train model
    if not exporter.train_multi_output_model():
        print("❌ Failed to train model")
        return
    
    # Export model
    export_path = exporter.export_model()
    if export_path:
        print(f"\n🎉 SUCCESS! Fixed model exported to: {export_path}")
        print("\n🧪 Test the fixed model:")
        print(f"   cd {export_path}")
        print(f"   python fixed_multi_output_predictor.py")
    else:
        print("❌ Failed to export model")

if __name__ == "__main__":
    main()