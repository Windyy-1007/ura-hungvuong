"""
TRULY FIXED Multi-Output Medical Model
======================================

This version properly converts string-numeric columns to actual numeric types
during both training and prediction, fixing the root cause of all-zero predictions.
"""

import pandas as pd
import numpy as np
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

class TrulyFixedMultiOutputMedicalExporter:
    """
    TRULY FIXED version that converts string-numeric columns to proper numeric types.
    """
    
    def __init__(self, config_path, data_path, models_output_dir):
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
        
        # Define which columns should DEFINITELY be treated as numeric
        # These contain vital signs and measurements that must remain numeric
        self.force_numeric_columns = [
            'Năm sinh',  # Birth year
            'Mạch (nhập số nguyên)',  # Pulse rate
            'HA tâm thu (nhập số nguyên)',  # Systolic BP
            'HA tâm trương (nhập số nguyên)',  # Diastolic BP
            'Nhiệt độ (nhập số nguyên)',  # Temperature
            'TT cơ bản (nhập số nguyên)',  # Fetal heart rate
            'Đánh giá mức độ đau (VAS) (Điền số nguyên)',  # Pain score
            'Các cơn co TC/10 phút (điền số nguyên)',  # Contractions per 10 min
            'Thời gian của các cơn co TC (điền số nguyên)',  # Contraction duration
            'Para (điền 4 số)',  # Parity
            'Khởi phát chuyển dạ (1: Có, 0: Không)',  # Labor onset
            'Bạn đồng hành (1: Có, 0: Không)',  # Support person
            'Nước uống vào (1: Có, 0: Không)',  # Fluid intake
            'Ăn',  # Food intake
            'Thuốc',  # Medication
            'Truyền dịch',  # IV fluids
            'hour',  # Hour
        ]
        
        # Define truly categorical columns
        self.categorical_features = [
            'Dấu thời gian',
            'Họ và tên',
            'Tiền căn bệnh lý',
            'Chẩn đoán chuyển dạ hoạt động',
            'Ngày ối vỡ',
            'Giờ ối vỡ',
            'Yếu tố nguy cơ',
            'Ngày chuyển dạ hoạt động (xx/yy/zzzz)',
            'Ngày',
            'Giờ',
            'Nước tiểu',
            'CTG',
            'Nước ối (V: ối vỡ/Vg: Vàng)',
            'Kiểu thế',
            'Bướu HT',
            'Chồng khớp',
            'Nếu 10: 10X hay 10R? (Không phải 10 xin bỏ qua)',
            'Oxytocin (số hoặc số la mã)',
            'Thuốc gì?',
            'Nhận định và đánh giá',
            'Kế hoạch (xử trí)',
            'Sanh',
            'Giờ sanh (chỉ điền khi không phải sanh mổ)',
            'Lý do mổ (nếu có):',
            'Giới tính em bé',
            'Apgar (nhập phân số số nguyên X/Y) (ví dụ: 1/5)',
            'Cân nặng',
            'Cổ TC (KH: X)',
            'Độ lọt (KH: O)',
        ]
        
        print("🔧 TRULY FIXED Multi-Output Medical Model Exporter")
        print("   - Forces numeric conversion for vital signs")
        print("   - Only encodes truly categorical features")
        print("   - Should produce meaningful predictions")
    
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
    
    def force_numeric_conversion(self, df):
        """
        Convert columns that should be numeric but are stored as strings/objects.
        """
        print(f"🔢 Converting string-numeric columns to proper numeric types...")
        
        for col in self.force_numeric_columns:
            if col in df.columns:
                original_dtype = df[col].dtype
                
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Count how many values were converted
                if original_dtype == 'object':
                    non_null_count = df[col].notna().sum()
                    print(f"   ✅ {col}: {original_dtype} -> numeric ({non_null_count} valid values)")
                
        return df
    
    def load_and_prepare_data(self):
        """Load dataset and properly convert numeric columns."""
        try:
            print(f"\n📂 Loading dataset from: {self.data_path}")
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"✅ Dataset loaded successfully. Shape: {self.df.shape}")
            
            # CRITICAL: Convert string-numeric columns to proper numeric types
            self.df = self.force_numeric_conversion(self.df)
            
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
                    continue
                
                if included_targets and not any(inc in col for inc in included_targets):
                    continue
                
                self.target_columns.append(col)
            
            print(f"🎯 Using {len(self.target_columns)} target columns")
            
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
        TRULY FIXED preprocessing that preserves numeric values.
        """
        X = X.copy()
        
        print(f"🔧 Preprocessing {X.shape[0]} samples with {X.shape[1]} features")
        
        # Step 1: Force numeric conversion
        X = self.force_numeric_conversion(X)
        
        # Step 2: Handle missing values
        for col in X.columns:
            if col in self.categorical_features or X[col].dtype in ['object', 'string']:
                # For categorical columns, fill with mode or 'Unknown'
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                X[col] = X[col].fillna(mode_val)
            else:
                # For numerical columns, fill with median
                median_val = X[col].median() if not X[col].isna().all() else 0
                X[col] = X[col].fillna(median_val)
        
        # Step 3: Separate truly categorical vs numeric columns
        truly_categorical_cols = []
        numeric_cols = []
        
        for col in X.columns:
            if col in self.categorical_features:
                truly_categorical_cols.append(col)
                if fit_encoders:
                    print(f"   🏷️  Categorical: {col} (predefined)")
            elif col in self.force_numeric_columns:
                numeric_cols.append(col)
                if fit_encoders:
                    print(f"   🔢 Numeric: {col} (forced numeric)")
            elif X[col].dtype in ['object', 'string']:
                truly_categorical_cols.append(col)
                if fit_encoders:
                    print(f"   🏷️  Categorical: {col} (string dtype)")
            else:
                # Check if it has very few unique values (but not binary 0/1)
                unique_vals = set(X[col].unique())
                n_unique = len(unique_vals)
                is_binary = unique_vals.issubset({0, 1, 0.0, 1.0}) and n_unique == 2
                
                if n_unique <= 10 and not is_binary:
                    truly_categorical_cols.append(col)
                    if fit_encoders:
                        print(f"   🏷️  Categorical: {col} (few unique values: {n_unique})")
                else:
                    numeric_cols.append(col)
                    if fit_encoders:
                        print(f"   🔢 Numeric: {col} (many unique values: {n_unique})")
        
        print(f"📊 Final classification: {len(truly_categorical_cols)} categorical, {len(numeric_cols)} numeric")
        
        # Step 4: Apply label encoding ONLY to truly categorical columns
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
        
        # Step 5: Ensure numeric columns remain properly numeric
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isna().any():
                median_val = X[col].median() if not X[col].isna().all() else 0
                X[col] = X[col].fillna(median_val)
        
        print(f"✅ Preprocessing complete. Shape: {X.shape}")
        print(f"   Categorical encoders: {len(self.label_encoders)}")
        print(f"   Numeric columns preserved: {len(numeric_cols)}")
        
        return X
    
    def train_multi_output_model(self):
        """Train the truly fixed multi-output model."""
        
        print(f"\n🚀 Training TRULY FIXED Multi-Output Random Forest Model")
        print("=" * 60)
        
        # Prepare features and targets
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_columns].copy()
        
        print(f"📊 Dataset overview:")
        print(f"   Features shape: {X.shape}")
        print(f"   Targets shape: {y.shape}")
        
        # Check target distributions
        print(f"\n🎯 Target distributions (top 10):")
        for i, col in enumerate(self.target_columns[:10]):
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
            'model_type': 'TrulyFixedMultiOutputRandomForest',
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'categorical_features': list(self.label_encoders.keys()),
            'numeric_features': self.force_numeric_columns,
            'timestamp': datetime.now().isoformat(),
            'config_used': self.config
        }
        
        return True
    
    def export_model(self):
        """Export the truly fixed model."""
        
        if self.multi_output_model is None:
            print("❌ No trained model to export. Train model first.")
            return False
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.models_dir / f"truly_fixed_multi_output_model_{timestamp}"
        export_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Exporting TRULY FIXED model to: {export_dir}")
        
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
                'categorical_features': list(self.label_encoders.keys()),
                'force_numeric_columns': self.force_numeric_columns
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
            
            print(f"\n🎉 TRULY FIXED MODEL EXPORT COMPLETE!")
            print(f"📁 Export directory: {export_dir}")
            print(f"🔧 Key fixes:")
            print(f"   - Converts string-numeric columns to proper numeric types")
            print(f"   - Preserves vital signs as numeric values")
            print(f"   - Should produce meaningful positive predictions")
            
            return str(export_dir)
            
        except Exception as e:
            print(f"❌ Error exporting model: {str(e)}")
            return False

def main():
    """Main function to train and export the truly fixed model."""
    
    print("🔧 TRULY FIXED Multi-Output Medical Model Trainer")
    print("=" * 60)
    print("This version converts string-numeric columns to proper numeric types!")
    
    # Configuration
    config_path = "config_all_labels.json"
    data_path = "data/dataset_long_format_normalized_labeled.csv" 
    output_dir = "models"
    
    # Initialize exporter
    exporter = TrulyFixedMultiOutputMedicalExporter(config_path, data_path, output_dir)
    
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
        print(f"\n🎉 SUCCESS! Truly fixed model exported to: {export_path}")
        print("\n🧪 Now test with truly fixed predictor!")
    else:
        print("❌ Failed to export model")

if __name__ == "__main__":
    main()