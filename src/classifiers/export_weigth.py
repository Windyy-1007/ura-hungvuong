"""
Multi-Label Medical Prediction Model Export Script
=================================================

This script trains Random Forest models for ALL available medical assessment and treatment labels,
then exports them as a unified prediction system to the /models directory.

Features:
- Trains models for all available target labels
- Saves trained models with preprocessing components
- Creates a unified prediction interface
- Exports comprehensive model package for production use
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Add src to path for imports
src_path = str(Path(__file__).parent.parent)
sys.path.insert(0, src_path)

try:
    from classifiers.models.random_forest import MedicalRandomForestClassifier
except ImportError:
    # Alternative import path
    sys.path.insert(0, str(Path(__file__).parent))
    from models.random_forest import MedicalRandomForestClassifier

warnings.filterwarnings('ignore')

class MultiLabelModelExporter:
    """
    Exports trained models for all medical prediction labels.
    Creates a unified prediction system that can predict all labels simultaneously.
    """
    
    def __init__(self, data_path, models_output_dir):
        """
        Initialize the multi-label model exporter.
        
        Parameters:
        data_path (str): Path to the labeled dataset CSV file
        models_output_dir (str): Directory to save exported models
        """
        self.data_path = Path(data_path)
        self.models_dir = Path(models_output_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Storage for trained models and metadata
        self.trained_models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.model_metadata = {}
        self.training_stats = {}
        
        print(f"Multi-Label Model Exporter initialized")
        print(f"Data source: {self.data_path}")
        print(f"Models output: {self.models_dir}")
    
    def load_and_prepare_data(self):
        """
        Load the dataset and identify all target columns and features.
        
        Returns:
        bool: True if data loaded successfully, False otherwise
        """
        try:
            print(f"\n📂 Loading dataset from: {self.data_path}")
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"✅ Dataset loaded successfully. Shape: {self.df.shape}")
            
            # Identify target columns (binary label columns)
            self.target_columns = [col for col in self.df.columns 
                                 if ('Nhận định và đánh giá_' in col or 'Kế hoạch (xử trí)_' in col)]
            
            print(f"🎯 Found {len(self.target_columns)} target labels:")
            
            # Categorize targets
            assessment_targets = [col for col in self.target_columns if 'Nhận định và đánh giá_' in col]
            treatment_targets = [col for col in self.target_columns if 'Kế hoạch (xử trí)_' in col]
            
            print(f"   📋 Assessment labels: {len(assessment_targets)}")
            print(f"   🏥 Treatment labels: {len(treatment_targets)}")
            
            # Identify feature columns (exclude original text columns and target columns)
            text_columns = ['Nhận định và đánh giá', 'Kế hoạch (xử trí)']
            self.feature_columns = [col for col in self.df.columns 
                                  if col not in text_columns and col not in self.target_columns]
            
            print(f"🔢 Available feature columns: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def preprocess_features(self, X, fit_encoders=True):
        """
        Preprocess features including handling missing values and encoding.
        
        Parameters:
        X (pd.DataFrame): Input features
        fit_encoders (bool): Whether to fit new encoders or use existing ones
        
        Returns:
        pd.DataFrame: Preprocessed features
        """
        X = X.copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['object', 'string']:
                # For categorical columns, fill with mode or 'Unknown'
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
            else:
                # For numerical columns, fill with median
                X[col].fillna(X[col].median(), inplace=True)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle new categories during prediction
                    known_labels = set(self.label_encoders[col].classes_)
                    X[col] = X[col].astype(str)
                    # Map unknown labels to a default value
                    unknown_mask = ~X[col].isin(known_labels)
                    if unknown_mask.any():
                        X.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                    X[col] = self.label_encoders[col].transform(X[col])
            else:
                # Use existing encoders
                if col in self.label_encoders:
                    known_labels = set(self.label_encoders[col].classes_)
                    X[col] = X[col].astype(str)
                    # Map unknown labels to a default value
                    unknown_mask = ~X[col].isin(known_labels)
                    if unknown_mask.any():
                        X.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                    X[col] = self.label_encoders[col].transform(X[col])
                else:
                    print(f"Warning: No encoder found for column {col}")
        
        return X
    
    def train_model_for_target(self, target_column, min_samples=10, max_imbalance=50):
        """
        Train a Random Forest model for a specific target column.
        
        Parameters:
        target_column (str): The target column to predict
        min_samples (int): Minimum samples required for minority class
        max_imbalance (int): Maximum imbalance ratio allowed
        
        Returns:
        dict: Training results or None if training failed
        """
        try:
            print(f"\n🎯 Training model for: {target_column}")
            
            # Prepare features and target
            X = self.df[self.feature_columns].copy()
            y = self.df[target_column].copy()
            
            # Check class distribution
            class_dist = y.value_counts()
            minority_class_count = class_dist.min()
            majority_class_count = class_dist.max()
            imbalance_ratio = majority_class_count / minority_class_count if minority_class_count > 0 else float('inf')
            
            print(f"   📊 Class distribution: {class_dist.to_dict()}")
            print(f"   ⚖️ Imbalance ratio: {imbalance_ratio:.1f}:1")
            
            # Check if we have enough samples and reasonable balance
            if minority_class_count < min_samples:
                print(f"   ⚠️ SKIPPED: Minority class has only {minority_class_count} samples (min: {min_samples})")
                return None
            
            if imbalance_ratio > max_imbalance:
                print(f"   ⚠️ SKIPPED: Class imbalance too high ({imbalance_ratio:.1f}:1, max: {max_imbalance}:1)")
                return None
            
            if len(class_dist) < 2:
                print(f"   ⚠️ SKIPPED: Only one class present")
                return None
            
            # Remove rows with missing targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Preprocess features
            X_processed = self.preprocess_features(X, fit_encoders=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   📈 Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Train Random Forest model
            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            start_time = time.time()
            rf_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1] if len(rf_model.classes_) == 2 else None
            
            test_accuracy = accuracy_score(y_test, y_pred)
            train_accuracy = rf_model.score(X_train, y_train)
            
            # Calculate AUC if possible
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            except:
                auc_score = None
            
            # Get classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance
            feature_importance = dict(zip(X_processed.columns, rf_model.feature_importances_))
            
            # Store model and metadata
            model_info = {
                'model': rf_model,
                'target_column': target_column,
                'feature_columns': list(X_processed.columns),
                'classes': [int(cls) if isinstance(cls, np.integer) else cls for cls in rf_model.classes_],
                'training_time': float(training_time),
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'auc_score': float(auc_score) if auc_score else None,
                'class_distribution': {str(k): int(v) for k, v in class_dist.to_dict().items()},
                'imbalance_ratio': float(imbalance_ratio),
                'classification_report': class_report,
                'feature_importance': {k: float(v) for k, v in feature_importance.items()},
                'n_features': int(len(X_processed.columns)),
                'n_samples': int(len(X_processed))
            }
            
            self.trained_models[target_column] = model_info
            
            print(f"   ✅ SUCCESS! Training completed in {training_time:.1f}s")
            print(f"   📈 Train Accuracy: {train_accuracy:.4f}")
            print(f"   📊 Test Accuracy: {test_accuracy:.4f}")
            if auc_score:
                print(f"   🎯 AUC Score: {auc_score:.4f}")
            
            return model_info
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            return None
    
    def train_all_models(self, min_samples=10, max_imbalance=50):
        """
        Train models for all available target columns.
        
        Parameters:
        min_samples (int): Minimum samples required for minority class
        max_imbalance (int): Maximum imbalance ratio allowed
        
        Returns:
        dict: Summary of training results
        """
        print(f"\n🚀 Starting training for ALL {len(self.target_columns)} target labels...")
        print(f"⚙️ Training parameters:")
        print(f"   - Minimum minority class samples: {min_samples}")
        print(f"   - Maximum imbalance ratio: {max_imbalance}:1")
        print(f"   - Random Forest: 150 trees, balanced weights")
        
        start_time = time.time()
        successful_models = 0
        failed_models = []
        
        # Train models for each target
        for i, target in enumerate(self.target_columns, 1):
            print(f"\n[{i}/{len(self.target_columns)}] Processing: {target}")
            
            result = self.train_model_for_target(target, min_samples, max_imbalance)
            
            if result:
                successful_models += 1
            else:
                failed_models.append(target)
        
        total_time = time.time() - start_time
        
        # Training summary
        print(f"\n{'='*80}")
        print(f"🎉 TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"✅ Successfully trained: {successful_models}/{len(self.target_columns)} models")
        print(f"❌ Failed/Skipped: {len(failed_models)} models")
        print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
        
        if successful_models > 0:
            # Calculate average performance
            accuracies = [model['test_accuracy'] for model in self.trained_models.values()]
            auc_scores = [model['auc_score'] for model in self.trained_models.values() if model['auc_score']]
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Average Test Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            if auc_scores:
                print(f"   Average AUC Score: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            
            # Top performing models
            sorted_models = sorted(self.trained_models.items(), 
                                 key=lambda x: x[1]['test_accuracy'], reverse=True)
            
            print(f"\n🏆 TOP 10 PERFORMING MODELS:")
            for i, (target, model) in enumerate(sorted_models[:10], 1):
                auc_str = f"AUC: {model['auc_score']:.4f}" if model['auc_score'] else "AUC: N/A"
                print(f"   {i:2d}. {model['test_accuracy']:.4f} | {auc_str} | {target}")
        
        if failed_models:
            print(f"\n⚠️ FAILED/SKIPPED MODELS ({len(failed_models)}):")
            for target in failed_models[:10]:  # Show first 10
                short_name = target.split('_', 1)[1] if '_' in target else target
                print(f"   ❌ {short_name}")
            if len(failed_models) > 10:
                print(f"   ... and {len(failed_models) - 10} more")
        
        return {
            'successful_models': successful_models,
            'failed_models': len(failed_models),
            'total_targets': len(self.target_columns),
            'total_time': total_time,
            'training_stats': self.trained_models
        }
    
    def save_models(self):
        """
        Save all trained models and create a unified prediction interface.
        """
        if not self.trained_models:
            print("❌ No trained models to save!")
            return False
        
        print(f"\n💾 Saving {len(self.trained_models)} trained models...")
        
        try:
            # Create timestamp for this export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = self.models_dir / f"medical_models_{timestamp}"
            export_dir.mkdir(exist_ok=True)
            
            # Save individual models
            models_info = {}
            for target, model_info in self.trained_models.items():
                # Create safe filename
                safe_name = target.replace('(', '').replace(')', '').replace(' ', '_')
                model_filename = f"rf_model_{safe_name}.joblib"
                model_path = export_dir / model_filename
                
                # Save the trained model
                joblib.dump(model_info['model'], model_path)
                
                # Store model metadata (without the actual model object)
                models_info[target] = {k: v for k, v in model_info.items() if k != 'model'}
                models_info[target]['model_file'] = model_filename
                
                print(f"   ✅ Saved: {model_filename}")
            
            # Save preprocessing components
            preprocessing_path = export_dir / "preprocessing.joblib"
            preprocessing_data = {
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'target_columns': list(self.trained_models.keys())
            }
            joblib.dump(preprocessing_data, preprocessing_path)
            print(f"   ✅ Saved: preprocessing.joblib")
            
            # Save comprehensive metadata
            metadata = {
                'export_info': {
                    'timestamp': timestamp,
                    'total_models': len(self.trained_models),
                    'data_source': str(self.data_path),
                    'export_date': datetime.now().isoformat()
                },
                'models': models_info,
                'feature_columns': self.feature_columns,
                'target_columns': list(self.trained_models.keys())
            }
            
            metadata_path = export_dir / "models_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Saved: models_metadata.json")
            
            # Create unified prediction interface
            self.create_prediction_interface(export_dir)
            
            # Create model summary report
            self.create_summary_report(export_dir)
            
            print(f"\n🎉 Model export completed successfully!")
            print(f"📂 Models saved to: {export_dir}")
            print(f"📊 Total models exported: {len(self.trained_models)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False
    
    def create_prediction_interface(self, export_dir):
        """
        Create a unified prediction interface script.
        """
        interface_code = '''"""
Medical Prediction Interface
===========================

This script provides a unified interface for predicting all medical assessment and treatment labels
using the exported Random Forest models.

Usage:
    from medical_predictor import MedicalPredictor
    
    # Initialize predictor
    predictor = MedicalPredictor('path/to/models/directory')
    
    # Make predictions
    predictions = predictor.predict(patient_data)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MedicalPredictor:
    """
    Unified prediction interface for all medical assessment and treatment labels.
    """
    
    def __init__(self, models_dir):
        """
        Initialize the medical predictor.
        
        Parameters:
        models_dir (str): Path to the directory containing exported models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessing = None
        self.metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load all models and preprocessing components."""
        try:
            # Load metadata
            metadata_path = self.models_dir / "models_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load preprocessing components
            preprocessing_path = self.models_dir / "preprocessing.joblib"
            self.preprocessing = joblib.load(preprocessing_path)
            
            # Load individual models
            for target, model_info in self.metadata['models'].items():
                model_path = self.models_dir / model_info['model_file']
                self.models[target] = joblib.load(model_path)
            
            print(f"✅ Loaded {len(self.models)} models successfully")
            print(f"📋 Assessment models: {len([t for t in self.models.keys() if 'Nhận định' in t])}")
            print(f"🏥 Treatment models: {len([t for t in self.models.keys() if 'Kế hoạch' in t])}")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def preprocess_input(self, data):
        """
        Preprocess input data using the saved preprocessing components.
        
        Parameters:
        data (pd.DataFrame or dict): Input patient data
        
        Returns:
        pd.DataFrame: Preprocessed data ready for prediction
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        data = data.copy()
        
        # Ensure all feature columns are present
        for col in self.preprocessing['feature_columns']:
            if col not in data.columns:
                data[col] = np.nan
        
        # Select only feature columns
        data = data[self.preprocessing['feature_columns']]
        
        # Handle missing values
        for col in data.columns:
            if data[col].dtype in ['object', 'string']:
                data[col].fillna('Unknown', inplace=True)
            else:
                data[col].fillna(data[col].median() if not data[col].isna().all() else 0, inplace=True)
        
        # Apply label encoders
        for col, encoder in self.preprocessing['label_encoders'].items():
            if col in data.columns:
                data[col] = data[col].astype(str)
                # Handle unknown categories
                known_labels = set(encoder.classes_)
                unknown_mask = ~data[col].isin(known_labels)
                if unknown_mask.any():
                    data.loc[unknown_mask, col] = encoder.classes_[0]
                data[col] = encoder.transform(data[col])
        
        return data
    
    def predict(self, data, return_probabilities=False):
        """
        Make predictions for all labels.
        
        Parameters:
        data (pd.DataFrame or dict): Input patient data
        return_probabilities (bool): Whether to return prediction probabilities
        
        Returns:
        dict: Predictions for all labels
        """
        # Preprocess input
        processed_data = self.preprocess_input(data)
        
        predictions = {}
        
        for target, model in self.models.items():
            try:
                # Make prediction
                pred = model.predict(processed_data)[0]
                predictions[target] = int(pred)
                
                if return_probabilities:
                    prob = model.predict_proba(processed_data)[0]
                    predictions[f"{target}_probability"] = float(prob[1]) if len(prob) == 2 else float(max(prob))
                    
            except Exception as e:
                print(f"Warning: Error predicting {target}: {str(e)}")
                predictions[target] = 0
                if return_probabilities:
                    predictions[f"{target}_probability"] = 0.0
        
        return predictions
    
    def predict_batch(self, data, return_probabilities=False):
        """
        Make predictions for multiple patients.
        
        Parameters:
        data (pd.DataFrame): Input data for multiple patients
        return_probabilities (bool): Whether to return prediction probabilities
        
        Returns:
        pd.DataFrame: Predictions for all patients and labels
        """
        results = []
        
        for idx, row in data.iterrows():
            patient_predictions = self.predict(row.to_dict(), return_probabilities)
            patient_predictions['patient_index'] = idx
            results.append(patient_predictions)
        
        return pd.DataFrame(results)
    
    def get_assessment_predictions(self, data):
        """Get only assessment predictions."""
        all_predictions = self.predict(data, return_probabilities=True)
        return {k: v for k, v in all_predictions.items() if 'Nhận định' in k}
    
    def get_treatment_predictions(self, data):
        """Get only treatment predictions."""
        all_predictions = self.predict(data, return_probabilities=True)
        return {k: v for k, v in all_predictions.items() if 'Kế hoạch' in k}
    
    def get_model_info(self):
        """Get information about loaded models."""
        return self.metadata

# Example usage
if __name__ == "__main__":
    print("Medical Prediction Interface")
    print("=" * 50)
    
    # Example of how to use the predictor
    # predictor = MedicalPredictor('path/to/models/directory')
    
    # Sample patient data
    # patient_data = {
    #     'age': 25,
    #     'blood_pressure_systolic': 120,
    #     'heart_rate': 80,
    #     # ... other patient features
    # }
    
    # predictions = predictor.predict(patient_data, return_probabilities=True)
    # print("Predictions:", predictions)
'''
        
        interface_path = export_dir / "medical_predictor.py"
        with open(interface_path, 'w', encoding='utf-8') as f:
            f.write(interface_code)
        
        print(f"   ✅ Saved: medical_predictor.py")
    
    def create_summary_report(self, export_dir):
        """
        Create a comprehensive summary report of all models.
        """
        if not self.trained_models:
            return
        
        # Create summary statistics
        report = []
        report.append("MEDICAL PREDICTION MODELS - EXPORT SUMMARY")
        report.append("=" * 60)
        report.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Models: {len(self.trained_models)}")
        report.append(f"Data Source: {self.data_path}")
        report.append("")
        
        # Performance summary
        accuracies = [model['test_accuracy'] for model in self.trained_models.values()]
        auc_scores = [model['auc_score'] for model in self.trained_models.values() if model['auc_score']]
        
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        report.append(f"Average Test Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        if auc_scores:
            report.append(f"Average AUC Score: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        report.append(f"Best Model Accuracy: {max(accuracies):.4f}")
        report.append(f"Worst Model Accuracy: {min(accuracies):.4f}")
        report.append("")
        
        # Models by category
        assessment_models = [t for t in self.trained_models.keys() if 'Nhận định' in t]
        treatment_models = [t for t in self.trained_models.keys() if 'Kế hoạch' in t]
        
        report.append("MODELS BY CATEGORY:")
        report.append("-" * 30)
        report.append(f"Assessment Models: {len(assessment_models)}")
        report.append(f"Treatment Models: {len(treatment_models)}")
        report.append("")
        
        # Top performing models
        sorted_models = sorted(self.trained_models.items(), 
                             key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        report.append("TOP 15 PERFORMING MODELS:")
        report.append("-" * 50)
        report.append(f"{'Rank':<4} {'Accuracy':<8} {'AUC':<8} {'Target'}")
        report.append("-" * 50)
        
        for i, (target, model) in enumerate(sorted_models[:15], 1):
            auc_str = f"{model['auc_score']:.4f}" if model['auc_score'] else "N/A"
            short_target = target.split('_', 1)[1] if '_' in target else target
            report.append(f"{i:<4} {model['test_accuracy']:<8.4f} {auc_str:<8} {short_target}")
        
        report.append("")
        
        # Detailed model information
        report.append("DETAILED MODEL INFORMATION:")
        report.append("-" * 50)
        
        for target, model in sorted_models:
            report.append(f"\n{target}:")
            report.append(f"  Test Accuracy: {model['test_accuracy']:.4f}")
            report.append(f"  Train Accuracy: {model['train_accuracy']:.4f}")
            if model['auc_score']:
                report.append(f"  AUC Score: {model['auc_score']:.4f}")
            report.append(f"  Class Distribution: {model['class_distribution']}")
            report.append(f"  Imbalance Ratio: {model['imbalance_ratio']:.1f}:1")
            report.append(f"  Training Time: {model['training_time']:.1f}s")
            report.append(f"  Features Used: {model['n_features']}")
            report.append(f"  Training Samples: {model['n_samples']}")
        
        # Save report
        report_path = export_dir / "models_summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"   ✅ Saved: models_summary_report.txt")

def main():
    """
    Main function to export all medical prediction models.
    """
    print("🚀 Medical Multi-Label Model Export System")
    print("=" * 60)
    
    # Set up paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "dataset_long_format_normalized_labeled.csv"
    models_output_dir = project_root / "models"
    
    # Verify data file exists
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please ensure the labeled dataset exists.")
        return
    
    print(f"📂 Data source: {data_path}")
    print(f"💾 Models output: {models_output_dir}")
    
    # Initialize exporter
    exporter = MultiLabelModelExporter(str(data_path), str(models_output_dir))
    
    # Load and prepare data
    if not exporter.load_and_prepare_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Train all models
    print(f"\n🎯 Training models for {len(exporter.target_columns)} labels...")
    training_results = exporter.train_all_models(min_samples=10, max_imbalance=50)
    
    if training_results['successful_models'] == 0:
        print("❌ No models were successfully trained. Exiting.")
        return
    
    # Save models
    if exporter.save_models():
        print(f"\n🎉 SUCCESS! Medical prediction models exported successfully!")
        print(f"✅ Models trained: {training_results['successful_models']}/{training_results['total_targets']}")
        print(f"📊 Success rate: {training_results['successful_models']/training_results['total_targets']*100:.1f}%")
        print(f"⏱️ Total time: {training_results['total_time']/60:.1f} minutes")
        
        print(f"\n📋 USAGE INSTRUCTIONS:")
        print(f"1. Import the predictor: from medical_predictor import MedicalPredictor")
        print(f"2. Initialize: predictor = MedicalPredictor('models/medical_models_YYYYMMDD_HHMMSS')")
        print(f"3. Make predictions: predictions = predictor.predict(patient_data)")
        print(f"4. Check models_summary_report.txt for detailed performance metrics")
        
    else:
        print("❌ Failed to save models.")

if __name__ == "__main__":
    main()