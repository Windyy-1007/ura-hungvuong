"""
Multi-Output Medical Prediction Model Export Script
==================================================

This script trains a SINGLE Random Forest model that can predict ALL treatment labels
simultaneously using multi-output classification. This creates just one weight file
instead of separate models for each label.

Features:
- Single multi-output Random Forest model
- Configurable features and targets via config.json
- One unified weight file for all predictions
- Efficient prediction for multiple labels at once
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

class MultiOutputMedicalExporter:
    """
    Exports a single multi-output trained model for all medical prediction labels.
    Creates one unified weight file that can predict all labels simultaneously.
    """
    
    def __init__(self, config_path, data_path, models_output_dir):
        """
        Initialize the multi-output model exporter.
        
        Parameters:
        config_path (str): Path to the configuration JSON file
        data_path (str): Path to the labeled dataset CSV file
        models_output_dir (str): Directory to save exported models
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
        
        print(f"Multi-Output Medical Model Exporter initialized")
        print(f"Config: {self.config_path}")
        print(f"Data source: {self.data_path}")
        print(f"Models output: {self.models_dir}")
    
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
        """
        Load the dataset and identify target columns and features based on config.
        
        Returns:
        bool: True if data loaded successfully, False otherwise
        """
        try:
            print(f"\n📂 Loading dataset from: {self.data_path}")
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"✅ Dataset loaded successfully. Shape: {self.df.shape}")
            
            # Identify target columns based on config
            all_target_columns = [col for col in self.df.columns 
                                if ('Nhận định và đánh giá_' in col or 'Kế hoạch (xử trí)_' in col)]
            
            # Filter targets based on configuration
            target_config = self.config['target_config']
            self.target_columns = []
            
            if target_config.get('predict_assessments', True):
                assessment_targets = [col for col in all_target_columns if 'Nhận định và đánh giá_' in col]
                self.target_columns.extend(assessment_targets)
            
            if target_config.get('predict_treatments', True):
                treatment_targets = [col for col in all_target_columns if 'Kế hoạch (xử trí)_' in col]
                self.target_columns.extend(treatment_targets)
            
            # Add custom targets
            if target_config.get('custom_targets'):
                self.target_columns.extend(target_config['custom_targets'])
            
            # Remove excluded targets
            if target_config.get('exclude_targets'):
                self.target_columns = [col for col in self.target_columns 
                                     if col not in target_config['exclude_targets']]
            
            print(f"🎯 Selected {len(self.target_columns)} target labels for multi-output prediction")
            
            # Categorize targets for reporting
            assessment_targets = [col for col in self.target_columns if 'Nhận định và đánh giá_' in col]
            treatment_targets = [col for col in self.target_columns if 'Kế hoạch (xử trí)_' in col]
            
            print(f"   📋 Assessment labels: {len(assessment_targets)}")
            print(f"   🏥 Treatment labels: {len(treatment_targets)}")
            
            # Identify feature columns based on config
            feature_config = self.config['feature_config']
            
            if feature_config.get('use_all_features', True):
                # Start with all columns
                self.feature_columns = list(self.df.columns)
                
                # Remove target columns
                self.feature_columns = [col for col in self.feature_columns if col not in self.target_columns]
                
                # Remove excluded features
                exclude_features = feature_config.get('exclude_features', [])
                self.feature_columns = [col for col in self.feature_columns if col not in exclude_features]
            else:
                # Use only specified features
                self.feature_columns = feature_config.get('include_features', [])
            
            print(f"🔢 Selected feature columns: {len(self.feature_columns)}")
            
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
    
    def filter_valid_targets(self):
        """
        Filter target columns to only include those suitable for training.
        Removes targets with insufficient samples or extreme imbalance.
        
        Returns:
        list: Valid target columns for multi-output training
        """
        training_config = self.config['training_config']
        min_samples = training_config.get('min_samples', 10)
        max_imbalance = training_config.get('max_imbalance', 50)
        
        valid_targets = []
        skipped_targets = []
        
        print(f"\n🔍 Filtering targets for multi-output training...")
        print(f"   Minimum minority samples: {min_samples}")
        print(f"   Maximum imbalance ratio: {max_imbalance}:1")
        
        for target in self.target_columns:
            class_dist = self.df[target].value_counts()
            
            if len(class_dist) < 2:
                skipped_targets.append((target, "Only one class present"))
                continue
            
            minority_count = class_dist.min()
            majority_count = class_dist.max()
            imbalance_ratio = majority_count / minority_count
            
            if minority_count < min_samples:
                skipped_targets.append((target, f"Minority class has only {minority_count} samples"))
                continue
            
            if imbalance_ratio > max_imbalance:
                skipped_targets.append((target, f"Class imbalance too high ({imbalance_ratio:.1f}:1)"))
                continue
            
            valid_targets.append(target)
        
        print(f"\n✅ Valid targets for multi-output: {len(valid_targets)}/{len(self.target_columns)}")
        
        if skipped_targets:
            print(f"\n⚠️ Skipped targets ({len(skipped_targets)}):")
            for target, reason in skipped_targets:
                short_name = target.split('_', 1)[1] if '_' in target else target
                print(f"   ❌ {short_name}: {reason}")
        
        return valid_targets
    
    def train_multi_output_model(self):
        """
        Train a single multi-output Random Forest model for all valid targets.
        
        Returns:
        dict: Training results and metadata
        """
        print(f"\n🚀 Training multi-output Random Forest model...")
        
        # Filter valid targets
        valid_targets = self.filter_valid_targets()
        
        if len(valid_targets) == 0:
            print("❌ No valid targets found for training!")
            return None
        
        # Update target columns to only valid ones
        self.target_columns = valid_targets
        
        # Prepare features and targets
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_columns].copy()
        
        # Remove rows with any missing targets
        valid_mask = ~y.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"   📊 Training samples: {len(X)}")
        print(f"   🎯 Target labels: {len(self.target_columns)}")
        print(f"   🔢 Features: {len(self.feature_columns)}")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit_encoders=True)
        
        # Split data
        training_config = self.config['training_config']
        test_size = training_config.get('test_size', 0.2)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )
        
        print(f"   📈 Training set: {len(X_train)}")
        print(f"   📊 Test set: {len(X_test)}")
        
        # Get model parameters from config
        model_config = self.config['model_config']
        model_params = model_config.get('parameters', {})
        
        # Create multi-output Random Forest
        base_rf = RandomForestClassifier(**model_params)
        self.multi_output_model = MultiOutputClassifier(base_rf)
        
        print(f"\n⚙️ Training multi-output model...")
        start_time = time.time()
        
        # Train the model
        self.multi_output_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        print(f"✅ Multi-output training completed in {training_time:.1f}s")
        
        # Evaluate the model
        print(f"\n📊 Evaluating multi-output model...")
        
        # Make predictions
        y_pred_train = self.multi_output_model.predict(X_train)
        y_pred_test = self.multi_output_model.predict(X_test)
        
        # Ensure predictions are numpy arrays
        y_pred_train = np.array(y_pred_train)
        y_pred_test = np.array(y_pred_test)
        
        # Calculate per-target metrics
        target_metrics = {}
        
        for i, target in enumerate(self.target_columns):
            # Get actual values for this target
            y_train_target = y_train.iloc[:, i].values
            y_test_target = y_test.iloc[:, i].values
            y_pred_train_target = y_pred_train[:, i]
            y_pred_test_target = y_pred_test[:, i]
            
            train_acc = accuracy_score(y_train_target, y_pred_train_target)
            test_acc = accuracy_score(y_test_target, y_pred_test_target)
            
            # Try to calculate AUC
            try:
                y_pred_proba = self.multi_output_model.predict_proba(X_test)
                if len(y_pred_proba[i]) > 0 and y_pred_proba[i].shape[1] == 2:
                    auc_score = roc_auc_score(y_test_target, y_pred_proba[i][:, 1])
                else:
                    auc_score = None
            except:
                auc_score = None
            
            # Get classification report
            class_report = classification_report(y_test_target, y_pred_test_target, output_dict=True)
            
            target_metrics[target] = {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'auc_score': float(auc_score) if auc_score else None,
                'classification_report': class_report
            }
            
            short_name = target.split('_', 1)[1] if '_' in target else target
            auc_display = f"{auc_score:.3f}" if auc_score else "N/A"
            print(f"   {short_name:<35} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | AUC: {auc_display}")
        
        # Overall metrics
        overall_train_acc = np.mean([metrics['train_accuracy'] for metrics in target_metrics.values()])
        overall_test_acc = np.mean([metrics['test_accuracy'] for metrics in target_metrics.values()])
        auc_scores = [metrics['auc_score'] for metrics in target_metrics.values() if metrics['auc_score']]
        overall_auc = np.mean(auc_scores) if auc_scores else None
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Average Train Accuracy: {overall_train_acc:.4f}")
        print(f"   Average Test Accuracy: {overall_test_acc:.4f}")
        if overall_auc:
            print(f"   Average AUC Score: {overall_auc:.4f}")
        
        # Store training results
        training_results = {
            'model': self.multi_output_model,
            'target_columns': self.target_columns,
            'feature_columns': list(X_processed.columns),
            'training_time': float(training_time),
            'overall_train_accuracy': float(overall_train_acc),
            'overall_test_accuracy': float(overall_test_acc),
            'overall_auc_score': float(overall_auc) if overall_auc else None,
            'target_metrics': target_metrics,
            'n_targets': len(self.target_columns),
            'n_features': len(X_processed.columns),
            'n_samples': len(X_processed),
            'config_used': self.config
        }
        
        return training_results
    
    def save_multi_output_model(self, training_results):
        """
        Save the multi-output model and create prediction interface.
        
        Parameters:
        training_results (dict): Results from training
        
        Returns:
        bool: True if saved successfully
        """
        if not training_results:
            print("❌ No training results to save!")
            return False
        
        print(f"\n💾 Saving multi-output model...")
        
        try:
            # Create timestamp for this export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = self.models_dir / f"multi_output_medical_model_{timestamp}"
            export_dir.mkdir(exist_ok=True)
            
            # Save the multi-output model (single weight file)
            model_path = export_dir / "multi_output_medical_model.joblib"
            joblib.dump(training_results['model'], model_path)
            print(f"   ✅ Saved: multi_output_medical_model.joblib")
            
            # Save preprocessing components
            preprocessing_path = export_dir / "preprocessing.joblib"
            preprocessing_data = {
                'label_encoders': self.label_encoders,
                'feature_columns': training_results['feature_columns'],
                'target_columns': training_results['target_columns']
            }
            joblib.dump(preprocessing_data, preprocessing_path)
            print(f"   ✅ Saved: preprocessing.joblib")
            
            # Save metadata (without the model object)
            metadata = {
                'export_info': {
                    'timestamp': timestamp,
                    'export_date': datetime.now().isoformat(),
                    'model_type': 'multi_output_random_forest',
                    'total_targets': training_results['n_targets'],
                    'total_features': training_results['n_features'],
                    'data_source': str(self.data_path)
                },
                'performance': {
                    'overall_train_accuracy': training_results['overall_train_accuracy'],
                    'overall_test_accuracy': training_results['overall_test_accuracy'],
                    'overall_auc_score': training_results['overall_auc_score'],
                    'training_time': training_results['training_time']
                },
                'target_metrics': training_results['target_metrics'],
                'feature_columns': training_results['feature_columns'],
                'target_columns': training_results['target_columns'],
                'config_used': training_results['config_used']
            }
            
            metadata_path = export_dir / "model_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Saved: model_metadata.json")
            
            # Create unified prediction interface
            self.create_multi_output_predictor(export_dir, training_results)
            
            # Create summary report
            self.create_multi_output_report(export_dir, training_results)
            
            print(f"\n🎉 Multi-output model export completed successfully!")
            print(f"📂 Model saved to: {export_dir}")
            print(f"📊 Single model file predicts {training_results['n_targets']} labels")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def create_multi_output_predictor(self, export_dir, training_results):
        """Create a unified prediction interface for the multi-output model."""
        
        interface_code = '''"""
Multi-Output Medical Prediction Interface
=========================================

This interface provides predictions for ALL medical labels using a single
multi-output Random Forest model. More efficient than individual models.

Usage:
    from multi_output_predictor import MultiOutputMedicalPredictor
    
    # Initialize predictor
    predictor = MultiOutputMedicalPredictor('path/to/model/directory')
    
    # Make predictions for all labels at once
    predictions = predictor.predict(patient_data)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultiOutputMedicalPredictor:
    """
    Multi-output prediction interface for all medical labels using a single model.
    More efficient and consistent than individual models.
    """
    
    def __init__(self, models_dir):
        """
        Initialize the multi-output medical predictor.
        
        Parameters:
        models_dir (str): Path to the directory containing exported model
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.preprocessing = None
        self.metadata = None
        
        self.load_model()
    
    def load_model(self):
        """Load the multi-output model and preprocessing components."""
        try:
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load preprocessing components
            preprocessing_path = self.models_dir / "preprocessing.joblib"
            self.preprocessing = joblib.load(preprocessing_path)
            
            # Load the multi-output model
            model_path = self.models_dir / "multi_output_medical_model.joblib"
            self.model = joblib.load(model_path)
            
            print(f"✅ Multi-output model loaded successfully")
            print(f"🎯 Predicts {len(self.preprocessing['target_columns'])} labels simultaneously")
            print(f"📋 Assessment labels: {len([t for t in self.preprocessing['target_columns'] if 'Nhận định' in t])}")
            print(f"🏥 Treatment labels: {len([t for t in self.preprocessing['target_columns'] if 'Kế hoạch' in t])}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
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
        Make predictions for all labels using the multi-output model.
        
        Parameters:
        data (pd.DataFrame or dict): Input patient data
        return_probabilities (bool): Whether to return prediction probabilities
        
        Returns:
        dict: Predictions for all labels
        """
        # Preprocess input
        processed_data = self.preprocess_input(data)
        
        # Make predictions
        predictions_array = self.model.predict(processed_data)[0]
        
        # Convert to dictionary
        predictions = {}
        for i, target in enumerate(self.preprocessing['target_columns']):
            predictions[target] = int(predictions_array[i])
        
        if return_probabilities:
            try:
                # Get probabilities for all outputs
                probabilities_list = self.model.predict_proba(processed_data)
                
                for i, target in enumerate(self.preprocessing['target_columns']):
                    if len(probabilities_list[i][0]) == 2:  # Binary classification
                        predictions[f"{target}_probability"] = float(probabilities_list[i][0][1])
                    else:
                        predictions[f"{target}_probability"] = float(max(probabilities_list[i][0]))
            except Exception as e:
                print(f"Warning: Could not calculate probabilities: {e}")
        
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
        """Get information about loaded model."""
        return self.metadata
    
    def get_target_summary(self):
        """Get a summary of prediction targets."""
        targets = self.preprocessing['target_columns']
        assessments = [t for t in targets if 'Nhận định' in t]
        treatments = [t for t in targets if 'Kế hoạch' in t]
        
        return {
            'total_targets': len(targets),
            'assessment_targets': len(assessments),
            'treatment_targets': len(treatments),
            'all_targets': targets,
            'assessments': assessments,
            'treatments': treatments
        }

# Example usage
if __name__ == "__main__":
    print("Multi-Output Medical Prediction Interface")
    print("=" * 50)
    
    # Example of how to use the predictor
    # predictor = MultiOutputMedicalPredictor('path/to/model/directory')
    
    # Sample patient data
    # patient_data = {
    #     'Mạch (nhập số nguyên)': 80,
    #     'HA tâm thu (nhập số nguyên)': 120,
    #     # ... other patient features
    # }
    
    # predictions = predictor.predict(patient_data, return_probabilities=True)
    # print("All predictions:", predictions)
'''
        
        interface_path = export_dir / "multi_output_predictor.py"
        with open(interface_path, 'w', encoding='utf-8') as f:
            f.write(interface_code)
        
        print(f"   ✅ Saved: multi_output_predictor.py")
    
    def create_multi_output_report(self, export_dir, training_results):
        """Create a comprehensive summary report."""
        
        report = []
        report.append("MULTI-OUTPUT MEDICAL PREDICTION MODEL - EXPORT SUMMARY")
        report.append("=" * 70)
        report.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model Type: Multi-Output Random Forest")
        report.append(f"Total Target Labels: {training_results['n_targets']}")
        report.append(f"Total Features: {training_results['n_features']}")
        report.append(f"Training Samples: {training_results['n_samples']}")
        report.append(f"Data Source: {self.data_path}")
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        report.append(f"Average Train Accuracy: {training_results['overall_train_accuracy']:.4f}")
        report.append(f"Average Test Accuracy: {training_results['overall_test_accuracy']:.4f}")
        if training_results['overall_auc_score']:
            report.append(f"Average AUC Score: {training_results['overall_auc_score']:.4f}")
        report.append(f"Training Time: {training_results['training_time']:.1f}s")
        report.append("")
        
        # Target breakdown
        assessment_targets = [t for t in training_results['target_columns'] if 'Nhận định' in t]
        treatment_targets = [t for t in training_results['target_columns'] if 'Kế hoạch' in t]
        
        report.append("TARGETS BY CATEGORY:")
        report.append("-" * 30)
        report.append(f"Assessment Targets: {len(assessment_targets)}")
        report.append(f"Treatment Targets: {len(treatment_targets)}")
        report.append("")
        
        # Individual target performance
        report.append("INDIVIDUAL TARGET PERFORMANCE:")
        report.append("-" * 60)
        report.append(f"{'Target':<40} {'Test Acc':<10} {'AUC':<8}")
        report.append("-" * 60)
        
        # Sort by test accuracy
        sorted_targets = sorted(training_results['target_metrics'].items(), 
                              key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for target, metrics in sorted_targets:
            short_target = target.split('_', 1)[1] if '_' in target else target
            auc_str = f"{metrics['auc_score']:.4f}" if metrics['auc_score'] else "N/A"
            report.append(f"{short_target:<40} {metrics['test_accuracy']:<10.4f} {auc_str:<8}")
        
        report.append("")
        
        # Configuration used
        report.append("CONFIGURATION USED:")
        report.append("-" * 30)
        config = training_results['config_used']
        
        report.append(f"Model: {config['model_config']['algorithm']}")
        report.append(f"Estimators: {config['model_config']['parameters']['n_estimators']}")
        report.append(f"Max Depth: {config['model_config']['parameters']['max_depth']}")
        report.append(f"Test Size: {config['training_config']['test_size']}")
        report.append(f"Min Samples: {config['training_config']['min_samples']}")
        report.append(f"Max Imbalance: {config['training_config']['max_imbalance']}")
        
        # Save report
        report_path = export_dir / "multi_output_model_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"   ✅ Saved: multi_output_model_report.txt")

def main():
    """Main function to export multi-output medical prediction model."""
    
    print("🚀 Multi-Output Medical Model Export System")
    print("=" * 70)
    
    # Set up paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent if current_dir.name == "src" else current_dir.parent.parent
    
    config_path = project_root / "config_all_labels.json"
    data_path = project_root / "data" / "dataset_long_format_normalized_labeled.csv"
    models_output_dir = project_root / "models"
    
    # Verify files exist
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        return
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    print(f"📂 Config: {config_path}")
    print(f"📊 Data: {data_path}")
    print(f"💾 Output: {models_output_dir}")
    
    # Initialize exporter
    exporter = MultiOutputMedicalExporter(str(config_path), str(data_path), str(models_output_dir))
    
    # Load and prepare data
    if not exporter.load_and_prepare_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Train multi-output model
    training_results = exporter.train_multi_output_model()
    
    if not training_results:
        print("❌ Training failed. Exiting.")
        return
    
    # Save model
    if exporter.save_multi_output_model(training_results):
        print(f"\n🎉 SUCCESS! Multi-output medical model exported!")
        print(f"✅ Single model file predicts {training_results['n_targets']} labels")
        print(f"📊 Average accuracy: {training_results['overall_test_accuracy']:.4f}")
        print(f"⏱️ Training time: {training_results['training_time']:.1f}s")
        
        print(f"\n📋 USAGE INSTRUCTIONS:")
        print(f"1. Import: from multi_output_predictor import MultiOutputMedicalPredictor")
        print(f"2. Initialize: predictor = MultiOutputMedicalPredictor('models/multi_output_medical_model_YYYYMMDD_HHMMSS')")
        print(f"3. Predict: predictions = predictor.predict(patient_data)")
        print(f"4. Single model file handles all {training_results['n_targets']} labels efficiently!")
        
    else:
        print("❌ Failed to save model.")

if __name__ == "__main__":
    main()
