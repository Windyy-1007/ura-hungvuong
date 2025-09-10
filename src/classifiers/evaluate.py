"""
Comprehensive Model Evaluation Script
====================================

This script evaluates trained models with detailed metrics including:
- Accuracy, Precision, Recall, F1-Score
- Per-class and overall performance
- Confusion matrices
- ROC curves and AUC scores
- Classification reports

Supports both individual models and multi-output models.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from dataloader.loader import load_data

class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics"""
    
    def __init__(self, config_path=None):
        """Initialize evaluator with configuration"""
        self.config_path = config_path or "config_all_labels.json"
        self.config = self._load_config()
        self.results = {}
        
    def _load_config(self):
        """Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration if file not found"""
        return {
            "training_config": {"test_size": 0.2, "min_samples": 10, "max_imbalance": 50},
            "feature_config": {"exclude_features": []},
            "target_config": {"predict_treatments": True, "predict_assessments": False}
        }
    
    def _handle_missing_values(self, X):
        """Handle missing values in features"""
        # Fill numeric columns with median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
        
        return X
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        return X
    
    def load_data_and_split(self):
        """Load and split data for evaluation"""
        print("📊 Loading and preparing data...")
        
        # Load data - try to find the dataset file
        possible_files = [
            "data/dataset_long_format_normalized_labeled.csv",
            "data/dataset.csv",
            "data/dataset_long_format.csv"
        ]
        
        file_path = None
        for f in possible_files:
            if os.path.exists(f):
                file_path = f
                break
        
        if file_path is None:
            raise FileNotFoundError(f"Could not find dataset file. Tried: {possible_files}")
        
        self.df = load_data(file_path)
        if self.df is None:
            raise FileNotFoundError(f"Could not load data from {file_path}")
        print(f"✅ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        
        # Get features and targets
        exclude_features = self.config['feature_config'].get('exclude_features', [])
        
        # Identify target columns
        target_prefixes = []
        if self.config['target_config'].get('predict_treatments', True):
            target_prefixes.append('Kế hoạch (xử trí)')
        if self.config['target_config'].get('predict_assessments', False):
            target_prefixes.append('Nhận định và đánh giá')
        
        target_columns = [col for col in self.df.columns 
                         if any(col.startswith(prefix) for prefix in target_prefixes)]
        
        # Get feature columns
        feature_columns = [col for col in self.df.columns 
                          if col not in target_columns and col not in exclude_features]
        
        print(f"📋 Features: {len(feature_columns)} columns")
        print(f"🎯 Targets: {len(target_columns)} columns")
        
        # Prepare features
        X = self.df[feature_columns].copy()
        
        # Handle missing values and encode categorical variables
        X = self._handle_missing_values(X)
        X = self._encode_categorical_features(X)
        
        # Prepare targets
        y = self.df[target_columns].copy()
        
        # Filter targets with sufficient data
        valid_targets = []
        min_samples = self.config.get('training_config', {}).get('min_samples', 10)
        max_imbalance = self.config.get('training_config', {}).get('max_imbalance', 50)
        
        for col in target_columns:
            if col in y.columns:
                value_counts = y[col].value_counts()
                if len(value_counts) >= 2:  # At least 2 classes
                    min_count = value_counts.min()
                    max_count = value_counts.max()
                    
                    if min_count >= min_samples and max_count / min_count <= max_imbalance:
                        valid_targets.append(col)
                    else:
                        print(f"⚠️  Skipping {col}: min_samples={min_count}, imbalance={max_count/min_count:.1f}")
        
        y = y[valid_targets]
        print(f"✅ Using {len(valid_targets)} valid targets")
        
        # Train-test split
        test_size = self.config.get('training_config', {}).get('test_size', 0.2)
        random_state = self.config.get('training_config', {}).get('random_state', 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y.iloc[:, 0] if len(y.columns) > 0 else None
        )
        
        self.feature_columns = feature_columns
        self.target_columns = valid_targets
        
        print(f"📈 Training set: {len(self.X_train)} samples")
        print(f"📊 Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def evaluate_multi_output_model(self, model_path):
        """Evaluate multi-output model"""
        print(f"\n🔍 Evaluating multi-output model: {model_path}")
        
        # Load model
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, "multi_output_medical_model.joblib")
            if not os.path.exists(model_file):
                # Look for any .joblib file in the directory
                joblib_files = [f for f in os.listdir(model_path) if f.endswith('.joblib') and 'multi_output' in f]
                if joblib_files:
                    model_file = os.path.join(model_path, joblib_files[0])
                else:
                    raise FileNotFoundError(f"No model file found in {model_path}")
        else:
            model_file = model_path
        
        print(f"📁 Loading model from: {model_file}")
        model = joblib.load(model_file)
        
        # Load data
        self.load_data_and_split()
        
        # Make predictions
        print("🔮 Making predictions...")
        y_pred = model.predict(self.X_test)
        
        # If model has predict_proba method, get probabilities
        try:
            y_pred_proba = model.predict_proba(self.X_test)
            has_proba = True
        except:
            y_pred_proba = None
            has_proba = False
        
        # Calculate metrics for each target
        evaluation_results = []
        
        for i, target in enumerate(self.target_columns):
            target_name = target.replace('Kế hoạch (xử trí)_', '').replace('Nhận định và đánh giá_', '')
            
            # Get true and predicted values for this target
            y_true_target = self.y_test.iloc[:, i] if i < self.y_test.shape[1] else self.y_test[target]
            y_pred_target = y_pred[:, i] if y_pred.ndim > 1 else y_pred
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_true_target, y_pred_target)
            precision = precision_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
            recall = recall_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
            f1 = f1_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
            
            # Calculate AUC if probabilities available
            auc = None
            if has_proba and y_pred_proba is not None:
                try:
                    if len(np.unique(y_true_target)) == 2:  # Binary classification
                        if hasattr(y_pred_proba, '__iter__') and len(y_pred_proba) > i:
                            if hasattr(y_pred_proba[i], '__iter__') and len(y_pred_proba[i][0]) > 1:
                                auc = roc_auc_score(y_true_target, y_pred_proba[i][:, 1])
                            else:
                                auc = roc_auc_score(y_true_target, y_pred_proba[i])
                        else:
                            # Try alternative approach
                            unique_classes = np.unique(y_true_target)
                            if len(unique_classes) == 2:
                                auc = roc_auc_score(y_true_target, y_pred_target)
                except Exception as e:
                    print(f"⚠️  Could not calculate AUC for {target_name}: {e}")
                    auc = None
            
            # Store results
            result = {
                'Target': target_name,
                'Full_Target_Name': target,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'AUC': auc,
                'Support': len(y_true_target)
            }
            
            evaluation_results.append(result)
            
            print(f"📊 {target_name:<25} | Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}" + 
                  (f" | AUC: {auc:.3f}" if auc is not None else ""))
        
        # Create results DataFrame
        results_df = pd.DataFrame(evaluation_results)
        
        # Calculate overall metrics
        overall_metrics = {
            'Average_Accuracy': results_df['Accuracy'].mean(),
            'Average_Precision': results_df['Precision'].mean(),
            'Average_Recall': results_df['Recall'].mean(),
            'Average_F1_Score': results_df['F1_Score'].mean(),
            'Average_AUC': results_df['AUC'].dropna().mean() if not results_df['AUC'].dropna().empty else None,
            'Total_Targets': len(results_df),
            'Total_Samples': len(self.X_test)
        }
        
        # Print summary
        print(f"\n📋 OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"Average Accuracy:  {overall_metrics['Average_Accuracy']:.3f}")
        print(f"Average Precision: {overall_metrics['Average_Precision']:.3f}")
        print(f"Average Recall:    {overall_metrics['Average_Recall']:.3f}")
        print(f"Average F1-Score:  {overall_metrics['Average_F1_Score']:.3f}")
        if overall_metrics['Average_AUC'] is not None:
            print(f"Average AUC:       {overall_metrics['Average_AUC']:.3f}")
        print(f"Total Targets:     {overall_metrics['Total_Targets']}")
        print(f"Test Samples:      {overall_metrics['Total_Samples']}")
        
        return results_df, overall_metrics
    
    def save_evaluation_report(self, results_df, overall_metrics, model_path, output_dir="data"):
        """Save detailed evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path.rstrip('/\\'))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_file = os.path.join(output_dir, f"evaluation_report_{model_name}_{timestamp}.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV report: {csv_file}")
        
        # Save detailed text report
        txt_file = os.path.join(output_dir, f"evaluation_report_{model_name}_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Model Evaluation Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config_path}\n\n")
            
            f.write(f"OVERALL METRICS\n")
            f.write(f"{'-'*30}\n")
            for key, value in overall_metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"{key:<20}: {value:.4f}\n")
                    else:
                        f.write(f"{key:<20}: {value}\n")
            
            f.write(f"\nDETAILED RESULTS BY TARGET\n")
            f.write(f"{'-'*50}\n")
            f.write(f"{'Target':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Support':<10}\n")
            f.write(f"{'-'*100}\n")
            
            for _, row in results_df.iterrows():
                auc_str = f"{row['AUC']:.3f}" if pd.notna(row['AUC']) else "N/A"
                f.write(f"{row['Target']:<25} {row['Accuracy']:<10.3f} {row['Precision']:<10.3f} {row['Recall']:<10.3f} {row['F1_Score']:<10.3f} {auc_str:<10} {row['Support']:<10}\n")
        
        print(f"📄 Saved detailed report: {txt_file}")
        
        return csv_file, txt_file
    
    def create_evaluation_table(self, results_df, title="Model Evaluation Results"):
        """Create a formatted evaluation table"""
        print(f"\n{title}")
        print("=" * len(title))
        print()
        
        # Format the dataframe for display
        display_df = results_df.copy()
        
        # Round numeric columns
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        # Replace NaN with 'N/A'
        display_df = display_df.fillna('N/A')
        
        # Print formatted table
        print(f"{'Target':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Support':<10}")
        print("-" * 100)
        
        for _, row in display_df.iterrows():
            target_name = row['Target'][:29] if len(row['Target']) > 29 else row['Target']
            print(f"{target_name:<30} {row['Accuracy']:<10} {row['Precision']:<10} {row['Recall']:<10} {row['F1_Score']:<10} {row['AUC']:<10} {row['Support']:<10}")
        
        print("-" * 100)
        
        # Print averages
        avg_acc = display_df[display_df['Accuracy'] != 'N/A']['Accuracy'].mean()
        avg_prec = display_df[display_df['Precision'] != 'N/A']['Precision'].mean()
        avg_rec = display_df[display_df['Recall'] != 'N/A']['Recall'].mean()
        avg_f1 = display_df[display_df['F1_Score'] != 'N/A']['F1_Score'].mean()
        avg_auc = display_df[display_df['AUC'] != 'N/A']['AUC'].mean() if len(display_df[display_df['AUC'] != 'N/A']) > 0 else 'N/A'
        total_support = display_df[display_df['Support'] != 'N/A']['Support'].sum() if len(display_df[display_df['Support'] != 'N/A']) > 0 else 'N/A'
        
        print(f"{'AVERAGE':<30} {avg_acc:<10.3f} {avg_prec:<10.3f} {avg_rec:<10.3f} {avg_f1:<10.3f} {avg_auc if avg_auc == 'N/A' else f'{avg_auc:<10.3f}'} {total_support}")
        print()


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', '-m', 
                       default='models/multi_output_medical_model_20250910_162904',
                       help='Path to model file or directory')
    parser.add_argument('--config', '-c',
                       default='config_all_labels.json',
                       help='Path to configuration file')
    parser.add_argument('--output', '-o',
                       default='data',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Check if model path exists
    if not os.path.exists(args.model):
        print(f"❌ Model path not found: {args.model}")
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}")
                elif item.endswith('.joblib'):
                    print(f"  📄 {item}")
        return
    
    print(f"🚀 Starting evaluation for: {args.model}")
    
    try:
        # Evaluate model
        results_df, overall_metrics = evaluator.evaluate_multi_output_model(args.model)
        
        # Create evaluation table
        evaluator.create_evaluation_table(results_df, "🎯 Multi-Output Model Evaluation Results")
        
        # Save report
        csv_file, txt_file = evaluator.save_evaluation_report(
            results_df, overall_metrics, args.model, args.output
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: {csv_file}")
        print(f"📄 Detailed report: {txt_file}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
