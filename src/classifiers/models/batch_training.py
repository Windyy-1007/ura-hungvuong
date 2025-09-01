"""
Batch Random Forest Training for Medical Prediction
Trains models for multiple target variables and compares performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from classifiers.models.random_forest import MedicalRandomForestClassifier

def train_multiple_targets(data_path, target_list=None, save_results=True):
    """
    Train Random Forest models for multiple target variables.
    
    Parameters:
    data_path (str): Path to the labeled CSV file
    target_list (list): List of target columns to train. If None, trains popular targets.
    save_results (bool): Whether to save results to CSV
    
    Returns:
    pd.DataFrame: Results summary for all models
    """
    
    print("Batch Medical Random Forest Training")
    print("=" * 60)
    
    # Initialize classifier
    rf_classifier = MedicalRandomForestClassifier()
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = rf_classifier.load_data(data_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Get available targets if not specified
    available_targets = rf_classifier.get_available_targets()
    
    if target_list is None:
        # Select high-frequency targets for training
        target_list = [
            'Nhận định và đánh giá_CTG_Group_II',           # 28.1% prevalence
            'Kế hoạch (xử trí)_Monitor_Labor',               # 26.5% prevalence  
            'Kế hoạch (xử trí)_Report_Doctor',               # 30.0% prevalence
            'Nhận định và đánh giá_Guidance_Push',           # 16.4% prevalence
            'Kế hoạch (xử trí)_Neonatal_Resuscitation',      # 16.3% prevalence
            'Kế hoạch (xử trí)_Prepare_Delivery',            # 16.3% prevalence
            'Nhận định và đánh giá_Patient_Stable',          # 8.7% prevalence
            'Nhận định và đánh giá_Position_Unfavorable',    # 8.8% prevalence
            'Kế hoạch (xử trí)_Prevent_Hemorrhage',          # 10.2% prevalence
            'Kế hoạch (xử trí)_Any_Resuscitation'            # 20.9% prevalence
        ]
    
    # Filter to only available targets
    target_list = [t for t in target_list if t in available_targets]
    
    print(f"\\nTraining models for {len(target_list)} targets:")
    for i, target in enumerate(target_list, 1):
        print(f"  {i:2d}. {target}")
    
    # Results storage
    results_list = []
    
    # Train models for each target
    for i, target in enumerate(target_list, 1):
        print(f"\\n{'='*80}")
        print(f"[{i}/{len(target_list)}] Training: {target}")
        print(f"{'='*80}")
        
        try:
            # Set target and train
            success = rf_classifier.set_target_column(target)
            if not success:
                continue
                
            # Train model
            training_results = rf_classifier.train_model(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced'
            )
            
            if training_results is None:
                print(f"Failed to train model for {target}")
                continue
            
            # Evaluate model
            eval_results = rf_classifier.evaluate_model(save_plots=True)
            
            if eval_results is None:
                continue
            
            # Extract key metrics
            class_report = eval_results['classification_report']
            
            # Store results
            result_row = {
                'target_column': target,
                'category': 'Assessment' if 'Nhận định' in target else 'Treatment',
                'train_accuracy': training_results['train_accuracy'],
                'test_accuracy': training_results['test_accuracy'],
                'auc_score': training_results['auc_score'],
                'cv_mean': training_results['cv_mean'],
                'cv_std': training_results['cv_std'],
                'precision_0': class_report['0']['precision'],
                'recall_0': class_report['0']['recall'],
                'f1_score_0': class_report['0']['f1-score'],
                'precision_1': class_report['1']['precision'],
                'recall_1': class_report['1']['recall'],
                'f1_score_1': class_report['1']['f1-score'],
                'macro_precision': class_report['macro avg']['precision'],
                'macro_recall': class_report['macro avg']['recall'],
                'macro_f1': class_report['macro avg']['f1-score'],
                'weighted_f1': class_report['weighted avg']['f1-score'],
                'n_features': training_results['n_features'],
                'class_0_count': training_results['class_distribution'][0],
                'class_1_count': training_results['class_distribution'][1]
            }
            
            results_list.append(result_row)
            
            print(f"✅ Model completed successfully!")
            print(f"   Test Accuracy: {training_results['test_accuracy']:.4f}")
            if training_results['auc_score']:
                print(f"   AUC Score: {training_results['auc_score']:.4f}")
            print(f"   F1 Score: {result_row['weighted_f1']:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {target}: {str(e)}")
            continue
    
    # Create results DataFrame
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        print(f"\\n{'='*80}")
        print("BATCH TRAINING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"\\nSuccessfully trained {len(results_df)} models")
        print(f"Average test accuracy: {results_df['test_accuracy'].mean():.4f}")
        print(f"Average AUC score: {results_df['auc_score'].dropna().mean():.4f}")
        print(f"Average F1 score: {results_df['weighted_f1'].mean():.4f}")
        
        # Top performers
        print(f"\\nTop 5 Models by Test Accuracy:")
        top_models = results_df.nlargest(5, 'test_accuracy')[['target_column', 'test_accuracy', 'auc_score', 'weighted_f1']]
        for idx, row in top_models.iterrows():
            print(f"  {row['test_accuracy']:.4f} - {row['target_column']}")
        
        # Save results
        if save_results:
            results_file = Path(data_path).parent / "random_forest_batch_results.csv"
            results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
            print(f"\\nResults saved to: {results_file}")
        
        return results_df
    
    else:
        print("\\n❌ No models were successfully trained.")
        return None

def analyze_feature_importance(results_df=None, data_path=None):
    """
    Analyze feature importance across all models.
    
    Parameters:
    results_df (pd.DataFrame): Results from batch training
    data_path (str): Path to data file
    """
    
    if data_path is None:
        return
    
    print("\\nFeature Importance Analysis")
    print("=" * 50)
    
    # This would require storing feature importance from each model
    # For now, print a placeholder
    print("Feature importance analysis would aggregate the top features")
    print("across all trained models to identify the most predictive")
    print("medical variables for different outcomes.")

def main():
    """
    Main function for batch Random Forest training.
    """
    
    # Set up paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    data_path = project_root / "data" / "dataset_long_format_normalized_labeled.csv"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Run batch training
    results = train_multiple_targets(str(data_path))
    
    if results is not None:
        print(f"\\n🎉 Batch training completed successfully!")
        print(f"Trained {len(results)} Random Forest models")
        print(f"Results saved and plots generated")
        
        # Optional: Analyze feature importance
        analyze_feature_importance(results, str(data_path))
    else:
        print("\\n❌ Batch training failed")

if __name__ == "__main__":
    main()
