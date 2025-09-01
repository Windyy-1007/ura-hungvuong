"""
Complete Medical Model Training - All Targets
Trains Random Forest models for ALL available target columns and provides comprehensive analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.random_forest import MedicalRandomForestClassifier

def get_target_categories(target_list):
    """
    Categorize targets into different medical categories.
    
    Parameters:
    target_list (list): List of target column names
    
    Returns:
    dict: Categorized targets
    """
    categories = {
        'CTG_Assessment': [],
        'Patient_Condition': [],
        'Labor_Progress': [],
        'Monitoring': [],
        'Resuscitation': [],
        'Delivery_Prep': [],
        'Complications': [],
        'Composite': []
    }
    
    for target in target_list:
        if 'CTG' in target:
            categories['CTG_Assessment'].append(target)
        elif any(word in target for word in ['Patient', 'Stable', 'Pain', 'Fever']):
            categories['Patient_Condition'].append(target)
        elif any(word in target for word in ['Contractions', 'Guidance', 'Multipara', 'Position']):
            categories['Labor_Progress'].append(target)
        elif any(word in target for word in ['Monitor', 'Report', 'Notify', 'Reassess']):
            categories['Monitoring'].append(target)
        elif 'Resuscitation' in target:
            categories['Resuscitation'].append(target)
        elif any(word in target for word in ['Prepare', 'Prevent', 'Delivery']):
            categories['Delivery_Prep'].append(target)
        elif any(word in target for word in ['Hemorrhage', 'Amniotic', 'Pelvis']):
            categories['Complications'].append(target)
        elif any(word in target for word in ['Multiple', 'Any', 'Abnormal']):
            categories['Composite'].append(target)
        else:
            # Default category
            categories['Patient_Condition'].append(target)
    
    return categories

def train_all_targets(data_path, save_detailed_results=True, create_plots=True, min_samples=10, max_imbalance=50):
    """
    Train Random Forest models for ALL available target columns.
    
    Parameters:
    data_path (str): Path to the labeled CSV file
    save_detailed_results (bool): Whether to save detailed results
    create_plots (bool): Whether to create evaluation plots
    min_samples (int): Minimum samples in minority class
    max_imbalance (float): Maximum class imbalance ratio
    
    Returns:
    dict: Complete training results
    """
    
    print("🏥 COMPLETE MEDICAL MODEL TRAINING - ALL TARGETS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize classifier
    rf_classifier = MedicalRandomForestClassifier()
    
    # Load data
    print(f"\\n📊 Loading data from: {data_path}")
    df = rf_classifier.load_data(data_path)
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return None
    
    # Get all available targets
    all_targets = rf_classifier.get_available_targets()
    print(f"\\n🎯 Found {len(all_targets)} target columns to train")
    
    # Categorize targets
    target_categories = get_target_categories(all_targets)
    
    print("\\n📋 Target Categories:")
    for category, targets in target_categories.items():
        if targets:
            print(f"  {category}: {len(targets)} targets")
    
    # Results storage
    all_results = []
    failed_targets = []
    category_performance = {}
    
    start_time = time.time()
    
    # Train models for each target
    for i, target in enumerate(all_targets, 1):
        print(f"\\n{'='*100}")
        print(f"🔄 [{i:2d}/{len(all_targets)}] Training: {target}")
        print(f"{'='*100}")
        
        target_start_time = time.time()
        
        try:
            # Set target and train
            success = rf_classifier.set_target_column(target)
            if not success:
                failed_targets.append((target, "Failed to set target"))
                continue
            
            # Check class distribution first
            class_dist = df[target].value_counts()
            minority_class_size = class_dist.min()
            majority_class_size = class_dist.max()
            imbalance_ratio = majority_class_size / minority_class_size
            
            print(f"📊 Class distribution: {class_dist.to_dict()}")
            print(f"⚖️  Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            # Skip if too imbalanced or too few samples
            if minority_class_size < min_samples:
                print(f"⚠️  Skipping {target}: Too few minority class samples ({minority_class_size})")
                failed_targets.append((target, f"Too few minority samples: {minority_class_size}"))
                continue
            
            if imbalance_ratio > max_imbalance:
                print(f"⚠️  Skipping {target}: Extremely imbalanced ({imbalance_ratio:.1f}:1)")
                failed_targets.append((target, f"Too imbalanced: {imbalance_ratio:.1f}:1"))
                continue
            
            # Train model with appropriate parameters for imbalanced data
            training_results = rf_classifier.train_model(
                n_estimators=150,
                max_depth=25,
                min_samples_split=max(2, minority_class_size // 10),
                min_samples_leaf=max(1, minority_class_size // 20),
                class_weight='balanced',
                test_size=0.2
            )
            
            if training_results is None:
                failed_targets.append((target, "Training failed"))
                continue
            
            # Evaluate model
            eval_results = rf_classifier.evaluate_model(save_plots=create_plots)
            
            if eval_results is None:
                failed_targets.append((target, "Evaluation failed"))
                continue
            
            # Extract detailed metrics
            class_report = eval_results['classification_report']
            
            # Determine target category
            target_category = 'Other'
            for category, targets in target_categories.items():
                if target in targets:
                    target_category = category
                    break
            
            # Calculate additional metrics
            training_time = time.time() - target_start_time
            
            # Store comprehensive results
            result_row = {
                'target_column': target,
                'category': target_category,
                'training_time_seconds': training_time,
                
                # Basic metrics
                'train_accuracy': training_results['train_accuracy'],
                'test_accuracy': training_results['test_accuracy'],
                'auc_score': training_results.get('auc_score'),
                'cv_mean': training_results['cv_mean'],
                'cv_std': training_results['cv_std'],
                
                # Class-specific metrics
                'precision_class_0': class_report['0']['precision'],
                'recall_class_0': class_report['0']['recall'],
                'f1_score_class_0': class_report['0']['f1-score'],
                'support_class_0': class_report['0']['support'],
                
                'precision_class_1': class_report['1']['precision'],
                'recall_class_1': class_report['1']['recall'],
                'f1_score_class_1': class_report['1']['f1-score'],
                'support_class_1': class_report['1']['support'],
                
                # Overall metrics
                'macro_precision': class_report['macro avg']['precision'],
                'macro_recall': class_report['macro avg']['recall'],
                'macro_f1': class_report['macro avg']['f1-score'],
                'weighted_precision': class_report['weighted avg']['precision'],
                'weighted_recall': class_report['weighted avg']['recall'],
                'weighted_f1': class_report['weighted avg']['f1-score'],
                
                # Model characteristics
                'n_features': training_results['n_features'],
                'class_0_count': training_results['class_distribution'][0],
                'class_1_count': training_results['class_distribution'][1],
                'imbalance_ratio': imbalance_ratio,
            }
            
            all_results.append(result_row)
            
            # Update category performance
            if target_category not in category_performance:
                category_performance[target_category] = []
            category_performance[target_category].append(result_row)
            
            print(f"✅ SUCCESS! Training completed in {training_time:.1f}s")
            print(f"   📈 Test Accuracy: {training_results['test_accuracy']:.4f}")
            if training_results.get('auc_score'):
                print(f"   📊 AUC Score: {training_results['auc_score']:.4f}")
            print(f"   🎯 F1 Score: {result_row['weighted_f1']:.4f}")
            
        except Exception as e:
            print(f"❌ ERROR training {target}: {str(e)}")
            failed_targets.append((target, str(e)))
            continue
    
    total_time = time.time() - start_time
    
    # Create comprehensive results
    results_summary = {
        'total_targets': len(all_targets),
        'successful_models': len(all_results),
        'failed_models': len(failed_targets),
        'total_training_time': total_time,
        'results_dataframe': pd.DataFrame(all_results) if all_results else None,
        'failed_targets': failed_targets,
        'category_performance': category_performance
    }
    
    print(f"\\n{'='*100}")
    print("🎉 TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*100}")
    
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"✅ Successfully trained: {len(all_results)}/{len(all_targets)} models")
    print(f"❌ Failed models: {len(failed_targets)}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        print(f"\\n📊 OVERALL PERFORMANCE METRICS:")
        print(f"   Average Test Accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
        print(f"   Average AUC Score: {results_df['auc_score'].dropna().mean():.4f} ± {results_df['auc_score'].dropna().std():.4f}")
        print(f"   Average F1 Score: {results_df['weighted_f1'].mean():.4f} ± {results_df['weighted_f1'].std():.4f}")
        
        print(f"\\n🏆 TOP 10 PERFORMING MODELS:")
        top_models = results_df.nlargest(10, 'test_accuracy')[['target_column', 'test_accuracy', 'auc_score', 'weighted_f1']]
        for idx, row in top_models.iterrows():
            auc_str = f"{row['auc_score']:.4f}" if pd.notna(row['auc_score']) else "N/A"
            print(f"   {row['test_accuracy']:.4f} | AUC: {auc_str} | F1: {row['weighted_f1']:.4f} | {row['target_column']}")
        
        print(f"\\n📈 PERFORMANCE BY CATEGORY:")
        for category, results in category_performance.items():
            if results:
                cat_df = pd.DataFrame(results)
                avg_acc = cat_df['test_accuracy'].mean()
                avg_f1 = cat_df['weighted_f1'].mean()
                print(f"   {category:20s}: {len(results):2d} models | Avg Accuracy: {avg_acc:.4f} | Avg F1: {avg_f1:.4f}")
    
    if failed_targets:
        print(f"\\n⚠️  FAILED MODELS ({len(failed_targets)}):")
        for target, reason in failed_targets[:10]:  # Show first 10 failures
            print(f"   ❌ {target}: {reason}")
        if len(failed_targets) > 10:
            print(f"   ... and {len(failed_targets) - 10} more")
    
    # Save results
    if save_detailed_results and all_results:
        save_comprehensive_results(results_summary, data_path)
    
    return results_summary

def save_comprehensive_results(results_summary, data_path):
    """
    Save comprehensive training results to files.
    
    Parameters:
    results_summary (dict): Complete training results
    data_path (str): Original data path
    """
    
    data_dir = Path(data_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if results_summary['results_dataframe'] is not None:
        # Save detailed results
        results_file = data_dir / f"all_models_results_{timestamp}.csv"
        results_summary['results_dataframe'].to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"\\n💾 Detailed results saved to: {results_file}")
        
        # Save summary report
        report_file = data_dir / f"all_models_summary_{timestamp}.txt"
        create_summary_report(results_summary, report_file)
        print(f"💾 Summary report saved to: {report_file}")
        
        # Save category analysis
        category_file = data_dir / f"category_performance_{timestamp}.csv"
        create_category_analysis(results_summary, category_file)
        print(f"💾 Category analysis saved to: {category_file}")

def create_summary_report(results_summary, report_file):
    """Create a comprehensive text summary report."""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("COMPLETE MEDICAL MODEL TRAINING REPORT\\n")
        f.write("=" * 60 + "\\n\\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Total Targets: {results_summary['total_targets']}\\n")
        f.write(f"Successful Models: {results_summary['successful_models']}\\n")
        f.write(f"Failed Models: {results_summary['failed_models']}\\n")
        f.write(f"Success Rate: {results_summary['successful_models']/results_summary['total_targets']*100:.1f}%\\n")
        f.write(f"Total Training Time: {results_summary['total_training_time']/60:.1f} minutes\\n\\n")
        
        if results_summary['results_dataframe'] is not None:
            df = results_summary['results_dataframe']
            
            f.write("PERFORMANCE SUMMARY\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Average Test Accuracy: {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}\\n")
            f.write(f"Average AUC Score: {df['auc_score'].dropna().mean():.4f} ± {df['auc_score'].dropna().std():.4f}\\n")
            f.write(f"Average F1 Score: {df['weighted_f1'].mean():.4f} ± {df['weighted_f1'].std():.4f}\\n")
            f.write(f"Best Test Accuracy: {df['test_accuracy'].max():.4f}\\n")
            f.write(f"Worst Test Accuracy: {df['test_accuracy'].min():.4f}\\n\\n")
            
            f.write("TOP 15 PERFORMING MODELS\\n")
            f.write("-" * 30 + "\\n")
            top_models = df.nlargest(15, 'test_accuracy')
            for idx, row in top_models.iterrows():
                auc_str = f"{row['auc_score']:.4f}" if pd.notna(row['auc_score']) else "N/A"
                f.write(f"{row['test_accuracy']:.4f} | AUC: {auc_str} | F1: {row['weighted_f1']:.4f} | {row['target_column']}\\n")
            
            f.write("\\nCATEGORY PERFORMANCE\\n")
            f.write("-" * 30 + "\\n")
            for category, results in results_summary['category_performance'].items():
                if results:
                    cat_df = pd.DataFrame(results)
                    f.write(f"{category}: {len(results)} models, Avg Accuracy: {cat_df['test_accuracy'].mean():.4f}, Avg F1: {cat_df['weighted_f1'].mean():.4f}\\n")
        
        if results_summary['failed_targets']:
            f.write("\\nFAILED MODELS\\n")
            f.write("-" * 30 + "\\n")
            for target, reason in results_summary['failed_targets']:
                f.write(f"{target}: {reason}\\n")

def create_category_analysis(results_summary, category_file):
    """Create detailed category performance analysis."""
    
    category_rows = []
    for category, results in results_summary['category_performance'].items():
        if results:
            cat_df = pd.DataFrame(results)
            category_rows.append({
                'category': category,
                'model_count': len(results),
                'avg_test_accuracy': cat_df['test_accuracy'].mean(),
                'std_test_accuracy': cat_df['test_accuracy'].std(),
                'min_test_accuracy': cat_df['test_accuracy'].min(),
                'max_test_accuracy': cat_df['test_accuracy'].max(),
                'avg_auc_score': cat_df['auc_score'].dropna().mean() if not cat_df['auc_score'].dropna().empty else None,
                'avg_f1_score': cat_df['weighted_f1'].mean(),
                'avg_training_time': cat_df['training_time_seconds'].mean(),
                'best_model': cat_df.loc[cat_df['test_accuracy'].idxmax(), 'target_column'],
                'best_accuracy': cat_df['test_accuracy'].max()
            })
    
    if category_rows:
        category_df = pd.DataFrame(category_rows)
        category_df.to_csv(category_file, index=False, encoding='utf-8-sig')

def main():
    """
    Main function to train ALL models.
    """
    
    # Set up paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "dataset_long_format_normalized_labeled.csv"
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    # Create plots directory
    plots_dir = project_root / "data" / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting COMPLETE model training for ALL targets...")
    print(f"📁 Data: {data_path}")
    print(f"📊 Plots: {plots_dir}")
    print(f"\\n⚙️  Training Parameters:")
    print(f"   - Minimum minority class samples: 10")
    print(f"   - Maximum class imbalance ratio: 50:1")
    print(f"   - Random Forest with 150 trees, balanced weights")
    print(f"   - Cross-validation with 5 folds")
    
    # Run complete training
    results = train_all_targets(
        str(data_path), 
        save_detailed_results=True, 
        create_plots=True,
        min_samples=10,
        max_imbalance=50
    )
    
    if results and results['successful_models'] > 0:
        print(f"\\n🎉 MISSION ACCOMPLISHED!")
        print(f"✅ Successfully trained {results['successful_models']}/{results['total_targets']} models")
        print(f"📊 Success rate: {results['successful_models']/results['total_targets']*100:.1f}%")
        print(f"📈 All evaluation plots generated")
        print(f"⏱️  Total time: {results['total_training_time']/60:.1f} minutes")
        
        if results['results_dataframe'] is not None:
            df = results['results_dataframe']
            print(f"\\n🏆 BEST MODEL:")
            best_model = df.loc[df['test_accuracy'].idxmax()]
            print(f"   {best_model['target_column']}")
            print(f"   Accuracy: {best_model['test_accuracy']:.4f}")
            print(f"   AUC: {best_model['auc_score']:.4f}")
            print(f"   F1: {best_model['weighted_f1']:.4f}")
    else:
        print("\\n❌ Training failed or no models were successfully trained")

if __name__ == "__main__":
    main()
