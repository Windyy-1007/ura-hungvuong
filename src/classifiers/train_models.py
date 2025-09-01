"""
Interactive Random Forest Training Script for Medical Dataset
Allows easy selection and training of models for different target columns.
"""

import sys
import os
from pathlib import Path

# Add the parent directories to path to import our module
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.random_forest import MedicalRandomForestClassifier

def select_target_interactively(rf_classifier):
    """
    Interactive target selection from available target columns.
    
    Parameters:
    rf_classifier: MedicalRandomForestClassifier instance
    
    Returns:
    str: Selected target column name
    """
    targets = rf_classifier.get_available_targets()
    
    print(f"\nAvailable Target Columns ({len(targets)}):")
    print("=" * 50)
    
    # Group targets by category
    assessment_targets = [t for t in targets if 'Nhận định và đánh giá' in t]
    treatment_targets = [t for t in targets if 'Kế hoạch (xử trí)' in t]
    
    print("\nASSESSMENT TARGETS (Nhận định và đánh giá):")
    for i, target in enumerate(assessment_targets, 1):
        short_name = target.replace('Nhận định và đánh giá_', '')
        print(f"  {i:2d}. {short_name}")
    
    print(f"\nTREATMENT TARGETS (Kế hoạch xử trí):")
    start_idx = len(assessment_targets) + 1
    for i, target in enumerate(treatment_targets, start_idx):
        short_name = target.replace('Kế hoạch (xử trí)_', '')
        print(f"  {i:2d}. {short_name}")
    
    print(f"\nEnter the number (1-{len(targets)}) or type 'quit' to exit:")
    
    while True:
        try:
            user_input = input("Selection: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                return None
            
            selection = int(user_input)
            if 1 <= selection <= len(targets):
                selected_target = targets[selection - 1]
                print(f"\nSelected: {selected_target}")
                return selected_target
            else:
                print(f"Please enter a number between 1 and {len(targets)}")
                
        except ValueError:
            print("Please enter a valid number or 'quit'")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def train_selected_model(rf_classifier, target_column):
    """
    Train a Random Forest model for the selected target column.
    
    Parameters:
    rf_classifier: MedicalRandomForestClassifier instance
    target_column: Target column name
    
    Returns:
    dict: Training results
    """
    print(f"\n" + "=" * 70)
    print(f"TRAINING RANDOM FOREST MODEL")
    print(f"Target: {target_column}")
    print("=" * 70)
    
    # Set target column
    if not rf_classifier.set_target_column(target_column):
        return None
    
    # Train model
    print("\n1. Training Random Forest...")
    results = rf_classifier.train_model(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        test_size=0.2
    )
    
    if results is None:
        print("Training failed!")
        return None
    
    # Evaluate model
    print("\n2. Evaluating Model...")
    eval_results = rf_classifier.evaluate_model(save_plots=True)
    
    if eval_results is None:
        print("Evaluation failed!")
        return results
    
    # Display summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Target Column: {target_column}")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"AUC Score: {results['auc_score']:.4f}" if results['auc_score'] else "AUC Score: N/A")
    print(f"Cross-validation: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
    print(f"Number of Features: {results['n_features']}")
    print(f"Class Distribution: {results['class_distribution']}")
    
    # Show top features
    feature_importance = eval_results['feature_importance'][:10]
    print(f"\nTop 10 Important Features:")
    for i, feat in enumerate(feature_importance, 1):
        print(f"  {i:2d}. {feat['feature']}: {feat['importance']:.4f}")
    
    return results

def batch_train_models(rf_classifier, target_list=None):
    """
    Train models for multiple targets in batch.
    
    Parameters:
    rf_classifier: MedicalRandomForestClassifier instance
    target_list: List of target column names. If None, uses common targets.
    
    Returns:
    dict: Results for all trained models
    """
    if target_list is None:
        # Default list of important medical targets
        target_list = [
            'Nhận định và đánh giá_CTG_Group_II',
            'Nhận định và đánh giá_CTG_Group_I', 
            'Nhận định và đánh giá_Patient_Stable',
            'Nhận định và đánh giá_Position_Unfavorable',
            'Kế hoạch (xử trí)_Monitor_Labor',
            'Kế hoạch (xử trí)_Report_Doctor',
            'Kế hoạch (xử trí)_Prepare_Delivery',
            'Kế hoạch (xử trí)_Any_Resuscitation'
        ]
    
    all_results = {}
    available_targets = rf_classifier.get_available_targets()
    
    print(f"BATCH TRAINING: {len(target_list)} Models")
    print("=" * 70)
    
    for i, target in enumerate(target_list, 1):
        if target not in available_targets:
            print(f"{i}. SKIPPED: {target} (not found)")
            continue
        
        print(f"\n{i}. Training: {target}")
        print("-" * 50)
        
        try:
            results = train_selected_model(rf_classifier, target)
            if results:
                all_results[target] = results
                print(f"✓ SUCCESS: {target}")
            else:
                print(f"✗ FAILED: {target}")
                
        except Exception as e:
            print(f"✗ ERROR training {target}: {str(e)}")
    
    # Summary of all results
    print(f"\n" + "=" * 70)
    print("BATCH TRAINING SUMMARY")
    print("=" * 70)
    print(f"Successfully trained: {len(all_results)} models")
    
    if all_results:
        print(f"\nModel Performance Summary:")
        print(f"{'Target':<40} {'Test Acc':<10} {'AUC':<8} {'CV Mean':<8}")
        print("-" * 70)
        
        for target, results in all_results.items():
            short_name = target.split('_', 1)[1] if '_' in target else target
            test_acc = results['test_accuracy']
            auc = results['auc_score'] if results['auc_score'] else 0.0
            cv_mean = results['cv_mean']
            
            print(f"{short_name:<40} {test_acc:<10.3f} {auc:<8.3f} {cv_mean:<8.3f}")
    
    return all_results

def main():
    """
    Main interactive function for training Random Forest models.
    """
    print("Medical Random Forest Training Interface")
    print("=" * 50)
    
    # Initialize classifier
    rf_classifier = MedicalRandomForestClassifier()
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "dataset_long_format_normalized_labeled.csv"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the labeled dataset exists.")
        return
    
    df = rf_classifier.load_data(str(data_path))
    if df is None:
        print("Failed to load dataset!")
        return
    
    print(f"\nDataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
    
    while True:
        print(f"\n" + "=" * 50)
        print("TRAINING OPTIONS")
        print("=" * 50)
        print("1. Interactive Target Selection")
        print("2. Batch Train Common Targets") 
        print("3. Show Available Targets")
        print("4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                # Interactive selection
                target = select_target_interactively(rf_classifier)
                if target:
                    train_selected_model(rf_classifier, target)
                    
            elif choice == '2':
                # Batch training
                print("\nStarting batch training for common medical targets...")
                batch_train_models(rf_classifier)
                
            elif choice == '3':
                # Show targets
                targets = rf_classifier.get_available_targets()
                print(f"\nAll {len(targets)} Available Targets:")
                for i, target in enumerate(targets, 1):
                    print(f"  {i:2d}. {target}")
                    
            elif choice == '4' or choice.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
