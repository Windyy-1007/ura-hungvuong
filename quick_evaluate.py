"""
Quick Model Evaluation Summary
=============================

Easy-to-use script for getting evaluation metrics on your models.
Just run: python quick_evaluate.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from classifiers.evaluate import ModelEvaluator

def quick_evaluate(model_path=None):
    """Quick evaluation of the default multi-output model"""
    
    if model_path is None:
        # Find the latest multi-output model
        models_dir = Path("models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "multi_output" in d.name]
            if model_dirs:
                model_path = max(model_dirs, key=os.path.getmtime)  # Latest by modification time
            else:
                print("❌ No multi-output model found in models/ directory")
                return
        else:
            print("❌ Models directory not found")
            return
    
    print(f"Quick Evaluation: {model_path.name if hasattr(model_path, 'name') else model_path}")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Evaluate model
        results_df, overall_metrics = evaluator.evaluate_multi_output_model(str(model_path))
        
        print(f"\nQUICK SUMMARY")
        print(f"{'='*40}")
        print(f"Overall Accuracy:  {overall_metrics['Average_Accuracy']:.1%}")
        print(f"Overall Precision: {overall_metrics['Average_Precision']:.1%}")
        print(f"Overall Recall:    {overall_metrics['Average_Recall']:.1%}")
        print(f"Overall F1-Score:  {overall_metrics['Average_F1_Score']:.1%}")
        print(f"Overall AUC:       {overall_metrics['Average_AUC']:.1%}")
        print(f"Total Labels:      {overall_metrics['Total_Targets']}")
        print(f"Test Samples:      {overall_metrics['Total_Samples']}")
        
        print(f"\nTOP 5 PERFORMING LABELS")
        print(f"{'-'*40}")
        top_5 = results_df.nlargest(5, 'Accuracy')
        for _, row in top_5.iterrows():
            print(f"{row['Target']:<25} {row['Accuracy']:.1%}")
        
        print(f"\nLOWEST 3 PERFORMING LABELS")
        print(f"{'-'*40}")
        bottom_3 = results_df.nsmallest(3, 'Accuracy')
        for _, row in bottom_3.iterrows():
            print(f"{row['Target']:<25} {row['Accuracy']:.1%}")
        
        return results_df, overall_metrics
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None, None

if __name__ == "__main__":
    quick_evaluate()
