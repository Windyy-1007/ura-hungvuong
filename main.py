"""
URA Medical Prediction System - Main Entry Point
===============================================

This is the main entry point for the URA Medical Prediction System.
A machine learning system for predicting medical treatment recommendations
based on patient vital signs and clinical measurements.

Author: URA Team
Date: September 2025
"""

import sys
import os
from pathlib import Path
import argparse

def main():
    """Main entry point with command options"""
    parser = argparse.ArgumentParser(
        description="URA Medical Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    # Train new model
  python main.py --evaluate                 # Evaluate existing model
  python main.py --predict patient_data.csv # Make predictions
  python main.py --help                     # Show this help
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Train new multi-output model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model performance')
    parser.add_argument('--predict', type=str, metavar='FILE',
                       help='Make predictions on CSV file')
    parser.add_argument('--quick-eval', action='store_true',
                       help='Quick evaluation summary')
    parser.add_argument('--config', type=str, default='config_all_labels.json',
                       help='Configuration file (default: config_all_labels.json)')
    parser.add_argument('--model', type=str,
                       help='Specific model path (auto-detect if not provided)')
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("🏥 URA Medical Prediction System")
    print("=" * 50)
    
    try:
        if args.train:
            train_model(args.config)
        elif args.evaluate:
            evaluate_model(args.model, args.config)
        elif args.quick_eval:
            quick_evaluation()
        elif args.predict:
            make_predictions(args.predict, args.model)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def train_model(config_path):
    """Train a new multi-output model"""
    print(f"🚀 Training new model with config: {config_path}")
    
    # Import and run training
    sys.path.append('src/classifiers')
    from export_multi_output import MultiOutputMedicalExporter
    
    # Set paths
    data_path = "data/dataset_long_format_normalized_labeled.csv"
    models_dir = "models"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the dataset is available.")
        return
    
    # Create exporter and train
    exporter = MultiOutputMedicalExporter(config_path, data_path, models_dir)
    model_path = exporter.export_multi_output_model()
    
    print(f"✅ Model training completed!")
    print(f"📁 Model saved to: {model_path}")

def evaluate_model(model_path, config_path):
    """Evaluate model performance"""
    print(f"📊 Evaluating model performance...")
    
    # Import evaluation
    sys.path.append('src/classifiers')
    from evaluate import ModelEvaluator
    
    # Auto-detect model if not provided
    if not model_path:
        models_dir = Path("models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "multi_output" in d.name]
            if model_dirs:
                model_path = max(model_dirs, key=os.path.getmtime)
            else:
                print("❌ No model found. Please train a model first.")
                return
    
    # Run evaluation
    evaluator = ModelEvaluator(config_path)
    results_df, overall_metrics = evaluator.evaluate_multi_output_model(str(model_path))
    
    # Create evaluation table
    evaluator.create_evaluation_table(results_df, "🎯 Model Evaluation Results")
    
    # Save reports
    csv_file, txt_file = evaluator.save_evaluation_report(
        results_df, overall_metrics, str(model_path), "data"
    )
    
    print(f"\n✅ Evaluation completed!")
    print(f"📊 CSV report: {csv_file}")
    print(f"📄 Detailed report: {txt_file}")

def quick_evaluation():
    """Quick evaluation summary"""
    print(f"Quick evaluation...")
    
    # Run simple evaluation to avoid Unicode issues
    try:
        import subprocess
        result = subprocess.run([sys.executable, "simple_evaluate.py"], 
                              capture_output=True, text=True, encoding='ascii', errors='ignore')
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error in evaluation: {result.stderr}")
    except Exception as e:
        print(f"Error running evaluation: {e}")

def make_predictions(data_file, model_path):
    """Make predictions on new data"""
    print(f"🔮 Making predictions on: {data_file}")
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Auto-detect model if not provided
    if not model_path:
        models_dir = Path("models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "multi_output" in d.name]
            if model_dirs:
                model_path = max(model_dirs, key=os.path.getmtime)
            else:
                print("❌ No model found. Please train a model first.")
                return
    
    # Load predictor
    sys.path.append(str(Path(model_path)))
    from multi_output_predictor import MultiOutputMedicalPredictor
    
    # Make predictions
    predictor = MultiOutputMedicalPredictor(str(model_path))
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    print(f"📋 Loaded {len(data)} patients for prediction")
    
    # Batch predictions
    results = predictor.predict_batch(data, return_probabilities=True)
    
    # Save results
    output_file = data_file.replace('.csv', '_predictions.csv')
    results.to_csv(output_file, index=False)
    
    print(f"✅ Predictions completed!")
    print(f"📄 Results saved to: {output_file}")

def show_system_info():
    """Show system information and status"""
    print("🏥 URA Medical Prediction System")
    print("=" * 50)
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    # Check for required files
    required_files = [
        "config_all_labels.json",
        "data/dataset_long_format_normalized_labeled.csv",
        "src/classifiers/export_multi_output.py",
        "src/classifiers/evaluate.py"
    ]
    
    print(f"\n📋 System Status:")
    for file_path in required_files:
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"{status} {file_path}")
    
    # Check for trained models
    models_dir = Path("models")
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "multi_output" in d.name]
        if model_dirs:
            print(f"\n🤖 Available Models:")
            for model_dir in sorted(model_dirs, key=os.path.getmtime, reverse=True):
                print(f"   📦 {model_dir.name}")
        else:
            print(f"\n⚠️  No trained models found")
    else:
        print(f"\n⚠️  Models directory not found")

if __name__ == "__main__":
    # Show system info if no arguments
    if len(sys.argv) == 1:
        show_system_info()
        print(f"\nUsage: python main.py --help")
    else:
        main()
