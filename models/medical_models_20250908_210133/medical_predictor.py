"""
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
