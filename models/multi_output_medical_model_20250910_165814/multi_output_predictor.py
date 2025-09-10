"""
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
