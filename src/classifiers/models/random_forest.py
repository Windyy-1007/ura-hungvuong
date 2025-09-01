import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MedicalRandomForestClassifier:
    """
    Random Forest Classifier for medical assessment and treatment prediction.
    Automatically handles target column selection and hides other target columns during training.
    """
    
    def __init__(self, target_column=None, random_state=42):
        """
        Initialize the Random Forest classifier.
        
        Parameters:
        target_column (str): The target column to predict. If None, will be set later.
        random_state (int): Random state for reproducibility
        """
        self.target_column = target_column
        self.random_state = random_state
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        
    def load_data(self, csv_path, encoding='utf-8-sig'):
        """
        Load the labeled medical dataset.
        
        Parameters:
        csv_path (str): Path to the labeled CSV file
        encoding (str): File encoding
        
        Returns:
        pd.DataFrame: Loaded dataset
        """
        print(f"Loading dataset from: {csv_path}")
        
        try:
            self.df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
            
            # Identify target columns (binary label columns)
            self.target_columns = [col for col in self.df.columns 
                                 if ('Nhận định và đánh giá_' in col or 'Kế hoạch (xử trí)_' in col)]
            
            print(f"Found {len(self.target_columns)} target columns")
            
            # Identify feature columns (exclude original text columns and target columns)
            text_columns = ['Nhận định và đánh giá', 'Kế hoạch (xử trí)']
            self.feature_columns = [col for col in self.df.columns 
                                  if col not in text_columns and col not in self.target_columns]
            
            print(f"Available feature columns: {len(self.feature_columns)}")
            
            return self.df
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def get_available_targets(self):
        """
        Get list of available target columns for prediction.
        
        Returns:
        list: List of available target column names
        """
        if not hasattr(self, 'target_columns'):
            print("Please load data first using load_data()")
            return []
        
        return sorted(self.target_columns)
    
    def set_target_column(self, target_column):
        """
        Set the target column for prediction.
        
        Parameters:
        target_column (str): Name of the target column to predict
        """
        if target_column not in self.target_columns:
            print(f"Error: '{target_column}' is not a valid target column.")
            print("Available targets:")
            for i, col in enumerate(self.get_available_targets(), 1):
                print(f"  {i:2d}. {col}")
            return False
        
        self.target_column = target_column
        print(f"Target column set to: {target_column}")
        return True
    
    def prepare_features(self, handle_missing='median'):
        """
        Prepare features for training by handling missing values and encoding categorical variables.
        
        Parameters:
        handle_missing (str): Strategy for handling missing values ('median', 'mean', 'drop')
        
        Returns:
        tuple: (X, y) prepared features and target
        """
        if self.target_column is None:
            print("Error: Target column not set. Use set_target_column() first.")
            return None, None
        
        print("Preparing features...")
        
        # Get features (excluding all target columns except the one we're predicting)
        other_targets = [col for col in self.target_columns if col != self.target_column]
        X_columns = [col for col in self.feature_columns if col not in other_targets]
        
        X = self.df[X_columns].copy()
        y = self.df[self.target_column].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values in features
        if handle_missing == 'median':
            # Fill numeric columns with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col].fillna(X[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
                
        elif handle_missing == 'mean':
            # Fill numeric columns with mean
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col].fillna(X[col].mean(), inplace=True)
                
            # Fill categorical columns with mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
                
        elif handle_missing == 'drop':
            # Drop rows with missing values
            initial_len = len(X)
            X = X.dropna()
            y = y.loc[X.index]
            print(f"Dropped {initial_len - len(X)} rows with missing values")
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values in target
        if y.isnull().any():
            print(f"Removing {y.isnull().sum()} rows with missing target values")
            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]
        
        print(f"Final prepared data shape: {X.shape}")
        print(f"Final target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, test_size=0.2, n_estimators=100, max_depth=None, 
                   min_samples_split=2, min_samples_leaf=1, class_weight='balanced'):
        """
        Train the Random Forest model.
        
        Parameters:
        test_size (float): Proportion of data to use for testing
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required at a leaf node
        class_weight (str or dict): Strategy for handling class imbalance
        
        Returns:
        dict: Training results and metrics
        """
        if self.target_column is None:
            print("Error: Target column not set. Use set_target_column() first.")
            return None
        
        print(f"Training Random Forest for target: {self.target_column}")
        print("=" * 60)
        
        # Prepare features
        X, y = self.prepare_features()
        if X is None or y is None:
            return None
        
        # Check class distribution
        class_dist = y.value_counts()
        print(f"Class distribution: {class_dist.to_dict()}")
        
        if len(class_dist) < 2:
            print("Error: Target variable has only one class. Cannot train classifier.")
            return None
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Initialize and train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Training Random Forest...")
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = self.rf_model.predict(self.X_train)
        y_pred_test = self.rf_model.predict(self.X_test)
        y_pred_proba_test = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = self.rf_model.score(self.X_train, self.y_train)
        test_accuracy = self.rf_model.score(self.X_test, self.y_test)
        
        try:
            auc_score = roc_auc_score(self.y_test, y_pred_proba_test)
        except:
            auc_score = None
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5)
        
        self.is_trained = True
        
        results = {
            'target_column': self.target_column,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(X.columns, self.rf_model.feature_importances_)),
            'class_distribution': class_dist.to_dict(),
            'n_features': len(X.columns),
            'model_params': self.rf_model.get_params()
        }
        
        print("\nTraining Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        if auc_score:
            print(f"AUC Score: {auc_score:.4f}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def evaluate_model(self, save_plots=True, output_dir=None):
        """
        Evaluate the trained model with detailed metrics and visualizations.
        
        Parameters:
        save_plots (bool): Whether to save plots
        output_dir (str): Directory to save plots
        
        Returns:
        dict: Evaluation metrics
        """
        if not self.is_trained:
            print("Error: Model not trained yet. Use train_model() first.")
            return None
        
        print(f"Evaluating model for: {self.target_column}")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.rf_model.predict(self.X_test)
        y_pred_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        if save_plots:
            self.create_evaluation_plots(y_pred, y_pred_proba, feature_importance, output_dir)
        
        # Calculate additional metrics
        try:
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
        except:
            auc_score = None
        
        evaluation_results = {
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'auc_score': auc_score,
            'test_accuracy': self.rf_model.score(self.X_test, self.y_test)
        }
        
        return evaluation_results
    
    def create_evaluation_plots(self, y_pred, y_pred_proba, feature_importance, output_dir=None):
        """
        Create evaluation plots for the model.
        
        Parameters:
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities
        feature_importance (pd.DataFrame): Feature importance dataframe
        output_dir (str): Directory to save plots
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "data" / "plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Random Forest Evaluation: {self.target_column}', fontsize=16)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Feature Importance
        top_features = feature_importance.head(15)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_title('Top 15 Feature Importance')
        axes[0, 1].set_xlabel('Importance')
        
        # 3. ROC Curve
        try:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
        except:
            axes[1, 0].text(0.5, 0.5, 'ROC Curve not available\n(single class in test set)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('ROC Curve - Not Available')
        
        # 4. Prediction Distribution
        axes[1, 1].hist(y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Predicted Probabilities')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"rf_evaluation_{self.target_column.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {plot_path}")
        plt.close()
    
    def hyperparameter_tuning(self, param_grid=None, cv=5, scoring='roc_auc'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        param_grid (dict): Parameter grid for tuning
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for optimization
        
        Returns:
        dict: Best parameters and scores
        """
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("Error: Please train the model first using train_model()")
            return None
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        print("Performing hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Check if we can use the specified scoring metric
        try:
            grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
        except:
            print(f"Warning: {scoring} not available, using accuracy instead")
            grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        self.rf_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def predict_new_data(self, new_data):
        """
        Make predictions on new data.
        
        Parameters:
        new_data (pd.DataFrame): New data to predict
        
        Returns:
        tuple: (predictions, probabilities)
        """
        if not self.is_trained:
            print("Error: Model not trained yet. Use train_model() first.")
            return None, None
        
        # Prepare the new data (same preprocessing as training data)
        X_new = new_data[self.feature_columns].copy()
        
        # Handle categorical encoding
        categorical_cols = X_new.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                X_new[col] = self.label_encoders[col].transform(X_new[col].astype(str))
        
        # Make predictions
        predictions = self.rf_model.predict(X_new)
        probabilities = self.rf_model.predict_proba(X_new)[:, 1]
        
        return predictions, probabilities

def main():
    """
    Main function to demonstrate Random Forest classification on medical data.
    """
    print("Medical Random Forest Classifier")
    print("=" * 50)
    
    # Initialize classifier
    rf_classifier = MedicalRandomForestClassifier()
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "dataset_long_format_normalized_labeled.csv"
    df = rf_classifier.load_data(str(data_path))
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Show available targets
    targets = rf_classifier.get_available_targets()
    print(f"\nAvailable target columns ({len(targets)}):")
    for i, target in enumerate(targets[:20], 1):  # Show first 20
        print(f"  {i:2d}. {target}")
    
    if len(targets) > 20:
        print(f"  ... and {len(targets) - 20} more")
    
    # Example: Train model for a common target
    example_targets = [
        'Nhận định và đánh giá_CTG_Group_II',
        'Kế hoạch (xử trí)_Monitor_Labor',
        'Nhận định và đánh giá_Patient_Stable'
    ]
    
    for target in example_targets:
        if target in targets:
            print(f"\n" + "=" * 60)
            print(f"Training Random Forest for: {target}")
            print("=" * 60)
            
            # Set target and train
            rf_classifier.set_target_column(target)
            results = rf_classifier.train_model(
                n_estimators=100,
                max_depth=20,
                class_weight='balanced'
            )
            
            if results:
                # Evaluate model
                eval_results = rf_classifier.evaluate_model(save_plots=True)
                
                print(f"\nModel successfully trained and evaluated for: {target}")
                break
    
    print("\nRandom Forest training completed!")

if __name__ == "__main__":
    main()
