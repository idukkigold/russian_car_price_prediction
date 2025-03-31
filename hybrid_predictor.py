import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
from supplemental_english import REGION_CODES, GOVERNMENT_CODES

# Reuse the enhanced feature engineering from ensemble_predictor.py
from ensemble_predictor import extract_region_info, extract_plate_patterns, check_government_plate, extract_features

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def smape_keras(y_true, y_pred):
    """SMAPE loss function for Keras"""
    return tf.reduce_mean(200 * tf.abs(y_pred - y_true) / (tf.abs(y_true) + tf.abs(y_pred)))

class NeuralNetwork:
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=smape_keras  # Use SMAPE as loss function
        )
        
    def fit(self, X, y, validation_data=None):
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        return self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

class HybridEnsemble:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.base_models = []
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_catboost(self):
        return CatBoostRegressor(
            iterations=200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            loss_function='MAPE',  # Use MAPE as it's similar to SMAPE
            eval_metric='MAPE',    # Monitor MAPE during training
            verbose=False
        )
    
    def fit(self, X, y, feature_names=None):
        print("\nTraining Hybrid Ensemble...")
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize models for each fold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_smapes = []
        # Train base models
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            print(f"\nFold {fold + 1}/{self.n_splits}")
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train CatBoost
            print("Training CatBoost...")
            cat_model = self.create_catboost()
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            # Train Neural Network
            print("Training Neural Network...")
            nn_model = NeuralNetwork(X_train.shape[1])
            nn_model.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            self.base_models.append({
                'cat': cat_model,
                'nn': nn_model
            })
            
            # Print validation scores
            cat_pred = cat_model.predict(X_val)
            nn_pred = nn_model.predict(X_val)
            cat_smape = smape(y_val, cat_pred)
            nn_smape = smape(y_val, nn_pred)
            print(f"CatBoost SMAPE: {cat_smape:.4f}")
            print(f"Neural Network SMAPE: {nn_smape:.4f}")
            fold_smapes.append(min(cat_smape, nn_smape))
        
        print(f"\nAverage fold SMAPE: {np.mean(fold_smapes):.4f}")
        
        # Train meta-model (another CatBoost)
        print("\nTraining meta-model...")
        meta_features = np.zeros((X.shape[0], 2))  # 2 predictions per sample
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_val = X_scaled[val_idx]
            models = self.base_models[fold]
            
            # Get predictions from both models
            cat_pred = models['cat'].predict(X_val)
            nn_pred = models['nn'].predict(X_val)
            
            meta_features[val_idx, 0] = cat_pred
            meta_features[val_idx, 1] = nn_pred
        
        self.meta_model = self.create_catboost()
        self.meta_model.fit(meta_features, y)
        
        # Print final ensemble score
        ensemble_pred = self.predict(X)
        final_smape = smape(y, ensemble_pred)
        print(f"\nFinal Ensemble SMAPE: {final_smape:.4f}")
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.zeros((X.shape[0], 2))
        
        # Get predictions from all base models
        for models in self.base_models:
            cat_pred = models['cat'].predict(X_scaled)
            nn_pred = models['nn'].predict(X_scaled)
            
            meta_features[:, 0] += cat_pred / len(self.base_models)
            meta_features[:, 1] += nn_pred / len(self.base_models)
        
        # Make final prediction using meta-model
        return self.meta_model.predict(meta_features)
    
    def get_feature_importance(self, X, y):
        """Get feature importance from CatBoost models."""
        importances = np.zeros(len(self.feature_names))
        
        # Average feature importance across all CatBoost models
        for models in self.base_models:
            importances += models['cat'].get_feature_importance()
        
        importances /= len(self.base_models)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

def plot_feature_importance(importance_df, title, top_n=20):
    plt.figure(figsize=(12, 8))
    importance_df.head(top_n).plot(x='feature', y='importance', kind='barh')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Process data
    print("Processing training data...")
    train_df, le_dict = extract_features(train_df)
    
    print("Processing test data...")
    test_df, _ = extract_features(test_df)
    
    # Ensure test set uses the same encoding as training
    for col, le in le_dict.items():
        test_df[f'{col}_encoded'] = test_df[col].map(dict(zip(le.classes_, le.transform(le.classes_))))
        test_df[f'{col}_encoded'] = test_df[f'{col}_encoded'].fillna(-1)
    
    # Prepare features
    feature_columns = [col for col in train_df.columns 
                      if col not in ['id', 'plate', 'date', 'price', 'region_code', 'region_name', 'description', 'dept_type']]
    
    X_train = train_df[feature_columns].values
    y_train = train_df['price'].values
    X_test = test_df[feature_columns].values
    
    # Train model
    model = HybridEnsemble(n_splits=5)
    model.fit(X_train, y_train, feature_names=feature_columns)
    
    # Calculate feature importance
    print("\nCalculating feature importance...")
    importance_df = model.get_feature_importance(X_train, y_train)
    
    # Plot and display feature importance
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    plot_feature_importance(importance_df, "Hybrid Model Feature Importance")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Ensure no negative predictions
    predictions = np.maximum(predictions, 0)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'price': predictions.round(0)
    })
    
    # Save submission
    submission.to_csv('hybrid_submission.csv', index=False)
    print("\nSubmission file created!")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Mean predicted price: {predictions.mean():,.2f}")
    print(f"Median predicted price: {np.median(predictions):,.2f}")
    print(f"Min predicted price: {predictions.min():,.2f}")
    print(f"Max predicted price: {predictions.max():,.2f}")

if __name__ == "__main__":
    main()
