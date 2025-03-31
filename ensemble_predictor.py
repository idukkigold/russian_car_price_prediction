import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
from supplemental_english import REGION_CODES, GOVERNMENT_CODES

def extract_region_info(plate):
    """Enhanced region code extraction with more detailed analysis."""
    region_code = plate[-3:]
    is_moscow = any(region_code in codes for codes in REGION_CODES["Moscow"])
    is_spb = any(region_code in codes for codes in REGION_CODES["Saint Petersburg"])
    
    # Calculate region name and tier
    region_name = "Other"
    region_tier = 3  # Default tier for other regions
    for name, codes in REGION_CODES.items():
        if any(region_code in code for code in codes):
            region_name = name
            # Assign tiers based on economic importance
            if name in ["Moscow", "Saint Petersburg"]:
                region_tier = 1
            elif name in ["Moscow Region", "Leningrad Region"]:
                region_tier = 2
            break
    
    # Calculate region development score (1-100)
    development_scores = {
        "Moscow": 100,
        "Saint Petersburg": 95,
        "Moscow Region": 85,
        "Leningrad Region": 80,
        "Other": 60
    }
    development_score = development_scores.get(region_name, 60)
    
    return {
        'region_code': region_code,
        'region_name': region_name,
        'is_moscow': is_moscow,
        'is_spb': is_spb,
        'region_tier': region_tier,
        'development_score': development_score
    }

def extract_plate_patterns(plate):
    """Enhanced pattern detection for license plates."""
    letters = ''.join(c for c in plate if c.isalpha())
    numbers = ''.join(c for c in plate if c.isdigit())
    region_code = numbers[-3:]
    number_part = numbers[:-3] if len(numbers) > 3 else "0"
    
    # Calculate pattern scores
    patterns = {
        'repeating_letters': max(letters.count(c) for c in set(letters)) if letters else 0,
        'repeating_numbers': max(numbers[:-3].count(c) for c in set(numbers[:-3])) if len(numbers) > 3 else 0,
        'sequential_numbers': sum(str(i) + str(i+1) in numbers[:-3] for i in range(9)),
        'palindrome_numbers': 1 if numbers[:-3] == numbers[:-3][::-1] else 0,
        'all_same_letters': 1 if len(set(letters)) == 1 and letters else 0,
        'letter_count': len(letters),
        'number_count': len(numbers) - 3,  # Exclude region code
        'first_letter_value': ord(letters[0]) - ord('A') if letters else 0,
        'last_letter_value': ord(letters[-1]) - ord('A') if letters else 0,
        'numeric_value': int(number_part),
        'has_zeros': 1 if '0' in number_part else 0,
        'has_ones': 1 if '1' in number_part else 0,
        'has_lucky_number': 1 if any(n in number_part for n in ['7', '8', '9']) else 0,
        'number_complexity': len(set(number_part)),  # Unique digits in number
    }
    
    # Calculate prestige score (0-100)
    prestige_score = 0
    if patterns['all_same_letters']:
        prestige_score += 30
    if patterns['repeating_numbers'] >= 2:
        prestige_score += 25
    if patterns['palindrome_numbers']:
        prestige_score += 20
    if patterns['sequential_numbers'] > 0:
        prestige_score += 15
    if patterns['has_lucky_number']:
        prestige_score += 10
    
    patterns['prestige_score'] = min(100, prestige_score)
    return patterns

def extract_features(df):
    """Enhanced feature extraction with focus on important predictors."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['days_until_2025'] = (pd.Timestamp('2025-01-01') - df['date']).dt.days
    
    # Extract all features
    features = []
    region_counts = {}
    plate_number_counts = {}  # Track popularity of number combinations
    
    # First pass to calculate frequencies
    for plate in df['plate']:
        region_code = plate[-3:]
        number_part = ''.join(c for c in plate if c.isdigit())[:-3]
        region_counts[region_code] = region_counts.get(region_code, 0) + 1
        plate_number_counts[number_part] = plate_number_counts.get(number_part, 0) + 1
    
    # Calculate region value metrics
    total_plates = len(df)
    region_value_metrics = {}
    for region, count in region_counts.items():
        frequency = count / total_plates
        # Regions with moderate frequency (not too common, not too rare) might be more valuable
        rarity_score = 1 - abs(0.5 - frequency)  # Score peaks at 0.5 frequency
        region_value_metrics[region] = {
            'count': count,
            'frequency': frequency,
            'rarity_score': rarity_score
        }
    
    # Second pass to extract features
    for plate in df['plate']:
        # Region information
        region_info = extract_region_info(plate)
        region_code = region_info['region_code']
        region_metrics = region_value_metrics[region_code]
        
        # Plate patterns
        patterns = extract_plate_patterns(plate)
        
        # Government plate features
        govt_features = check_government_plate(plate)
        
        # Number popularity score
        number_part = ''.join(c for c in plate if c.isdigit())[:-3]
        number_popularity = plate_number_counts[number_part] / total_plates
        number_rarity_score = 1 - number_popularity  # Rarer numbers might be more valuable
        
        # Combine all features
        feature_dict = {
            **region_info,
            **patterns,
            **govt_features,
            'region_frequency': region_metrics['frequency'],
            'region_rarity_score': region_metrics['rarity_score'],
            'number_popularity': number_popularity,
            'number_rarity_score': number_rarity_score,
            'total_value_score': (patterns['prestige_score'] * 0.4 +
                                region_info['development_score'] * 0.3 +
                                number_rarity_score * 100 * 0.3)  # Weighted combination
        }
        features.append(feature_dict)
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame(features)
    
    # Combine with original features
    df = pd.concat([df, feature_df], axis=1)
    
    # Encode categorical variables
    categorical_columns = ['region_code', 'region_name']
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    return df, le_dict

def check_government_plate(plate):
    """Enhanced government plate checking with more detailed features."""
    letters = ''.join(c for c in plate if c.isalpha())
    numbers = ''.join(c for c in plate if c.isdigit())
    region_code = numbers[-3:]
    number_part = numbers[:-3] if len(numbers) > 3 else "0"
    
    features = {
        'is_govt': False,
        'has_advantage': False,
        'max_significance': 0,
        'description': "Regular plate",
        'is_forbidden': False,
        'dept_type': 'civilian'
    }
    
    for (l_pattern, num_range, r_code), (desc, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        if (letters == l_pattern and 
            int(number_part) >= num_range[0] and 
            int(number_part) <= num_range[1] and 
            region_code == r_code):
            
            features['is_govt'] = True
            features['has_advantage'] = features['has_advantage'] or advantage
            features['max_significance'] = max(features['max_significance'], significance)
            features['description'] = desc
            features['is_forbidden'] = forbidden
            
            # Determine department type
            if 'police' in desc.lower() or 'internal affairs' in desc.lower():
                features['dept_type'] = 'police'
            elif 'military' in desc.lower() or 'defense' in desc.lower():
                features['dept_type'] = 'military'
            elif 'government' in desc.lower() or 'administration' in desc.lower():
                features['dept_type'] = 'government'
            
    return features

class StackingEnsemble:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
            ('gb2', GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=8, random_state=24))
        ]
        self.meta_model = Ridge(alpha=1.0)
        self.base_predictions = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        y_log = np.log1p(y)
        
        # Initialize out-of-fold predictions
        self.base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # Train each base model using cross-validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"\nTraining {name}...")
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                print(f"Fold {fold + 1}/{self.n_splits}")
                model.fit(X_scaled[train_idx], y_log[train_idx])
                self.base_predictions[val_idx, i] = model.predict(X_scaled[val_idx])
        
        # Train meta model
        self.meta_model.fit(self.base_predictions, y_log)
        
        # Retrain base models on full dataset
        for name, model in self.base_models:
            model.fit(X_scaled, y_log)
            
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.column_stack([
            model.predict(X_scaled) for _, model in self.base_models
        ])
        return np.expm1(self.meta_model.predict(meta_features))
    
    def score(self, X, y):
        """Score method required for permutation importance."""
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, squared=False)  # Negative RMSE (higher is better)
    
    def get_feature_importance(self, X, y):
        """Calculate and return feature importance from multiple perspectives."""
        X_scaled = self.scaler.transform(X)
        y_log = np.log1p(y)
        
        importances = {}
        
        # 1. Random Forest Feature Importance
        rf_model = self.base_models[0][1]  # Get the Random Forest model
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 2. Permutation Importance for the ensemble
        perm_importance = permutation_importance(self, X, y, n_repeats=5, random_state=42, scoring='neg_mean_squared_error')
        perm_imp_mean = perm_importance.importances_mean
        ensemble_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_imp_mean
        }).sort_values('importance', ascending=False)
        
        return {
            'random_forest': rf_importance,
            'ensemble': ensemble_importance
        }

def plot_feature_importance(importance_df, title, top_n=20):
    """Plot feature importance."""
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
                      if col.endswith('_encoded') 
                      or col in ['is_moscow', 'is_spb', 'is_govt', 'has_advantage',
                               'max_significance', 'region_popularity', 'repeating_letters',
                               'repeating_numbers', 'sequential_numbers', 'numeric_value',
                               'year', 'month', 'day', 'is_weekend', 'days_until_2025']]
    
    X_train = train_df[feature_columns].values
    y_train = train_df['price'].values
    X_test = test_df[feature_columns].values
    
    # Train model
    print("\nTraining ensemble model...")
    model = StackingEnsemble(n_splits=5)
    model.fit(X_train, y_train, feature_names=feature_columns)
    
    # Calculate feature importance
    print("\nCalculating feature importance...")
    importance_dict = model.get_feature_importance(X_train, y_train)
    
    # Plot and display feature importance
    print("\nTop 10 Most Important Features (Random Forest):")
    print(importance_dict['random_forest'].head(10))
    plot_feature_importance(importance_dict['random_forest'], "Random Forest Feature Importance")
    
    print("\nTop 10 Most Important Features (Ensemble):")
    print(importance_dict['ensemble'].head(10))
    plot_feature_importance(importance_dict['ensemble'], "Ensemble Feature Importance")
    
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
    submission.to_csv('ensemble_submission.csv', index=False)
    print("\nSubmission file created!")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Mean predicted price: {predictions.mean():,.2f}")
    print(f"Median predicted price: {np.median(predictions):,.2f}")
    print(f"Min predicted price: {predictions.min():,.2f}")
    print(f"Max predicted price: {predictions.max():,.2f}")

if __name__ == "__main__":
    main()
