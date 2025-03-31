import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
from supplemental_english import REGION_CODES, GOVERNMENT_CODES

def extract_region_info(plate):
    """Extract region code and check if it's a special region."""
    region_code = plate[-3:]  # Last 3 digits
    is_moscow = any(region_code in codes for codes in REGION_CODES["Moscow"])
    is_spb = any(region_code in codes for codes in REGION_CODES["Saint Petersburg"])
    
    # Find the region name
    region_name = None
    for name, codes in REGION_CODES.items():
        if any(region_code in code for code in codes):
            region_name = name
            break
    
    return region_code, is_moscow, is_spb, region_name

def check_government_plate(plate):
    """Check if the plate is a government plate and get its significance."""
    letters = ''.join(c for c in plate if c.isalpha())
    numbers = ''.join(c for c in plate if c.isdigit())
    region_code = numbers[-3:]
    number_part = numbers[:-3] if len(numbers) > 3 else "0"
    
    max_significance = 0
    is_govt = False
    has_advantage = False
    description = "Regular plate"
    
    for (l_pattern, num_range, r_code), (desc, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        if (letters == l_pattern and 
            int(number_part) >= num_range[0] and 
            int(number_part) <= num_range[1] and 
            region_code == r_code):
            is_govt = True
            has_advantage = has_advantage or advantage
            max_significance = max(max_significance, significance)
            description = desc
    
    return is_govt, has_advantage, max_significance, description

def extract_features(df):
    """Extract all features from the dataset."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Extract plate features
    features = []
    for plate in df['plate']:
        region_code, is_moscow, is_spb, region_name = extract_region_info(plate)
        is_govt, has_advantage, significance, govt_desc = check_government_plate(plate)
        
        # Count special characters in plate
        letters = ''.join(c for c in plate if c.isalpha())
        numbers = ''.join(c for c in plate if c.isdigit())
        
        # Check for special patterns
        has_repeating_numbers = any(numbers.count(str(i)) > 1 for i in range(10))
        has_sequential_numbers = any(str(i) + str(i+1) in numbers for i in range(9))
        all_same_letters = len(set(letters)) == 1 if letters else False
        
        features.append({
            'region_code': region_code,
            'region_name': region_name if region_name else "Unknown",
            'is_moscow': is_moscow,
            'is_spb': is_spb,
            'is_govt': is_govt,
            'has_advantage': has_advantage,
            'significance': significance,
            'govt_desc': govt_desc,
            'letters': len(letters),
            'numbers': len(numbers),
            'has_repeating_numbers': has_repeating_numbers,
            'has_sequential_numbers': has_sequential_numbers,
            'all_same_letters': all_same_letters,
            'total_chars': len(plate)
        })
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame(features)
    
    # Combine with original features
    df = pd.concat([df, feature_df], axis=1)
    
    # Encode categorical variables
    categorical_columns = ['region_code', 'region_name', 'govt_desc']
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    return df, le_dict

def train_model(train_df):
    """Train the model and return it."""
    # Prepare features
    features = [
        'year', 'month', 'day', 'dayofweek', 'is_weekend', 'quarter',
        'region_code_encoded', 'region_name_encoded', 'govt_desc_encoded',
        'is_moscow', 'is_spb', 'is_govt', 'has_advantage', 'significance',
        'letters', 'numbers', 'has_repeating_numbers', 'has_sequential_numbers',
        'all_same_letters', 'total_chars'
    ]
    
    X = train_df[features]
    y = train_df['price']
    
    # Log transform the target variable
    y_log = np.log1p(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Initialize and train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    
    # Train the model
    model.fit(X_scaled, y_log)
    
    # Calculate and print cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y_log, cv=5, scoring='neg_mean_absolute_error')
    print("\nCross-validation MAE scores (log scale):")
    print(f"Mean MAE: {-cv_scores.mean():,.4f}")
    print(f"Std MAE: {cv_scores.std():,.4f}")
    
    return model, features, scaler

def main():
    # Load data
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
        # Fill any new categories with -1
        test_df[f'{col}_encoded'] = test_df[f'{col}_encoded'].fillna(-1)
    
    # Train model
    print("Training model...")
    model, features, scaler = train_model(train_df)
    
    # Scale test features
    X_test = test_df[features]
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    predictions_log = model.predict(X_test_scaled)
    
    # Transform predictions back to original scale
    predictions = np.expm1(predictions_log)
    
    # Ensure no negative predictions
    predictions = np.maximum(predictions, 0)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'price': predictions.round(0)  # Round to nearest integer
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("Submission file created!")
    
    # Print some statistics
    print("\nPrediction Statistics:")
    print(f"Mean predicted price: {predictions.mean():,.2f}")
    print(f"Median predicted price: {np.median(predictions):,.2f}")
    print(f"Min predicted price: {predictions.min():,.2f}")
    print(f"Max predicted price: {predictions.max():,.2f}")

if __name__ == "__main__":
    main()
