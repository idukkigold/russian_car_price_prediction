# Russian Car Plate Prices Prediction

This project implements a machine learning solution for predicting car prices based on license plate numbers and other features in Russia.

## Models Implemented

1. **Hybrid Model (hybrid_predictor.py)**
   - Combines CatBoost and Neural Network
   - Uses SMAPE (Symmetric Mean Absolute Percentage Error) as evaluation metric
   - Features ensemble learning with 5-fold cross-validation

2. **Ensemble Model (ensemble_predictor.py)**
   - Enhanced feature engineering focusing on:
     - Region information and economic importance
     - Plate pattern recognition
     - Value metrics for number popularity and rarity

## Feature Engineering

Key features include:
- Region tiers based on economic importance
- Development scores for regions (1-100)
- Region rarity scores
- Plate pattern detection (prestige scores)
- Lucky numbers and number complexity
- Value metrics for popularity and rarity

## Performance

Best submission achieved a SMAPE score of 47.55 using the ensemble model with enhanced feature engineering.

## Requirements
- Python 3.9+
- TensorFlow
- CatBoost
- pandas
- numpy
- scikit-learn

## Usage

1. Place the training and test data in the root directory
2. Run the hybrid model:
   ```bash
   python hybrid_predictor.py
   ```
3. Run the ensemble model:
   ```bash
   python ensemble_predictor.py
   ```

## Competition Metric
Submissions are evaluated using the Symmetric Mean Absolute Percentage Error (SMAPE):
```
SMAPE = (200% * |actual - predicted| / (|actual| + |predicted|))
```
