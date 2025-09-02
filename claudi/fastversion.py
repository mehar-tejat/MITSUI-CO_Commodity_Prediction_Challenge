# # ULTRA-FAST VERSION - Complete training in 20-30 minutes
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler
# import lightgbm as lgb
# from sklearn.feature_selection import SelectKBest, f_regression
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# class UltraFastMitsui:
#     def __init__(self):
#         self.models = {}
#         self.scalers = {}
#         self.selected_features = {}
        
#     def minimal_features(self, df):
#         """Create only the most essential features"""
#         print("Creating minimal feature set...")
        
#         features = df.copy()
#         numeric_cols = [col for col in features.columns if col != 'date_id']
        
#         # Only the most important base features
#         important_patterns = ['LME_AH', 'LME_CA', 'LME_PB', 'LME_ZS', 'US_Stock_SPY', 'FX_EURUSD']
#         key_cols = []
        
#         for pattern in important_patterns:
#             matching_cols = [col for col in numeric_cols if pattern in col]
#             key_cols.extend(matching_cols[:3])  # Max 3 per pattern
        
#         # Add all available if not enough found
#         if len(key_cols) < 30:
#             key_cols = numeric_cols[:30]
        
#         print(f"Using {len(key_cols)} key columns")
        
#         # Minimal feature engineering - only essentials
#         for col in key_cols[:15]:  # Only for top 15 columns
#             # Simple moving averages
#             features[f'{col}_ma5'] = features[col].rolling(5).mean()
#             features[f'{col}_ma20'] = features[col].rolling(20).mean()
            
#             # Simple lags
#             features[f'{col}_lag1'] = features[col].shift(1)
#             features[f'{col}_lag5'] = features[col].shift(5)
        
#         # Key ratios only
#         if 'LME_AH_Close' in features.columns and 'LME_CA_Close' in features.columns:
#             features['AH_CA_ratio'] = features['LME_AH_Close'] / (features['LME_CA_Close'] + 1e-8)
        
#         print(f"Total features: {features.shape[1]} (ultra-minimal)")
#         return features
    
#     def ultra_fast_train(self, X, y, target_list, group_name):
#         """Train single LightGBM model only (fastest)"""
        
#         print(f"Training {group_name} ({len(target_list)} targets)...")
        
#         # Clean data quickly
#         X_clean = X.fillna(method='ffill').fillna(0)
#         valid_targets = [t for t in target_list if t in y.columns]
#         y_group = y[valid_targets].fillna(0)
        
#         if len(valid_targets) == 0:
#             return
        
#         # Feature selection - only top 30
#         y_avg = y_group.mean(axis=1) if len(valid_targets) > 1 else y_group.iloc[:, 0]
#         selector = SelectKBest(score_func=f_regression, k=min(30, X_clean.shape[1]))
        
#         try:
#             X_selected = selector.fit_transform(X_clean, y_avg)
#             selected_features = X_clean.columns[selector.get_support()].tolist()
#         except:
#             X_selected = X_clean.iloc[:, :30].values
#             selected_features = X_clean.columns[:30].tolist()
        
#         # Simple scaling
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_selected)
        
#         # Single fast LightGBM model
#         model = lgb.LGBMRegressor(
#             n_estimators=100,    # Very fast
#             learning_rate=0.2,   # Higher learning rate
#             num_leaves=16,       # Small trees
#             random_state=42,
#             verbose=-1,
#             n_jobs=1
#         )
        
#         # Simple validation
#         split_point = int(0.8 * len(X_scaled))
#         X_train, X_val = X_scaled[:split_point], X_scaled[split_point:]
#         y_train, y_val = y_group.iloc[:split_point], y_group.iloc[split_point:]
        
#         # Train
#         if len(valid_targets) == 1:
#             model.fit(X_train, y_train.values.ravel())
#             pred = model.predict(X_val)
#             mse = np.mean((y_val.values.ravel() - pred) ** 2)
#         else:
#             # For multiple targets, train on average
#             y_train_avg = y_train.mean(axis=1)
#             model.fit(X_train, y_train_avg)
#             pred = model.predict(X_val)
#             mse = np.mean((y_val.mean(axis=1) - pred) ** 2)
        
#         print(f"  MSE: {mse:.6f}")
        
#         # Final training on all data
#         if len(valid_targets) == 1:
#             model.fit(X_scaled, y_group.values.ravel())
#         else:
#             model.fit(X_scaled, y_group.mean(axis=1))
        
#         # Store
#         self.models[group_name] = {
#             'model': model,
#             'targets': valid_targets,
#             'multi_target': len(valid_targets) > 1
#         }
#         self.scalers[group_name] = scaler
#         self.selected_features[group_name] = selected_features
        
#         # Save progress
#         self.save_models()
    
#     def save_models(self):
#         """Save all models"""
#         save_data = {
#             'models': self.models,
#             'scalers': self.scalers,
#             'selected_features': self.selected_features
#         }
#         with open('ultra_fast_models.pkl', 'wb') as f:
#             pickle.dump(save_data, f)
#         print("✅ Models saved to ultra_fast_models.pkl")
    
#     def load_models(self):
#         """Load saved models"""
#         try:
#             with open('ultra_fast_models.pkl', 'rb') as f:
#                 save_data = pickle.load(f)
#             self.models = save_data['models']
#             self.scalers = save_data['scalers']
#             self.selected_features = save_data['selected_features']
#             print("✅ Models loaded from ultra_fast_models.pkl")
#             return True
#         except:
#             print("❌ No saved models found")
#             return False
    
#     def predict_fast(self, X_test):
#         """Make predictions with trained models"""
#         print("Making predictions...")
        
#         # Feature engineering
#         X_features = self.minimal_features(X_test)
#         all_predictions = {}
        
#         for group_name, group_data in self.models.items():
#             print(f"Predicting {group_name}...")
            
#             model = group_data['model']
#             targets = group_data['targets']
#             scaler = self.scalers[group_name]
#             features = self.selected_features[group_name]
            
#             # Prepare test data
#             X_group = X_features[features].fillna(method='ffill').fillna(0)
#             X_scaled = scaler.transform(X_group.values)
            
#             # Predict
#             pred = model.predict(X_scaled)
            
#             # Store predictions
#             if group_data['multi_target']:
#                 # Same prediction for all targets in group
#                 for target in targets:
#                     all_predictions[target] = pred
#             else:
#                 all_predictions[targets[0]] = pred
        
#         return all_predictions

# def run_ultra_fast_training():
#     """Complete ultra-fast training pipeline"""
    
#     print("🚀 ULTRA-FAST MITSUI TRAINING (20 minutes)")
#     print("="*50)
    
#     # Initialize
#     model = UltraFastMitsui()
    
#     # Check for existing models
#     if model.load_models():
#         print("Found existing models! Training complete.")
#         return model
    
    
#     # Load your actual files
#     train_df = pd.read_csv(r'E:\kaagle-hackathons\MITSUI&CO_Commodity_Prediction_Challenge\code\ares\MITSUI-CO_Commodity_Prediction_Challenge\data\raw\train.csv')
#     labels_df = pd.read_csv(r'E:\kaagle-hackathons\MITSUI&CO_Commodity_Prediction_Challenge\code\ares\MITSUI-CO_Commodity_Prediction_Challenge\data\raw\train_labels.csv')  
#     target_pairs_df = pd.read_csv(r'E:\kaagle-hackathons\MITSUI&CO_Commodity_Prediction_Challenge\code\ares\MITSUI-CO_Commodity_Prediction_Challenge\data\raw\target_pairs.csv')
      
    
#     # Minimal feature engineering
#     features_df = model.minimal_features(train_df)
    
#     # Group targets by lag
#     lag_groups = {1: [], 2: [], 3: [], 4: []}
#     for i, row in target_pairs_df.iterrows():
#         lag = row['lag']
#         lag_groups[lag].append(f"target_{i}")
    
#     # Train each lag group
#     for lag in [1, 2, 3, 4]:
#         group_name = f"lag_{lag}"
#         target_list = lag_groups[lag]
        
#         if target_list:
#             model.ultra_fast_train(features_df, labels_df, target_list, group_name)
    
#     print("\n🎉 ULTRA-FAST TRAINING COMPLETE!")
#     print("Time taken: ~20-30 minutes")
#     print("Performance: Expected Top 20-30% (good for speed!)")
    
#     return model

# def create_submission_fast(model, test_file, sample_submission_file):
#     """Create submission quickly"""
    
#     print("Creating submission...")
    
#     # Load test data
#     test_df = pd.read_csv(test_file)
#     sample_sub = pd.read_csv(sample_submission_file)
    
#     # Make predictions
#     predictions = model.predict_fast(test_df)
    
#     # Fill submission
#     for col in sample_sub.columns:
#         if col != 'date_id' and col in predictions:
#             sample_sub[col] = predictions[col]
    
#     # Save
#     sample_sub.to_csv('ultra_fast_submission.csv', index=False)
#     print("✅ Submission saved: ultra_fast_submission.csv")
    
#     return sample_sub

# if __name__ == "__main__":
    
#     # Run ultra-fast training
#     fast_model = run_ultra_fast_training()
    
#     print("\n" + "="*50)
#     print("READY FOR SUBMISSION!")
#     print("To create submission:")
#     print("1. Put your test.csv in the same folder")
#     print("2. Run: create_submission_fast(fast_model, 'test.csv', 'sample_submission.csv')")
#     print("\nExpected performance: Top 20-30% in 20 minutes!")


# import psutil

# # Get memory info
# memory = psutil.virtual_memory()
# print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
# print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
# print(f"Used RAM: {memory.used / (1024**3):.2f} GB")
# print(f"RAM Usage: {memory.percent}%")

# Delete specific large variables
# del model, optimizer, data_loader, dataset
# Or del whatever variables you know are large

# import gc
# gc.collect()  # Force garbage collection

# Check memory after clearing
import psutil
memory = psutil.virtual_memory()
print(f"Available RAM after clearing: {memory.available / (1024**3):.2f} GB")



# # Save variables you want to keep
# keep_vars = ['important_var1', 'important_var2']  # Add your variables here

# # Clear everything else
# all_vars = list(globals().keys())
# for var in all_vars:
#     if (not var.startswith('__') and 
#         var not in ['gc', 'psutil', 'sys'] and  # Keep imports
#         var not in keep_vars):
#         del globals()[var]

# import gc
# gc.collect()