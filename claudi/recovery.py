# MEMORY-SAFE VERSION - Prevents thread crashes and saves progress
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class MemorySafeMitsuiModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.target_groups = {}
        
    def save_progress(self, filename="mitsui_progress.pkl"):
        """Save current model state"""
        progress_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'target_groups': self.target_groups
        }
        with open(filename, 'wb') as f:
            pickle.dump(progress_data, f)
        print(f"Progress saved to {filename}")
    
    def load_progress(self, filename="mitsui_progress.pkl"):
        """Load previous model state"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                progress_data = pickle.load(f)
            self.models = progress_data['models']
            self.scalers = progress_data['scalers']
            self.feature_selectors = progress_data['feature_selectors']
            self.target_groups = progress_data['target_groups']
            print(f"Progress loaded from {filename}")
            return True
        return False
        
    def smart_feature_engineering(self, df):
        """Memory-efficient feature engineering"""
        print("Creating memory-efficient features...")
        
        features = df.copy()
        numeric_cols = [col for col in features.columns if col != 'date_id']
        print(f"Base features: {len(numeric_cols)}")
        
        # REDUCED FEATURE SET - Focus on quality over quantity
        key_windows = [5, 10, 20]  # Reduced from many windows
        
        # Select most important columns only
        lme_cols = [col for col in numeric_cols if 'LME_' in col][:6]  # Top 6 LME
        us_cols = [col for col in numeric_cols if 'US_Stock_' in col][:8]  # Top 8 US
        jpx_cols = [col for col in numeric_cols if 'JPX_' in col][:4]  # Top 4 JPX
        fx_cols = [col for col in numeric_cols if 'FX_' in col][:10]  # Top 10 FX
        
        important_cols = lme_cols + us_cols + jpx_cols + fx_cols
        print(f"Focusing on {len(important_cols)} key instruments")
        
        # Essential features only
        for col in important_cols:
            if col in features.columns:
                # Moving averages (reduced windows)
                for window in [5, 20]:  # Only 2 windows instead of 6
                    features[f'{col}_sma_{window}'] = features[col].rolling(window).mean()
                    features[f'{col}_vol_{window}'] = features[col].rolling(window).std()
                
                # Key lags only
                for lag in [1, 5]:  # Only 2 lags instead of many
                    features[f'{col}_ret_{lag}'] = features[col].pct_change(lag)
        
        # Group features (memory efficient)
        if len(lme_cols) > 1:
            features['LME_avg'] = features[lme_cols].mean(axis=1)
            
        if len(us_cols) > 1:
            features['US_avg'] = features[us_cols].mean(axis=1)
            
        if len(fx_cols) > 1:
            features['FX_avg'] = features[fx_cols].mean(axis=1)
        
        # Essential ratios
        if 'LME_AH_Close' in features.columns and 'LME_CA_Close' in features.columns:
            features['AH_CA_ratio'] = features['LME_AH_Close'] / (features['LME_CA_Close'] + 1e-8)
            
        if 'LME_PB_Close' in features.columns and 'LME_ZS_Close' in features.columns:
            features['PB_ZS_ratio'] = features['LME_PB_Close'] / (features['LME_ZS_Close'] + 1e-8)
        
        print(f"Total features: {len(features.columns)} (memory-safe)")
        return features
    
    def group_targets_by_lag(self, target_pairs_df):
        """Group targets by lag only (simplified)"""
        
        lag_groups = {1: [], 2: [], 3: [], 4: []}
        
        for i, row in target_pairs_df.iterrows():
            target_name = f"target_{i}"
            lag = row['lag']
            lag_groups[lag].append(target_name)
        
        self.target_groups = {
            'lag_1': lag_groups[1],
            'lag_2': lag_groups[2], 
            'lag_3': lag_groups[3],
            'lag_4': lag_groups[4]
        }
        
        print("Target groups:")
        for group, targets in self.target_groups.items():
            print(f"  {group}: {len(targets)} targets")
    
    def create_memory_safe_models(self):
        """Memory-safe models with n_jobs=1"""
        
        return {
            'lgb_safe': lgb.LGBMRegressor(
                n_estimators=300,  # Reduced
                learning_rate=0.1,
                num_leaves=32,     # Reduced
                feature_fraction=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=1  # CRITICAL: Single thread
            ),
            
            'xgb_safe': xgb.XGBRegressor(
                n_estimators=200,  # Reduced
                learning_rate=0.15,
                max_depth=5,       # Reduced
                random_state=42,
                verbosity=0,
                n_jobs=1  # CRITICAL: Single thread
            ),
            
            'ridge_safe': Ridge(alpha=0.5)  # Always single thread
        }
    
    def train_group_safe(self, X, y, group_name, target_list):
        """Memory-safe training with progress saving"""
        
        print(f"\nTraining {group_name} ({len(target_list)} targets)...")
        
        # Feature selection (aggressive)
        X_clean = X.fillna(method='ffill').fillna(0)
        valid_targets = [t for t in target_list if t in y.columns]
        
        if not valid_targets:
            print(f"No valid targets for {group_name}")
            return {}
            
        y_group = y[valid_targets].fillna(0)
        y_avg = y_group.mean(axis=1) if len(valid_targets) > 1 else y_group.iloc[:, 0]
        
        # Select only top 50 features (memory safe)
        selector = SelectKBest(score_func=f_regression, k=min(50, len(X_clean.columns)))
        try:
            selector.fit(X_clean, y_avg)
            selected_features = X_clean.columns[selector.get_support()].tolist()
        except:
            selected_features = X_clean.columns[:50].tolist()
        
        print(f"  Selected {len(selected_features)} features")
        
        X_selected = X_clean[selected_features]
        
        # Scaling
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Memory-safe models
        models = self.create_memory_safe_models()
        trained_models = {}
        cv_scores = {}
        
        # Train each model individually (memory safe)
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            try:
                # Multi-output with n_jobs=1
                if len(valid_targets) > 1:
                    model = MultiOutputRegressor(model, n_jobs=1)  # CRITICAL: Single thread
                
                # Simple 2-fold CV (memory safe)
                tscv = TimeSeriesSplit(n_splits=2, gap=5)
                fold_scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                    y_train, y_val = y_group.iloc[train_idx], y_group.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    mse = np.mean((y_val.values - y_pred) ** 2)
                    fold_scores.append(mse)
                
                cv_scores[model_name] = np.mean(fold_scores)
                
                # Final fit
                model.fit(X_scaled, y_group)
                trained_models[model_name] = model
                
                print(f"    {model_name}: MSE = {cv_scores[model_name]:.6f}")
                
            except Exception as e:
                print(f"    {model_name} failed: {e}")
                continue
        
        # Store results
        self.models[group_name] = trained_models
        self.scalers[group_name] = scaler
        self.feature_selectors[group_name] = selected_features
        
        # SAVE PROGRESS AFTER EACH GROUP
        self.save_progress()
        
        return cv_scores
    
    def resume_training(self, X, y, start_from_group=None):
        """Resume training from where it left off"""
        
        group_order = ['lag_1', 'lag_2', 'lag_3', 'lag_4']
        results = {}
        
        # Find starting point
        if start_from_group:
            start_idx = group_order.index(start_from_group)
            groups_to_train = group_order[start_idx:]
            print(f"Resuming training from {start_from_group}")
        else:
            # Check which groups are already trained
            groups_to_train = []
            for group in group_order:
                if group not in self.models:
                    groups_to_train.append(group)
            
            if not groups_to_train:
                print("All groups already trained!")
                return {}
            
            print(f"Need to train: {groups_to_train}")
        
        # Train remaining groups
        for group_name in groups_to_train:
            if self.target_groups[group_name]:
                print(f"\n{'='*50}")
                results[group_name] = self.train_group_safe(
                    X, y, group_name, self.target_groups[group_name]
                )
                print(f"✅ {group_name} completed and saved")
        
        return results

def recover_and_continue_training():
    """Recovery function to continue from where it crashed"""
    
    print("🔄 RECOVERY MODE - Continuing from crash...")
    
    # Initialize model
    model = MemorySafeMitsuiModel()
    
    # Try to load previous progress
    if model.load_progress():
        print("✅ Previous progress found!")
        print("Trained groups:", list(model.models.keys()))
    else:
        print("❌ No previous progress found, starting fresh...")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    labels_df = pd.read_csv('train_labels.csv')
    target_pairs_df = pd.read_csv('target_pairs.csv')
    
    # Feature engineering (if needed)
    print("Feature engineering...")
    features_df = model.smart_feature_engineering(train_df)
    
    # Group targets (if needed)
    if not model.target_groups:
        model.group_targets_by_lag(target_pairs_df)
    
    # Resume training
    print("\n" + "="*50)
    print("RESUMING TRAINING...")
    
    # Continue from lag_3 (where it crashed)
    results = model.resume_training(features_df, labels_df, start_from_group='lag_3')
    
    print("\n" + "="*50)
    print("🎉 RECOVERY COMPLETE!")
    
    return model

def run_memory_safe_pipeline():
    """Run the complete memory-safe pipeline"""
    
    print("=== MEMORY-SAFE MITSUI MODEL ===\n")
    
    model = MemorySafeMitsuiModel()
    

    # Load your actual files
    train_df = pd.read_csv(r'E:\kaagle-hackathons\MITSUI&CO_Commodity_Prediction_Challenge\code\ares\MITSUI-CO_Commodity_Prediction_Challenge\data\raw\train.csv')
    labels_df = pd.read_csv(r'E:\kaagle-hackathons\MITSUI&CO_Commodity_Prediction_Challenge\code\ares\MITSUI-CO_Commodity_Prediction_Challenge\data\raw\train_labels.csv')  
    target_pairs_df = pd.read_csv(r'E:\kaagle-hackathons\MITSUI&CO_Commodity_Prediction_Challenge\code\ares\MITSUI-CO_Commodity_Prediction_Challenge\data\raw\target_pairs.csv')
        


    
    # Feature engineering
    features_df = model.smart_feature_engineering(train_df)
    model.group_targets_by_lag(target_pairs_df)
    
    # Train with progress saving
    results = model.resume_training(features_df, labels_df)
    
    print("\n🎉 TRAINING COMPLETE!")
    return model

if __name__ == "__main__":
    
    # Check if we need recovery or fresh start
    if os.path.exists("mitsui_progress.pkl"):
        print("🔄 Found previous progress - running recovery...")
        model = recover_and_continue_training()
    else:
        print("🚀 Starting fresh training...")
        model = run_memory_safe_pipeline()