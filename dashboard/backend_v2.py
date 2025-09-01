import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import json

class PolarisBackendV2:
    """Updated backend with fixed models and real calculations"""
    
    def __init__(self):
        self.data = {}
        self.models = {}
        self.scalers = {}
        self.cache = {}
        self.load_assets()
        
    def load_assets(self):
        """Load fixed models and scalers"""
        
        print("Loading POLARIS v2 assets...")
        
        # Load data
        self.data['integrated'] = pd.read_parquet("data/processed/polaris_integrated_phase2.parquet")
        self.data['base_predictions'] = pd.read_parquet("data/processed/base_predictions.parquet")
        
        # Load FIXED models and scalers
        model_dir = Path("models/fixed")
        for model_file in model_dir.glob("*_fixed.pkl"):
            model_name = model_file.stem.replace('_fixed', '')
            self.models[model_name] = joblib.load(model_file)
            
            # Load corresponding scaler
            sector = model_name.split('_')[0]
            horizon = '1q' if '1q' in model_name else '4q'
            scaler_path = model_dir / f"{sector}_scaler_{horizon}.pkl"
            if scaler_path.exists():
                self.scalers[model_name] = joblib.load(scaler_path)
        
        # Load fixed features
        with open("data/processed/model_features_fixed.pkl", "rb") as f:
            self.features = pickle.load(f)
        
        # Load scenario cache
        cache_path = Path("data/cache/scenario_cache.pkl")
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self.cache = pickle.load(f)
        
        print(f"✓ Loaded {len(self.models)} fixed models with scalers")

    def get_current_state(self, sector: str = 'all') -> Dict:
        """Get current ESG state for sector"""
        
        data = self.data['integrated']
        
        # Filter to most recent quarter
        latest_data = data[data['year'] == data['year'].max()]
        latest_quarter = latest_data[latest_data['quarter'] == latest_data['quarter'].max()]
        
        if sector != 'all':
            if sector == 'Financial':
                latest_quarter = latest_quarter[latest_quarter['is_financial'] == 1]
            elif sector == 'Non-Financial':
                latest_quarter = latest_quarter[latest_quarter['is_financial'] == 0]
        
        state = {
            'avg_esg_score': latest_quarter['ESG Score_quarterly'].mean(),
            'avg_jobs': latest_quarter['total_postings'].mean() if 'total_postings' in latest_quarter.columns else 0,
            'avg_esg_jobs': latest_quarter['esg_jobs'].mean() if 'esg_jobs' in latest_quarter.columns else 0,
            'n_companies': latest_quarter['Instrument'].nunique(),
            'quarter': f"Q{latest_quarter['quarter'].iloc[0] if not latest_quarter.empty else 4} {latest_quarter['year'].iloc[0] if not latest_quarter.empty else 2024}"
        }
        
        return state
    
    def clean_data(self, X):
        """Clean data for prediction"""
        X = X.replace([np.inf, -np.inf], np.nan)
        
        for col in X.columns:
            if X[col].isna().all():
                X[col] = 0
            elif X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Clip outliers
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32']:
                q01 = X[col].quantile(0.01)
                q99 = X[col].quantile(0.99)
                X[col] = X[col].clip(lower=q01, upper=q99)
        
        return X
    
    def apply_live_scenario(self, 
                           scenario_type: str,
                           magnitude: float,
                           timing: str,
                           sector: str) -> pd.DataFrame:
        """Apply scenario with fixed models"""
        
        data = self.data['integrated'].copy()
        
        # Filter by sector
        if sector == 'Financial':
            data = data[data['is_financial'] == 1]
        elif sector == 'Non-Financial':
            data = data[data['is_financial'] == 0]
        
        # Apply shock
        shock_effects = self._get_shock_effects(scenario_type, magnitude)
        
        timing_map = {
            'Immediate': [1],
            'Phased': [1, 2, 3, 4],
            'Delayed': [3, 4]
        }
        quarters = timing_map.get(timing, [1])
        
        for q in quarters:
            future_mask = (data['year'] >= 2024) & (data['quarter'] >= q)
            for feature, effect in shock_effects.items():
                if feature in data.columns:
                    data.loc[future_mask, feature] *= (1 + effect)
        
        # Generate predictions with proper scaling
        predictions = self._generate_predictions_v2(data)
        
        return predictions
    
    def _generate_predictions_v2(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using fixed models with scaling"""
        
        predictions = []
        
        for model_name, model in self.models.items():
            # Determine sector
            if 'financial' in model_name:
                sector_data = data[data['is_financial'] == 1] if 'is_financial' in data.columns else data
            else:
                sector_data = data[data['is_financial'] == 0] if 'is_financial' in data.columns else data
            
            if len(sector_data) > 0:
                # Clean and scale data
                X = sector_data[self.features].copy()
                X = self.clean_data(X)
                
                # Apply scaler if available
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                else:
                    X_scaled = X
                
                # Predict
                y_pred = model.predict(X_scaled, num_iteration=model.best_iteration)
                
                pred_df = pd.DataFrame({
                    'year': sector_data['year'].values,
                    'quarter': sector_data['quarter'].values,
                    'prediction': y_pred,
                    'model': model_name,
                    'sector': 'Financial' if 'financial' in model_name else 'Non-Financial'
                })
                predictions.append(pred_df)
        
        return pd.concat(predictions) if predictions else pd.DataFrame()
    
    def calculate_real_sector_impacts(self, 
                                     scenario_predictions: pd.DataFrame,
                                     baseline_predictions: pd.DataFrame) -> Dict:
        """Calculate actual differential impacts"""
        
        results = {}
        
        # Baseline by sector
        baseline_financial = baseline_predictions[
            baseline_predictions['model'].str.contains('financial')
        ]['pred_target_esg_1q'].mean() if 'pred_target_esg_1q' in baseline_predictions.columns else 50
        
        baseline_non_financial = baseline_predictions[
            baseline_predictions['model'].str.contains('non-financial')
        ]['pred_target_esg_1q'].mean() if 'pred_target_esg_1q' in baseline_predictions.columns else 50
        
        # Scenario predictions by sector
        scenario_financial = scenario_predictions[
            scenario_predictions['sector'] == 'Financial'
        ]['prediction'].mean() if not scenario_predictions.empty else baseline_financial
        
        scenario_non_financial = scenario_predictions[
            scenario_predictions['sector'] == 'Non-Financial'
        ]['prediction'].mean() if not scenario_predictions.empty else baseline_non_financial
        
        # Calculate impacts
        results['Financial'] = {
            'baseline': baseline_financial,
            'scenario': scenario_financial,
            'absolute_impact': scenario_financial - baseline_financial,
            'relative_impact': ((scenario_financial - baseline_financial) / baseline_financial * 100) if baseline_financial != 0 else 0
        }
        
        results['Non-Financial'] = {
            'baseline': baseline_non_financial,
            'scenario': scenario_non_financial,
            'absolute_impact': scenario_non_financial - baseline_non_financial,
            'relative_impact': ((scenario_non_financial - baseline_non_financial) / baseline_non_financial * 100) if baseline_non_financial != 0 else 0
        }
        
        return results
    
    def _get_shock_effects(self, scenario_type: str, magnitude: float) -> Dict:
        """Differentiated shock effects by scenario"""
        
        effects = {
            'carbon_tax': {
                'Regulatory_Pressure_Score': magnitude * 0.5,
                'esg_jobs': magnitude * 0.2,
                'ESG Score_quarterly': magnitude * -0.1  # Carbon tax initially reduces scores
            },
            'esg_mandate': {
                'Regulatory_Pressure_Score': magnitude * 0.3,
                'esg_jobs': magnitude * 0.4,
                'total_postings': magnitude * 0.1
            },
            'talent_crisis': {
                'esg_jobs': -magnitude * 0.5,
                'total_postings': -magnitude * 0.3,
                'senior_ratio': -magnitude * 0.2
            },
            'supply_shock': {
                'ESG Score_quarterly': -magnitude * 0.2,
                'total_postings': -magnitude * 0.1
            },
            'green_transition': {
                'esg_jobs': magnitude * 0.6,
                'ESG Score_quarterly': magnitude * 0.15,
                'job_quality_index': magnitude * 0.3
            }
        }
        
        return effects.get(scenario_type, {})