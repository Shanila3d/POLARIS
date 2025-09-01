import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import json

class PolarisBackend:
    """Backend engine for POLARIS dashboard"""
    
    def __init__(self):
        self.data = {}
        self.models = {}
        self.cache = {}
        self.var_models = {}
        self.load_assets()
        
    def load_assets(self):
        """Load all necessary data and models"""
        
        print("Loading POLARIS assets...")
        
        # Load data
        self.data['integrated'] = pd.read_parquet("data/processed/polaris_integrated_phase2.parquet")
        self.data['base_predictions'] = pd.read_parquet("data/processed/base_predictions.parquet")
        
        # Load models
        model_dir = Path("models/trained")
        for model_file in model_dir.glob("*.pkl"):
            if 'var' not in model_file.stem:
                self.models[model_file.stem] = joblib.load(model_file)
        
        # Load features
        with open("data/processed/model_features.pkl", "rb") as f:
            self.features = pickle.load(f)
        
        # Load scenario cache
        with open("data/cache/scenario_cache.pkl", "rb") as f:
            self.cache = pickle.load(f)
        
        # Load VAR results
        with open("models/trained/var_results.json", "r") as f:
            self.var_results = json.load(f)
        
        print(f"✓ Loaded {len(self.models)} models, {len(self.cache)} cached scenarios")
    
    def get_current_state(self, sector: str = 'all') -> Dict:
        """Get current ESG state for sector"""
        
        data = self.data['integrated']
        
        # Filter to most recent quarter
        latest_data = data[data['year'] == data['year'].max()]
        latest_quarter = latest_data[latest_data['quarter'] == latest_data['quarter'].max()]
        
        if sector != 'all':
            if sector == 'Financial':
                latest_quarter = latest_quarter[latest_quarter['is_financial'] == 1]
            else:
                latest_quarter = latest_quarter[latest_quarter['is_financial'] == 0]
        
        state = {
            'avg_esg_score': latest_quarter['ESG Score_quarterly'].mean(),
            'avg_jobs': latest_quarter['total_postings'].mean(),
            'avg_esg_jobs': latest_quarter['esg_jobs'].mean(),
            'n_companies': latest_quarter['Instrument'].nunique(),
            'quarter': f"Q{latest_quarter['quarter'].iloc[0]} {latest_quarter['year'].iloc[0]}"
        }
        
        return state
    
    def apply_live_scenario(self, 
                           scenario_type: str,
                           magnitude: float,
                           timing: str,
                           sector: str) -> pd.DataFrame:
        """Apply scenario in real-time"""
        
        # Check cache first
        cache_key = f"{scenario_type}_{magnitude}_{timing}_{sector=='Financial'}"
        if cache_key in self.cache:
            return self.cache[cache_key]['predictions']
        
        # Otherwise compute live
        data = self.data['integrated'].copy()
        
        # Filter by sector if specified
        if sector != 'Both':
            if sector == 'Financial':
                data = data[data['is_financial'] == 1]
            else:
                data = data[data['is_financial'] == 0]
        
        # Apply shock based on timing
        timing_map = {
            'Immediate': [1],
            'Phased': [1, 2, 3, 4],
            'Delayed': [3, 4]
        }
        quarters = timing_map.get(timing, [1])
        
        # Define shocks
        shock_effects = self._get_shock_effects(scenario_type, magnitude)
        
        # Apply to future quarters
        for q in quarters:
            future_mask = (data['year'] >= 2024) & (data['quarter'] >= q)
            for feature, effect in shock_effects.items():
                if feature in data.columns:
                    data.loc[future_mask, feature] *= (1 + effect)
        
        # Predict with shocked data
        predictions = self._generate_predictions(data)
        
        return predictions
    
    def _get_shock_effects(self, scenario_type: str, magnitude: float) -> Dict:
        """Get shock effects for scenario"""
        
        effects = {
            'carbon_tax': {
                'Regulatory_Pressure_Score': magnitude * 0.5,
                'esg_jobs': magnitude * 0.2
            },
            'esg_mandate': {
                'Regulatory_Pressure_Score': magnitude * 0.3,
                'esg_jobs': magnitude * 0.4
            },
            'talent_crisis': {
                'esg_jobs': -magnitude * 0.5,
                'total_postings': -magnitude * 0.3
            },
            'supply_shock': {
                'ESG Score_quarterly': -magnitude * 0.2
            },
            'green_transition': {
                'esg_jobs': magnitude * 0.6,
                'ESG Score_quarterly': magnitude * 0.1
            }
        }
        
        return effects.get(scenario_type, {})
    
    def _generate_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from data"""
        
        predictions = []
        
        for model_name, model in self.models.items():
            # Determine sector
            if 'financial' in model_name:
                sector_data = data[data['is_financial'] == 1]
            else:
                sector_data = data[data['is_financial'] == 0]
            
            if len(sector_data) > 0:
                X = sector_data[self.features].fillna(0)
                y_pred = model.predict(X, num_iteration=model.best_iteration)
                
                pred_df = pd.DataFrame({
                    'year': sector_data['year'].values,
                    'quarter': sector_data['quarter'].values,
                    'prediction': y_pred,
                    'model': model_name
                })
                predictions.append(pred_df)
        
        return pd.concat(predictions) if predictions else pd.DataFrame()
    
    def get_scenario_comparison(self, scenarios: List[Dict]) -> pd.DataFrame:
        """Compare multiple scenarios"""
        
        results = []
        
        for scenario in scenarios:
            preds = self.apply_live_scenario(
                scenario['type'],
                scenario['magnitude'],
                scenario['timing'],
                scenario['sector']
            )
            preds['scenario_name'] = scenario['name']
            results.append(preds)
        
        return pd.concat(results) if results else pd.DataFrame()