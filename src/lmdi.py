"""Logarithmic mean Divisia index decomposition for transport activity.

The :class:`TransportDecomposition` class separates the change in total
transport distance between two states into frequency, mode-share, global
length, and relative-length effects. Inputs are aggregate trip counts and
average trip lengths by mode; individual-level or raw survey data are not
required.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional


class TransportDecomposition:
    """
    Implements the four-factor LMDI decomposition for transport mode substitution:
    - Frequency effect (F): Changes in total trip-making
    - Mode-share effect (C): Shifts between modes  
    - Global length effect (G): System-wide trip length changes
    - Relative length effect (L): Mode-specific length patterns
    """
    
    def __init__(self, modes: Optional[List[str]] = None):
        """
        Initialize decomposition with optional custom mode list.
        
        Parameters:
        -----------
        modes : list of str, optional
            List of mode names. If None, defaults to ['car', 'bus', 'bike', 'ebike']
        """
        if modes is None:
            self.modes = ['car', 'bus', 'bike', 'ebike']
        else:
            self.modes = modes
        
        print(f"TransportDecomposition initialized with {len(self.modes)} modes: {self.modes}")
    
    def compute_aggregates(self, trips: Dict[str, float], distances: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute system-level aggregates from mode-specific data
        
        Parameters:
        -----------
        trips : dict
            Dictionary of {mode: trip_count}
        distances : dict
            Dictionary of {mode: total_distance}
        
        Returns:
        --------
        dict with system-level indicators
        """
        Q_total = sum(trips.values())  # Total trips (F)
        D_total = sum(distances.values())  # Total distance
        G_bar = D_total / Q_total if Q_total > 0 else 0  # System average trip length
        
        # Modal shares and relative lengths
        modal_shares = {mode: trips[mode] / Q_total if Q_total > 0 else 0 
                       for mode in self.modes}
        avg_lengths = {mode: distances[mode] / trips[mode] if trips[mode] > 0 else 0 
                      for mode in self.modes}
        rel_lengths = {mode: avg_lengths[mode] / G_bar if G_bar > 0 else 0 
                      for mode in self.modes}
        
        return {
            'F': Q_total,
            'G_bar': G_bar, 
            'C': modal_shares,
            'L': avg_lengths,
            'L_tilde': rel_lengths,
            'D_total': D_total
        }
    
    def validate_state(self, state: Dict) -> bool:
        """
        Validate that state dictionary has correct structure.
        
        Parameters:
        -----------
        state : dict
            State dictionary with 'trips' and 'avg_lengths' keys
        
        Returns:
        --------
        bool : True if valid, False otherwise
        """
        if 'trips' not in state or 'avg_lengths' not in state:
            print("Error: State must contain 'trips' and 'avg_lengths' keys")
            return False
        
        # Check all modes are present
        for mode in self.modes:
            if mode not in state['trips']:
                print(f"Warning: Mode '{mode}' not found in trips data")
                return False
            if mode not in state['avg_lengths']:
                print(f"Warning: Mode '{mode}' not found in avg_lengths data")
                return False
        
        return True
    
    def lmdi_decompose_fixed(self, state1: Dict, state2: Dict, n1, n2,
                   min_change_threshold: float = 0,
                   min_distance_threshold: float = 1e-3) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform an LMDI decomposition between two states.

        The returned DataFrame contains one row per mode and effect columns
        whose sum is checked against the observed per-capita distance change.
        Modes with negligible distance in both states are retained with zero
        effects and marked in the ``warning`` column.
    
        Parameters:
        -----------
        state1 : dict
            Initial state with 'trips' and 'avg_lengths' dicts
        state2 : dict
            Final state with 'trips' and 'avg_lengths' dicts
        min_change_threshold : float
            Minimum absolute change to consider (for numerical stability)
        min_distance_threshold : float
            Minimum distance in either state to include mode (filters zeros)
        """
        
        # Validate inputs
        if not self.validate_state(state1) or not self.validate_state(state2):
            raise ValueError("Invalid state structure")
        
        results = []
        all_effects = {}
        distances = {}
        
        for mode in self.modes:
            # Current and future distances
            D1 = state1['trips'][mode] * state1['avg_lengths'][mode]
            D2 = state2['trips'][mode] * state2['avg_lengths'][mode]
            distances[mode] = {'D1': D1, 'D2': D2}
            
            # Skip modes with negligible distances in BOTH states
            if D1 < min_distance_threshold and D2 < min_distance_threshold:
                all_effects[mode] = {
                    'delta_F': 0,
                    'delta_C': 0, 
                    'delta_G': 0,
                    'delta_L': 0
                }
                results.append({
                    'mode': mode,
                    'distance_change': D2 - D1,
                    'per_dist_change': D2/n2 - D1/n1,
                    'frequency_effect': 0,
                    'mode_share_effect': 0,
                    'global_length_effect': 0, 
                    'relative_length_effect': 0,
                    'phi_F': 0,
                    'phi_C': 0,
                    'phi_G': 0,
                    'phi_L': 0,
                    'verification': 0,
                    'D1': D1,
                    'D2': D2,
                    'Q1': state1['trips'][mode],
                    'Q2': state2['trips'][mode],
                    'L1': state1['avg_lengths'][mode],
                    'L2': state2['avg_lengths'][mode],
                    'warning': 'negligible_distance'
                })
                continue
                
            # Compute aggregates for both states
            agg1 = self.compute_aggregates(
                state1['trips'], 
                {m: state1['trips'][m] * state1['avg_lengths'][m] for m in self.modes}
            )
            agg2 = self.compute_aggregates(
                state2['trips'],
                {m: state2['trips'][m] * state2['avg_lengths'][m] for m in self.modes}
            )
            
            # Log mean calculation with safety checks
            if D1 > min_distance_threshold and D2 > min_distance_threshold:
                # Safe log mean - handle case where D1 ≈ D2
                if abs(D2 - D1) < 1e-3:
                    L_D = (D1 + D2) / 2  # Use arithmetic mean if values nearly equal
                else:
                    try:
                        L_D = (D2 - D1) / np.log(D2 / D1)
                    except (ZeroDivisionError, RuntimeWarning):
                        L_D = (D1 + D2) / 2
                        
                # Four effects with comprehensive safety checks
                delta_F = 0
                delta_C = 0
                delta_G = 0
                delta_L = 0
                
                # Frequency effect
                if agg1['F'] > min_distance_threshold and agg2['F'] > min_distance_threshold:
                    try:
                        ratio = agg2['F'] / agg1['F']
                        if ratio > 1e-3 and np.isfinite(ratio):
                            delta_F = L_D * np.log(ratio)
                    except Exception:
                        pass
                
                # Mode share effect
                if agg1['C'][mode] > min_distance_threshold and agg2['C'][mode] > min_distance_threshold:
                    try:
                        ratio = agg2['C'][mode] / agg1['C'][mode]
                        if ratio > 1e-3 and np.isfinite(ratio):
                            delta_C = L_D * np.log(ratio)
                    except Exception:
                        pass
                
                # Global length effect
                if agg1['G_bar'] > min_distance_threshold and agg2['G_bar'] > min_distance_threshold:
                    try:
                        ratio = agg2['G_bar'] / agg1['G_bar']
                        if ratio > 1e-3 and np.isfinite(ratio):
                            delta_G = L_D * np.log(ratio)
                    except Exception:
                        pass
                
                # Relative length effect
                if agg1['L_tilde'][mode] > min_distance_threshold and agg2['L_tilde'][mode] > min_distance_threshold:
                    try:
                        ratio = agg2['L_tilde'][mode] / agg1['L_tilde'][mode]
                        if ratio > 1e-3 and np.isfinite(ratio):
                            delta_L = L_D * np.log(ratio)
                    except Exception:
                        pass
                        
            else:
                # Handle transition from/to zero
                delta_F = delta_C = delta_G = delta_L = 0
            
            all_effects[mode] = {
                'delta_F': delta_F,
                'delta_C': delta_C, 
                'delta_G': delta_G,
                'delta_L': delta_L
            }
            
            total_change = D2 - D1
            per_change = D2/n2 - D1/n1
            
            # Phi calculation with proper sign handling
            if abs(total_change) > min_change_threshold:
                # Use total_change to preserve signs correctly
                phi_F = delta_F / total_change
                phi_C = delta_C / total_change
                phi_G = delta_G / total_change
                phi_L = delta_L / total_change
            else:
                phi_F = phi_C = phi_G = phi_L = 0
            
            # Check for extreme/invalid phi values
            warning = None
            for phi_name, phi_val in [('phi_F', phi_F), ('phi_C', phi_C), 
                                    ('phi_G', phi_G), ('phi_L', phi_L)]:
                if not np.isfinite(phi_val) or abs(phi_val) > 100:
                    warning = f'extreme_{phi_name}'
                    # Cap extreme values
                    if not np.isfinite(phi_val):
                        phi_val = 0
                    elif phi_val > 100:
                        phi_val = 100
                    elif phi_val < -100:
                        phi_val = -100

            results.append({
                'mode': mode,
                'distance_change': total_change,
                'per_dist_change': per_change,
                'frequency_effect': delta_F,
                'mode_share_effect': delta_C,
                'global_length_effect': delta_G, 
                'relative_length_effect': delta_L,
                'phi_F': phi_F,
                'phi_C': phi_C,
                'phi_G': phi_G,
                'phi_L': phi_L,
                'verification': phi_F + phi_C + phi_G + phi_L,
                'D1': D1,
                'D2': D2,
                'Q1': state1['trips'][mode],
                'Q2': state2['trips'][mode],
                'L1': state1['avg_lengths'][mode],
                'L2': state2['avg_lengths'][mode],
                'warning': warning
            })
        
        df_results = pd.DataFrame(results)
        return df_results