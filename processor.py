import numpy as np
import time
import json
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NPCConfig:
    """Configuration parameters for NPC generation and properties"""
    total_npcs: int = 100000
    age_influence_sigma: float = 15.0
    age_network_sigma: float = 20.0
    max_influence_score: float = 100.0
    max_network_score: float = 100.0
    
    # Group configuration
    num_groups: int = 20
    
    random_seed: Optional[int] = 42

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation dynamics"""
    total_ticks: int = 20
    conversion_threshold: float = 0.05
    max_trust_delta_per_tick: float = 0.04  # 25 people force a point of view
    network_targeting_divisor: int = 5
    
    # Group communication preferences
    ingroup_communication_bias: float = 0.7  # 70% chance to communicate within group

@dataclass
class DisinformationEvent:
    """Represents a disinformation event with strength and reality"""
    strength: float  # S parameter (0-1)
    reality: float  # R parameter (0-1)
    disinformer_count: int  # Number of disinformers for this specific event

class NPCGenerator:
    """Handles NPC generation with customizable parameters"""
    
    def __init__(self, npc_config: NPCConfig):
        self.config = npc_config
        if npc_config.random_seed is not None:
            np.random.seed(npc_config.random_seed)
    
    def _calculate_age_based_scores(self, ages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate influence and network scores based on age using Gaussian distributions"""
        influence_scores = self.config.max_influence_score * np.exp(-((ages - 60) / self.config.age_influence_sigma) ** 2)
        network_scores = self.config.max_network_score * np.exp(-((ages - 60) / self.config.age_network_sigma) ** 2)
        return influence_scores, network_scores
    
    def _assign_groups(self, npc_count: int) -> np.ndarray:
        """Assign NPCs to groups randomly"""
        return np.random.randint(0, self.config.num_groups, npc_count)
    
    def generate_npcs(self, npc_count: Optional[int] = None, 
                     age_based_ratio: float = 0.5,
                     trust_range: Tuple[float, float] = (0.0, 1.0),
                     age_range: Tuple[int, int] = (18, 80),
                     influence_range: Tuple[float, float] = (0.0, 100.0),
                     network_range: Tuple[float, float] = (0.0, 100.0),
                     num_groups: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate NPCs with customizable parameters
        
        Args:
            npc_count: Number of NPCs to generate (uses config if None)
            age_based_ratio: Ratio of NPCs using age-based scoring (0.0-1.0)
            trust_range: Range for initial trust values
            age_range: Range for ages
            influence_range: Range for influence scores
            network_range: Range for network scores
            num_groups: Number of groups (uses config if None)
        """
        if npc_count is None:
            npc_count = self.config.total_npcs
        if num_groups is None:
            num_groups = self.config.num_groups
            
        age_based_count = int(npc_count * age_based_ratio)
        random_score_count = npc_count - age_based_count
        
        # Initialize arrays
        npc_ids = np.arange(1, npc_count + 1)
        trust_values = np.random.uniform(trust_range[0], trust_range[1], npc_count)
        group_ids = np.random.randint(0, num_groups, npc_count)
        
        # Generate age-based NPCs
        if age_based_count > 0:
            ages_first_part = np.random.randint(age_range[0], age_range[1], age_based_count)
            influence_first_part, network_first_part = self._calculate_age_based_scores(ages_first_part)
        else:
            influence_first_part = np.array([])
            network_first_part = np.array([])
        
        # Generate random score NPCs
        if random_score_count > 0:
            influence_second_part = np.random.uniform(influence_range[0], influence_range[1], random_score_count)
            network_second_part = np.random.uniform(network_range[0], network_range[1], random_score_count)
        else:
            influence_second_part = np.array([])
            network_second_part = np.array([])
        
        # Combine both parts
        influence_scores = np.concatenate([influence_first_part, influence_second_part])
        network_scores = np.concatenate([network_first_part, network_second_part])
        
        return {
            'npc_id': npc_ids,
            'trust': trust_values,
            'influence_score': influence_scores,
            'network_score': network_scores,
            'group_id': group_ids
        }

    @staticmethod
    def combine_npc_sets(npc_sets: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Combine multiple NPC sets into a single set with remapped IDs
        
        Args:
            npc_sets: List of NPC dictionaries to combine
            
        Returns:
            Combined NPC set with:
                - Sequential NPC IDs starting from 1
                - Remapped group IDs to avoid overlaps
                - Concatenated attributes (trust, influence, network scores)
        """
        if not npc_sets:
            return {
                'npc_id': np.array([], dtype=int),
                'trust': np.array([], dtype=float),
                'influence_score': np.array([], dtype=float),
                'network_score': np.array([], dtype=float),
                'group_id': np.array([], dtype=int)
            }
        
        # Initialize combined arrays
        combined_trust = []
        combined_influence = []
        combined_network = []
        combined_groups = []
        
        # Track IDs for remapping
        next_npc_id = 1
        group_offset = 0
        
        for npc_set in npc_sets:
            num_npcs = len(npc_set['npc_id'])
            
            # Remap NPC IDs
            new_npc_ids = np.arange(next_npc_id, next_npc_id + num_npcs)
            next_npc_id += num_npcs
            
            # Remap group IDs
            group_shift = group_offset - npc_set['group_id'].min() if num_npcs > 0 else 0
            new_group_ids = npc_set['group_id'] + group_shift
            group_offset = new_group_ids.max() + 1 if num_npcs > 0 else group_offset
            
            # Collect data
            combined_trust.append(npc_set['trust'])
            combined_influence.append(npc_set['influence_score'])
            combined_network.append(npc_set['network_score'])
            combined_groups.append(new_group_ids)
        
        return {
            'npc_id': np.concatenate([np.arange(1, next_npc_id)]),
            'trust': np.concatenate(combined_trust),
            'influence_score': np.concatenate(combined_influence),
            'network_score': np.concatenate(combined_network),
            'group_id': np.concatenate(combined_groups)
        }

class TrustDynamics:
    """Handles trust value calculations and updates"""
    
    def __init__(self, sim_config: SimulationConfig, npc_config: NPCConfig):
        self.sim_config = sim_config
        self.npc_config = npc_config
    
    def compute_trust_delta(self, receiver_trust: float, sender_influence: float, 
                          event: DisinformationEvent, same_group: bool = False,
                          trust_erosion_multiplier: float = 0.1,
                          group_trust_modifier: float = 1.2) -> float:
        """
        Compute trust change for a single interaction
        
        Args:
            receiver_trust: Current trust level of receiver (0-1)
            sender_influence: Influence score of sender (0-100)
            event: Disinformation event parameters
            same_group: Whether sender and receiver are in the same group
            trust_erosion_multiplier: Global multiplier for trust erosion speed
            group_trust_modifier: Multiplier for same-group interactions
        
        Returns:
            Trust delta (always negative or zero)
        """
        reception_probability = np.clip(100 + (event.reality - receiver_trust * 100), 0, 100)
        normalized_influence = sender_influence / self.npc_config.max_influence_score
        modifier = group_trust_modifier if same_group else 1.0
        delta = -(reception_probability / 100.0) * (event.strength - receiver_trust / 4.0) * normalized_influence * modifier
        delta *= trust_erosion_multiplier
        return min(0, delta)
    
    def apply_trust_deltas(self, trust_array: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Apply trust deltas with constraints"""
        clipped_deltas = np.clip(deltas, -self.sim_config.max_trust_delta_per_tick, 0)
        new_trust = trust_array + clipped_deltas
        return np.clip(new_trust, 0, 1)

class GroupCommunicationManager:
    """Manages group-based communication patterns"""
    
    def __init__(self, sim_config: SimulationConfig):
        self.sim_config = sim_config
    
    def get_communication_targets(self, sender_id: int, sender_group: int, 
                                 all_groups: np.ndarray, num_targets: int,
                                 ingroup_bias: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get communication targets with customizable group bias
        
        Args:
            sender_id: ID of the sender
            sender_group: Group ID of the sender
            all_groups: Array of all NPC group IDs
            num_targets: Number of targets to select
            ingroup_bias: Override default ingroup communication bias
            
        Returns:
            Tuple of (target_ids, same_group_flags)
        """
        if ingroup_bias is None:
            ingroup_bias = self.sim_config.ingroup_communication_bias
            
        total_npcs = len(all_groups)
        ingroup_targets = int(num_targets * ingroup_bias)
        outgroup_targets = num_targets - ingroup_targets
        same_group_mask = (all_groups == sender_group)
        same_group_ids = np.where(same_group_mask)[0]
        same_group_ids = same_group_ids[same_group_ids != sender_id]
        diff_group_ids = np.where(~same_group_mask)[0]
        
        targets = []
        same_group_flags = []

        if len(same_group_ids) > 0 and ingroup_targets > 0:
            selected_ingroup = np.random.choice(
                same_group_ids, 
                size=min(ingroup_targets, len(same_group_ids)), 
                replace=False
            )
            targets.extend(selected_ingroup)
            same_group_flags.extend([True] * len(selected_ingroup))
        

        if len(diff_group_ids) > 0 and outgroup_targets > 0:
            selected_outgroup = np.random.choice(
                diff_group_ids, 
                size=min(outgroup_targets, len(diff_group_ids)), 
                replace=False
            )
            targets.extend(selected_outgroup)
            same_group_flags.extend([False] * len(selected_outgroup))
        

        if len(targets) < num_targets:
            remaining_needed = num_targets - len(targets)
            all_available = np.setdiff1d(np.arange(total_npcs), [sender_id] + targets)
            if len(all_available) > 0:
                additional_targets = np.random.choice(
                    all_available, 
                    size=min(remaining_needed, len(all_available)), 
                    replace=False
                )
                targets.extend(additional_targets)
                # Check if additional targets are same group
                for target in additional_targets:
                    same_group_flags.append(all_groups[target] == sender_group)
        
        return np.array(targets), np.array(same_group_flags)

class DisinformationSimulator:
    """Main simulation engine - accepts external disinformation events"""
    
    def __init__(self, npc_data: Dict[str, np.ndarray], sim_config: SimulationConfig, npc_config: NPCConfig):
        self.trust = npc_data['trust'].copy()
        self.influence = npc_data['influence_score']
        self.network = npc_data['network_score']
        self.group_ids = npc_data['group_id']
        self.num_npcs = len(self.trust)
        self.sim_config = sim_config
        self.npc_config = npc_config
        self.trust_dynamics = TrustDynamics(sim_config, npc_config)
        self.comm_manager = GroupCommunicationManager(sim_config)
        self.logs = []
        
        # Track converted NPCs
        self.converted_npcs = set()
        
        logger.info(f"Initialized simulation with {self.num_npcs} NPCs in {len(np.unique(self.group_ids))} groups")
    
    def get_disinformers_for_event(self, event: DisinformationEvent, 
                                  include_converted: bool = True) -> np.ndarray:
        """
        Get disinformers for a specific event
        
        Args:
            event: Disinformation event
            include_converted: Whether to include converted NPCs as disinformers
            
        Returns:
            Array of NPC IDs that will spread disinformation
        """
        disinformers = []
        
        # Include converted NPCs if requested
        if include_converted:
            disinformers.extend(list(self.converted_npcs))
        
        # Add random disinformers up to the event's specified count
        remaining_slots = max(0, event.disinformer_count - len(disinformers))
        if remaining_slots > 0:
            exclude_list = list(self.converted_npcs) if include_converted else []
            available_npcs = np.setdiff1d(np.arange(self.num_npcs), exclude_list)
            if len(available_npcs) > 0:
                additional_disinformers = np.random.choice(
                    available_npcs, 
                    size=min(remaining_slots, len(available_npcs)), 
                    replace=False
                )
                disinformers.extend(additional_disinformers)
        
        return np.array(disinformers)
    
    def process_disinformation_event(self, event: DisinformationEvent, 
                                   include_converted: bool = True,
                                   trust_erosion_multiplier: float = 0.1,
                                   group_trust_modifier: float = 1.2,
                                   ingroup_bias: Optional[float] = None) -> Dict:
        """
        Process a single disinformation event
        
        Args:
            event: Disinformation event to process
            include_converted: Whether converted NPCs participate
            trust_erosion_multiplier: Speed of trust erosion
            group_trust_modifier: Multiplier for same-group interactions
            ingroup_bias: Override default ingroup communication bias
            
        Returns:
            Dictionary with event processing results
        """
        disinformers = self.get_disinformers_for_event(event, include_converted)
        trust_deltas = np.zeros(self.num_npcs)
        interaction_count = 0
        ingroup_interactions = 0
        outgroup_interactions = 0
        
        # Each disinformer spreads to their network
        for sender_id in disinformers:
            sender_group = self.group_ids[sender_id]
            num_targets = max(1, int(self.network[sender_id] / self.sim_config.network_targeting_divisor))
            
            targets, same_group_flags = self.comm_manager.get_communication_targets(
                sender_id, sender_group, self.group_ids, num_targets, ingroup_bias
            )
            
            for target_id, is_same_group in zip(targets, same_group_flags):
                delta = self.trust_dynamics.compute_trust_delta(
                    self.trust[target_id], 
                    self.influence[sender_id], 
                    event,
                    is_same_group,
                    trust_erosion_multiplier,
                    group_trust_modifier
                )
                trust_deltas[target_id] += delta
                interaction_count += 1
                
                if is_same_group:
                    ingroup_interactions += 1
                else:
                    outgroup_interactions += 1
        
        old_trust_mean = np.mean(self.trust)
        self.trust = self.trust_dynamics.apply_trust_deltas(self.trust, trust_deltas)
        
        newly_converted = set(np.where(self.trust < self.sim_config.conversion_threshold)[0])
        new_conversions = newly_converted - self.converted_npcs
        self.converted_npcs = newly_converted
        
        return {
            "strength": event.strength,
            "reality": event.reality,
            "disinformers": len(disinformers),
            "converted_disinformers": len(set(disinformers) & self.converted_npcs),
            "interactions": interaction_count,
            "ingroup_interactions": ingroup_interactions,
            "outgroup_interactions": outgroup_interactions,
            "new_conversions": len(new_conversions),
            "trust_change": float(np.mean(self.trust) - old_trust_mean),
            "trust_delta_stats": {
                "mean": float(np.mean(trust_deltas)),
                "min": float(np.min(trust_deltas)),
                "max": float(np.max(trust_deltas))
            }
        }
    
    def simulate_tick(self, tick: int, daily_events: List[DisinformationEvent],
                     **event_processing_kwargs) -> Dict:
        """
        Simulate one tick with external disinformation events
        
        Args:
            tick: Current tick number
            daily_events: List of disinformation events for this tick
            **event_processing_kwargs: Additional parameters for event processing
            
        Returns:
            Dictionary with tick results
        """
        tick_log = {
            "tick": tick,
            "day": tick + 1,
            "total_events": len(daily_events),
            "total_converted_npcs": len(self.converted_npcs),
            "events": []
        }
        
        # Process each disinformation event
        for event_idx, event in enumerate(daily_events):
            event_result = self.process_disinformation_event(event, **event_processing_kwargs)
            event_result["event_id"] = event_idx
            tick_log["events"].append(event_result)
        
        # Daily summary
        tick_log["daily_summary"] = {
            "avg_trust": float(np.mean(self.trust)),
            "total_converted": len(self.converted_npcs),
            "total_interactions": sum(event["interactions"] for event in tick_log["events"]),
            "total_ingroup_interactions": sum(event["ingroup_interactions"] for event in tick_log["events"]),
            "total_outgroup_interactions": sum(event["outgroup_interactions"] for event in tick_log["events"]),
            "total_new_conversions": sum(event["new_conversions"] for event in tick_log["events"])
        }
        
        self.logs.append(tick_log)
        
        # Log daily summary
        summary = tick_log["daily_summary"]
        logger.info(f"Day {tick + 1}: {len(daily_events)} events, "
                   f"Avg Trust: {summary['avg_trust']:.4f}, "
                   f"Converted: {summary['total_converted']}, "
                   f"New Conversions: {summary['total_new_conversions']}, "
                   f"Interactions: {summary['total_interactions']}")
        
        return tick_log
    
    def run_simulation(self, external_events: Dict[int, List[DisinformationEvent]],
                      **event_processing_kwargs) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Run simulation with externally provided events and export data
        
        Args:
            external_events: Dictionary mapping tick numbers to lists of events
            **event_processing_kwargs: Additional parameters for event processing
            
        Returns:
            Tuple of (tick logs, exported file paths)
        """
        logger.info("Starting disinformation simulation with external events...")
        max_tick = max(external_events.keys()) if external_events else 0
        total_ticks = max(self.sim_config.total_ticks, max_tick + 1)
        
        for tick in range(total_ticks):
            daily_events = external_events.get(tick, [])
            self.simulate_tick(tick, daily_events, **event_processing_kwargs)
            time.sleep(0.1)

        final_stats = self._generate_final_summary()
        self._log_final_summary(final_stats)
        
        return self.logs, final_stats
    
    def export_simulation_data(self, custom_id: str = None, output_dir: str = "simulation_output") -> Dict[str, str]:
        """
        Export comprehensive simulation data to CSV files
        
        Args:
            custom_id: Custom identifier for the simulation
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping data type to filename
        """
        final_stats = self._generate_final_summary()
        exporter = DataExporter(custom_id)
        return exporter.export_simulation_data(self, final_stats, self.logs, output_dir)
    
    def get_current_state(self) -> Dict:
        """Get current simulation state"""
        return {
            "current_trust": self.trust.copy(),
            "converted_npcs": list(self.converted_npcs),
            "avg_trust": float(np.mean(self.trust)),
            "total_converted": len(self.converted_npcs),
            "conversion_rate": len(self.converted_npcs) / self.num_npcs
        }
    
    def _generate_final_summary(self) -> Dict:
        """Generate comprehensive final statistics"""

        group_stats = {}
        unique_groups = np.unique(self.group_ids)
        for group_id in unique_groups:
            group_mask = self.group_ids == group_id
            group_trust = self.trust[group_mask]
            group_converted = np.sum(group_trust < self.sim_config.conversion_threshold)
            
            group_stats[int(group_id)] = {
                "size": int(np.sum(group_mask)),
                "avg_trust": float(np.mean(group_trust)),
                "converted_count": int(group_converted),
                "conversion_rate": float(group_converted / np.sum(group_mask))
            }
        
        return {
            "total_npcs": self.num_npcs,
            "total_groups": len(unique_groups),
            "final_avg_trust": float(np.mean(self.trust)),
            "total_converted": len(self.converted_npcs),
            "overall_conversion_rate": len(self.converted_npcs) / self.num_npcs,
            "group_statistics": group_stats
        }
    
    def _log_final_summary(self, stats: Dict):
        """Log final simulation summary"""
        logger.info("="*60)

class DataExporter:
    """Handles comprehensive data export and CSV generation for analysis"""
    
    def __init__(self, custom_id: str = None):
        self.custom_id = custom_id or f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_filename = f"simulation_{self.custom_id}"
        
    def export_simulation_data(self, simulator: 'DisinformationSimulator', 
                             final_stats: Dict, logs: List[Dict],
                             output_dir: str = "simulation_output") -> Dict[str, str]:
        """
        Export comprehensive simulation data to multiple CSV files
        
        Args:
            simulator: The simulation instance
            final_stats: Final summary statistics
            logs: Complete simulation logs
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping data type to filename
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # 1. NPC Final State Data
        exported_files['npc_final_state'] = self._export_npc_final_state(
            simulator, final_stats, output_dir
        )
        
        # 2. Tick-by-tick Summary Data
        exported_files['tick_summary'] = self._export_tick_summary(
            logs, output_dir
        )
        
        # 3. Event Details Data
        exported_files['event_details'] = self._export_event_details(
            logs, output_dir
        )
        
        # 4. Group Statistics Over Time
        exported_files['group_evolution'] = self._export_group_evolution(
            simulator, logs, output_dir
        )
                
        # 6. Interaction Analysis
        exported_files['interaction_analysis'] = self._export_interaction_analysis(
            logs, output_dir
        )
        
        # 7. Simulation Configuration
        exported_files['simulation_config'] = self._export_simulation_config(
            simulator, output_dir
        )
        
        # 8. Master Summary Report
        exported_files['master_summary'] = self._export_master_summary(
            final_stats, logs, output_dir
        )
        
        return exported_files
    
    def _export_npc_final_state(self, simulator: 'DisinformationSimulator', 
                               final_stats: Dict, output_dir: str) -> str:
        """Export final state of all NPCs"""
        filename = os.path.join(output_dir, f"{self.base_filename}_npc_final_state.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'simulation_id', 'npc_id', 'final_trust', 'initial_trust', 'trust_change',
                'influence_score', 'network_score', 'group_id', 'is_converted',
                'conversion_tick', 'total_interactions_received', 'final_trust_rank',
                'group_avg_trust', 'group_conversion_rate', 'trust_percentile'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Calculate additional metrics
            trust_percentiles = np.percentile(simulator.trust, np.arange(0, 101))
            trust_ranks = np.argsort(np.argsort(simulator.trust))
            
            for i in range(simulator.num_npcs):
                group_id = int(simulator.group_ids[i])
                group_stats = final_stats['group_statistics'][group_id]
                
                # Find conversion tick (simplified - would need tracking in main sim)
                conversion_tick = -1
                if i in simulator.converted_npcs:
                    # Estimate conversion tick (could be improved with actual tracking)
                    conversion_tick = len(simulator.logs) - 1
                
                writer.writerow({
                    'simulation_id': self.custom_id,
                    'npc_id': i + 1,
                    'final_trust': float(simulator.trust[i]),
                    'initial_trust': 0.5,  # Default - could track actual initial
                    'trust_change': float(simulator.trust[i] - 0.5),
                    'influence_score': float(simulator.influence[i]),
                    'network_score': float(simulator.network[i]),
                    'group_id': group_id,
                    'is_converted': i in simulator.converted_npcs,
                    'conversion_tick': conversion_tick,
                    'total_interactions_received': 0,  # Would need tracking
                    'final_trust_rank': int(trust_ranks[i]),
                    'group_avg_trust': group_stats['avg_trust'],
                    'group_conversion_rate': group_stats['conversion_rate'],
                    'trust_percentile': float(np.searchsorted(trust_percentiles, simulator.trust[i]))
                })
        
        return filename
    
    def _export_tick_summary(self, logs: List[Dict], output_dir: str) -> str:
        """Export tick-by-tick summary statistics"""
        filename = os.path.join(output_dir, f"{self.base_filename}_tick_summary.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'simulation_id', 'tick', 'day', 'total_events', 'total_converted_npcs',
                'avg_trust', 'total_interactions', 'total_ingroup_interactions',
                'total_outgroup_interactions', 'total_new_conversions',
                'ingroup_interaction_ratio', 'conversion_rate', 'trust_change_from_previous',
                'events_strength_avg', 'events_reality_avg', 'total_disinformers'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            prev_avg_trust = None
            for log in logs:
                summary = log['daily_summary']
                
                # Calculate event statistics
                events_strength = [e['strength'] for e in log['events']]
                events_reality = [e['reality'] for e in log['events']]
                total_disinformers = sum(e['disinformers'] for e in log['events'])
                
                # Calculate ratios and changes
                total_interactions = summary['total_interactions']
                ingroup_ratio = (summary['total_ingroup_interactions'] / total_interactions 
                               if total_interactions > 0 else 0)
                
                trust_change = (summary['avg_trust'] - prev_avg_trust 
                              if prev_avg_trust is not None else 0)
                prev_avg_trust = summary['avg_trust']
                
                writer.writerow({
                    'simulation_id': self.custom_id,
                    'tick': log['tick'],
                    'day': log['day'],
                    'total_events': log['total_events'],
                    'total_converted_npcs': log['total_converted_npcs'],
                    'avg_trust': summary['avg_trust'],
                    'total_interactions': total_interactions,
                    'total_ingroup_interactions': summary['total_ingroup_interactions'],
                    'total_outgroup_interactions': summary['total_outgroup_interactions'],
                    'total_new_conversions': summary['total_new_conversions'],
                    'ingroup_interaction_ratio': ingroup_ratio,
                    'conversion_rate': log['total_converted_npcs'] / 100000,  # Assuming total NPCs
                    'trust_change_from_previous': trust_change,
                    'events_strength_avg': np.mean(events_strength) if events_strength else 0,
                    'events_reality_avg': np.mean(events_reality) if events_reality else 0,
                    'total_disinformers': total_disinformers
                })
        
        return filename
    
    def _export_event_details(self, logs: List[Dict], output_dir: str) -> str:
        """Export detailed event-level data"""
        filename = os.path.join(output_dir, f"{self.base_filename}_event_details.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'simulation_id', 'tick', 'day', 'event_id', 'strength', 'reality',
                'disinformers', 'converted_disinformers', 'interactions',
                'ingroup_interactions', 'outgroup_interactions', 'new_conversions',
                'trust_change', 'trust_delta_mean', 'trust_delta_min', 'trust_delta_max',
                'effectiveness_score', 'interaction_efficiency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for log in logs:
                for event in log['events']:
                    # Calculate derived metrics
                    effectiveness = (event['new_conversions'] / event['interactions'] 
                                   if event['interactions'] > 0 else 0)
                    efficiency = (abs(event['trust_change']) / event['interactions'] 
                                if event['interactions'] > 0 else 0)
                    
                    writer.writerow({
                        'simulation_id': self.custom_id,
                        'tick': log['tick'],
                        'day': log['day'],
                        'event_id': event['event_id'],
                        'strength': event['strength'],
                        'reality': event['reality'],
                        'disinformers': event['disinformers'],
                        'converted_disinformers': event['converted_disinformers'],
                        'interactions': event['interactions'],
                        'ingroup_interactions': event['ingroup_interactions'],
                        'outgroup_interactions': event['outgroup_interactions'],
                        'new_conversions': event['new_conversions'],
                        'trust_change': event['trust_change'],
                        'trust_delta_mean': event['trust_delta_stats']['mean'],
                        'trust_delta_min': event['trust_delta_stats']['min'],
                        'trust_delta_max': event['trust_delta_stats']['max'],
                        'effectiveness_score': effectiveness,
                        'interaction_efficiency': efficiency
                    })
        
        return filename
    
    def _export_group_evolution(self, simulator: 'DisinformationSimulator', 
                               logs: List[Dict], output_dir: str) -> str:
        """Export group statistics evolution over time"""
        filename = os.path.join(output_dir, f"{self.base_filename}_group_evolution.csv")
        
        # This would require tracking group stats over time in the main simulation
        # For now, we'll export final group stats with tick information
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'simulation_id', 'group_id', 'group_size', 'final_avg_trust',
                'converted_count', 'conversion_rate', 'total_ticks'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            unique_groups = np.unique(simulator.group_ids)
            for group_id in unique_groups:
                group_mask = simulator.group_ids == group_id
                group_trust = simulator.trust[group_mask]
                group_converted = np.sum(group_trust < simulator.sim_config.conversion_threshold)
                group_size = np.sum(group_mask)
                
                writer.writerow({
                    'simulation_id': self.custom_id,
                    'group_id': int(group_id),
                    'group_size': int(group_size),
                    'final_avg_trust': float(np.mean(group_trust)),
                    'converted_count': int(group_converted),
                    'conversion_rate': float(group_converted / group_size),
                    'total_ticks': len(logs)
                })
        
        return filename
        
    def _export_interaction_analysis(self, logs: List[Dict], output_dir: str) -> str:
        """Export interaction pattern analysis"""
        filename = os.path.join(output_dir, f"{self.base_filename}_interaction_analysis.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'simulation_id', 'tick', 'total_interactions', 'ingroup_interactions',
                'outgroup_interactions', 'ingroup_ratio', 'avg_interactions_per_event',
                'interactions_per_disinformer', 'interaction_efficiency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for log in logs:
                summary = log['daily_summary']
                total_interactions = summary['total_interactions']
                total_events = log['total_events']
                total_disinformers = sum(e['disinformers'] for e in log['events'])
                total_conversions = summary['total_new_conversions']
                
                writer.writerow({
                    'simulation_id': self.custom_id,
                    'tick': log['tick'],
                    'total_interactions': total_interactions,
                    'ingroup_interactions': summary['total_ingroup_interactions'],
                    'outgroup_interactions': summary['total_outgroup_interactions'],
                    'ingroup_ratio': (summary['total_ingroup_interactions'] / total_interactions 
                                    if total_interactions > 0 else 0),
                    'avg_interactions_per_event': (total_interactions / total_events 
                                                 if total_events > 0 else 0),
                    'interactions_per_disinformer': (total_interactions / total_disinformers 
                                                   if total_disinformers > 0 else 0),
                    'interaction_efficiency': (total_conversions / total_interactions 
                                             if total_interactions > 0 else 0)
                })
        
        return filename
    
    def _export_simulation_config(self, simulator: 'DisinformationSimulator', 
                                 output_dir: str) -> str:
        """Export simulation configuration parameters"""
        filename = os.path.join(output_dir, f"{self.base_filename}_config.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['simulation_id', 'parameter', 'value', 'category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # NPC Config parameters
            npc_params = {
                'total_npcs': simulator.npc_config.total_npcs,
                'age_influence_sigma': simulator.npc_config.age_influence_sigma,
                'age_network_sigma': simulator.npc_config.age_network_sigma,
                'max_influence_score': simulator.npc_config.max_influence_score,
                'max_network_score': simulator.npc_config.max_network_score,
                'num_groups': simulator.npc_config.num_groups,
                'random_seed': simulator.npc_config.random_seed
            }
            
            # Simulation Config parameters
            sim_params = {
                'total_ticks': simulator.sim_config.total_ticks,
                'conversion_threshold': simulator.sim_config.conversion_threshold,
                'max_trust_delta_per_tick': simulator.sim_config.max_trust_delta_per_tick,
                'network_targeting_divisor': simulator.sim_config.network_targeting_divisor,
                'ingroup_communication_bias': simulator.sim_config.ingroup_communication_bias
            }
            
            for param, value in npc_params.items():
                writer.writerow({
                    'simulation_id': self.custom_id,
                    'parameter': param,
                    'value': str(value),
                    'category': 'NPC_Config'
                })
            
            for param, value in sim_params.items():
                writer.writerow({
                    'simulation_id': self.custom_id,
                    'parameter': param,
                    'value': str(value),
                    'category': 'Simulation_Config'
                })
        
        return filename
    
    def _export_master_summary(self, final_stats: Dict, logs: List[Dict], 
                              output_dir: str) -> str:
        """Export master summary with key metrics"""
        filename = os.path.join(output_dir, f"{self.base_filename}_master_summary.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'simulation_id', 'export_timestamp', 'total_npcs', 'total_groups',
                'total_ticks', 'final_avg_trust', 'total_converted', 'overall_conversion_rate',
                'total_events', 'total_interactions', 'total_ingroup_interactions',
                'total_outgroup_interactions', 'avg_ingroup_ratio', 'most_converted_group',
                'least_converted_group', 'trust_range', 'final_gini_coefficient'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Aggregate statistics
            total_events = sum(log['total_events'] for log in logs)
            total_interactions = sum(log['daily_summary']['total_interactions'] for log in logs)
            total_ingroup = sum(log['daily_summary']['total_ingroup_interactions'] for log in logs)
            total_outgroup = sum(log['daily_summary']['total_outgroup_interactions'] for log in logs)
            
            # Group analysis
            group_rates = [(gid, stats['conversion_rate']) 
                          for gid, stats in final_stats['group_statistics'].items()]
            most_converted = max(group_rates, key=lambda x: x[1])[0]
            least_converted = min(group_rates, key=lambda x: x[1])[0]
            
            writer.writerow({
                'simulation_id': self.custom_id,
                'export_timestamp': datetime.now().isoformat(),
                'total_npcs': final_stats['total_npcs'],
                'total_groups': final_stats['total_groups'],
                'total_ticks': len(logs),
                'final_avg_trust': final_stats['final_avg_trust'],
                'total_converted': final_stats['total_converted'],
                'overall_conversion_rate': final_stats['overall_conversion_rate'],
                'total_events': total_events,
                'total_interactions': total_interactions,
                'total_ingroup_interactions': total_ingroup,
                'total_outgroup_interactions': total_outgroup,
                'avg_ingroup_ratio': (total_ingroup / total_interactions if total_interactions > 0 else 0),
                'most_converted_group': most_converted,
                'least_converted_group': least_converted,
                'trust_range': f"0.0-1.0",  # Could calculate actual range
                'final_gini_coefficient': 0.0  # Would need calculation
            })
        
        return filename
        logger.info("FINAL SIMULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total NPCs: {stats['total_npcs']}")
        logger.info(f"Total Groups: {stats['total_groups']}")
        logger.info(f"Final Average Trust: {stats['final_avg_trust']:.4f}")
        logger.info(f"Total Converted NPCs: {stats['total_converted']}")
        logger.info(f"Overall Conversion Rate: {stats['overall_conversion_rate']:.2%}")
        logger.info("")
        logger.info("GROUP BREAKDOWN:")
        for group_id, group_stat in stats['group_statistics'].items():
            logger.info(f"  Group {group_id}: {group_stat['size']} NPCs, "
                       f"Trust: {group_stat['avg_trust']:.4f}, "
                       f"Converted: {group_stat['converted_count']} "
                       f"({group_stat['conversion_rate']:.1%})")
        logger.info("="*60)