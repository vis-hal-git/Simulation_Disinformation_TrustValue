import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging to replace print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    total_npcs: int = 100000
    total_ticks: int = 20
    conversion_threshold: float = 0.05
    max_trust_delta_per_tick: float = 0.01  # Very small for gradual trust erosion
    age_influence_sigma: float = 15.0
    age_network_sigma: float = 20.0
    max_influence_score: float = 100.0
    max_network_score: float = 100.0
    network_targeting_divisor: int = 5
    
    # Group communication preferences
    num_groups: int = 20
    ingroup_communication_bias: float = 0.7  # 70% chance to communicate within group
    
    random_seed: Optional[int] = 42

@dataclass
class DisinformationEvent:
    """Represents a disinformation event with strength and resistance"""
    strength: float  # S parameter (0-1)
    resistance: float  # R parameter (0-1)
    disinformer_count: int  # Number of disinformers for this specific event

class NPCGenerator:
    """Handles NPC generation with customizable parameters"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
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

class TrustDynamics:
    """Handles trust value calculations and updates"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
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
        # Probability of message reception based on resistance and current trust
        reception_probability = np.clip(100 + (event.resistance - receiver_trust * 100), 0, 100)
        
        # Normalize influence to 0-1 range
        normalized_influence = sender_influence / self.config.max_influence_score
        
        # Group trust modifier - same group messages are more trusted
        modifier = group_trust_modifier if same_group else 1.0
        
        # Calculate trust delta (always negative)
        delta = -(reception_probability / 100.0) * (event.strength - receiver_trust / 4.0) * normalized_influence * modifier
        
        # Apply global trust erosion multiplier
        delta *= trust_erosion_multiplier
        
        # Ensure delta is negative or zero
        return min(0, delta)
    
    def apply_trust_deltas(self, trust_array: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Apply trust deltas with constraints"""
        clipped_deltas = np.clip(deltas, -self.config.max_trust_delta_per_tick, 0)
        new_trust = trust_array + clipped_deltas
        return np.clip(new_trust, 0, 1)

class GroupCommunicationManager:
    """Manages group-based communication patterns"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
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
            ingroup_bias = self.config.ingroup_communication_bias
            
        total_npcs = len(all_groups)
        
        # Calculate how many targets should be from same group
        ingroup_targets = int(num_targets * ingroup_bias)
        outgroup_targets = num_targets - ingroup_targets
        
        # Find same group members (excluding sender)
        same_group_mask = (all_groups == sender_group)
        same_group_ids = np.where(same_group_mask)[0]
        same_group_ids = same_group_ids[same_group_ids != sender_id]
        
        # Find different group members
        diff_group_ids = np.where(~same_group_mask)[0]
        
        targets = []
        same_group_flags = []
        
        # Select ingroup targets
        if len(same_group_ids) > 0 and ingroup_targets > 0:
            selected_ingroup = np.random.choice(
                same_group_ids, 
                size=min(ingroup_targets, len(same_group_ids)), 
                replace=False
            )
            targets.extend(selected_ingroup)
            same_group_flags.extend([True] * len(selected_ingroup))
        
        # Select outgroup targets
        if len(diff_group_ids) > 0 and outgroup_targets > 0:
            selected_outgroup = np.random.choice(
                diff_group_ids, 
                size=min(outgroup_targets, len(diff_group_ids)), 
                replace=False
            )
            targets.extend(selected_outgroup)
            same_group_flags.extend([False] * len(selected_outgroup))
        
        # If we couldn't fill targets due to group constraints, fill randomly
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
    
    def __init__(self, npc_data: Dict[str, np.ndarray], config: SimulationConfig):
        self.trust = npc_data['trust'].copy()
        self.influence = npc_data['influence_score']
        self.network = npc_data['network_score']
        self.group_ids = npc_data['group_id']
        self.num_npcs = len(self.trust)
        self.config = config
        self.trust_dynamics = TrustDynamics(config)
        self.comm_manager = GroupCommunicationManager(config)
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
            num_targets = max(1, int(self.network[sender_id] / self.config.network_targeting_divisor))
            
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
        
        # Apply trust deltas
        old_trust_mean = np.mean(self.trust)
        self.trust = self.trust_dynamics.apply_trust_deltas(self.trust, trust_deltas)
        
        # Update converted NPCs
        newly_converted = set(np.where(self.trust < self.config.conversion_threshold)[0])
        new_conversions = newly_converted - self.converted_npcs
        self.converted_npcs = newly_converted
        
        # Return event processing results
        return {
            "strength": event.strength,
            "resistance": event.resistance,
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
                      **event_processing_kwargs) -> List[Dict]:
        """
        Run simulation with externally provided events
        
        Args:
            external_events: Dictionary mapping tick numbers to lists of events
            **event_processing_kwargs: Additional parameters for event processing
            
        Returns:
            List of tick logs
        """
        logger.info("Starting disinformation simulation with external events...")
        max_tick = max(external_events.keys()) if external_events else 0
        total_ticks = max(self.config.total_ticks, max_tick + 1)
        
        for tick in range(total_ticks):
            daily_events = external_events.get(tick, [])
            self.simulate_tick(tick, daily_events, **event_processing_kwargs)
            time.sleep(0.1)  # Small delay for readability
        
        # Generate final summary
        final_stats = self._generate_final_summary()
        self._log_final_summary(final_stats)
        
        return self.logs
    
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
        # Group statistics
        group_stats = {}
        unique_groups = np.unique(self.group_ids)
        for group_id in unique_groups:
            group_mask = self.group_ids == group_id
            group_trust = self.trust[group_mask]
            group_converted = np.sum(group_trust < self.config.conversion_threshold)
            
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

# Example usage functions (can be removed in production)
def create_sample_events() -> Dict[int, List[DisinformationEvent]]:
    """Example function to create sample disinformation events"""
    return {
        0: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        1: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        2: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        3: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        4: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        5: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        6: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        7: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        8: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        9: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        10: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        11: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        12: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        13: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        14: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
        15: [
            DisinformationEvent(strength=0.69, resistance=0.47, disinformer_count=275),
            DisinformationEvent(strength=0.60, resistance=0.36, disinformer_count=289),
            DisinformationEvent(strength=0.87, resistance=0.34, disinformer_count=111),
            DisinformationEvent(strength=0.48, resistance=0.32, disinformer_count=105),
            DisinformationEvent(strength=0.57, resistance=0.28, disinformer_count=299),
            DisinformationEvent(strength=0.78, resistance=0.40, disinformer_count=198),
            DisinformationEvent(strength=0.71, resistance=0.50, disinformer_count=267),
            DisinformationEvent(strength=0.66, resistance=0.22, disinformer_count=320),
            DisinformationEvent(strength=0.59, resistance=0.35, disinformer_count=190),
            DisinformationEvent(strength=0.74, resistance=0.44, disinformer_count=308),
            DisinformationEvent(strength=0.61, resistance=0.29, disinformer_count=263),
            DisinformationEvent(strength=0.50, resistance=0.42, disinformer_count=173),
            DisinformationEvent(strength=0.68, resistance=0.38, disinformer_count=252),
            DisinformationEvent(strength=0.76, resistance=0.25, disinformer_count=243),
            DisinformationEvent(strength=0.45, resistance=0.19, disinformer_count=337),
        ],
    }

def main():
    """Example usage of the simulation system"""
    
    # Create configuration
    config = SimulationConfig(
        total_npcs=50000,
        total_ticks=30,
        conversion_threshold=0.05,
        num_groups=20,
        ingroup_communication_bias=0.7,
        random_seed=42
    )
    
    # Generate NPCs with custom parameters
    logger.info("Generating NPCs...")
    npc_generator = NPCGenerator(config)
    npc_data = npc_generator.generate_npcs(
        npc_count=config.total_npcs,
        age_based_ratio=0.6,  # 60% age-based, 40% random
        trust_range=(0.3, 0.9),  # Higher initial trust
        num_groups=config.num_groups
    )
    
    # Create external events (in real usage, this would come from external system)
    external_events = create_sample_events()
    
    # Run simulation
    simulator = DisinformationSimulator(npc_data, config)
    logs = simulator.run_simulation(
        external_events,
        trust_erosion_multiplier=0.05,  # Slower trust erosion
        group_trust_modifier=1.5,       # Stronger group effects
        ingroup_bias=0.8                # Higher ingroup preference
    )
    
    # Save logs to file
    with open('external_events_simulation.json', 'w') as f:
        json.dump(logs, f, indent=2)
    
    logger.info("Simulation completed. Logs saved to external_events_simulation.json")

if __name__ == "__main__":
    main()