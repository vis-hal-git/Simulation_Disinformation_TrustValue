import json
from typing import Dict, List, Optional, Tuple
from processor import (
    NPCConfig,
    SimulationConfig,
    DisinformationEvent,
    NPCGenerator,
    DisinformationSimulator,
    logger
)

def create_sample_events() -> Dict[int, List[DisinformationEvent]]:
    """Create sample repeat events for days 0 to 15"""
    daily_events = [
        DisinformationEvent(strength=0.69, reality=0.47, disinformer_count=275),
        DisinformationEvent(strength=0.60, reality=0.36, disinformer_count=289),
        DisinformationEvent(strength=0.87, reality=0.34, disinformer_count=111),
        DisinformationEvent(strength=0.48, reality=0.32, disinformer_count=105),
        DisinformationEvent(strength=0.57, reality=0.28, disinformer_count=299),
        DisinformationEvent(strength=0.78, reality=0.40, disinformer_count=198),
        DisinformationEvent(strength=0.71, reality=0.50, disinformer_count=267),
        DisinformationEvent(strength=0.66, reality=0.22, disinformer_count=320),
        DisinformationEvent(strength=0.59, reality=0.35, disinformer_count=190),
        DisinformationEvent(strength=0.74, reality=0.44, disinformer_count=308),
        DisinformationEvent(strength=0.61, reality=0.29, disinformer_count=263),
        DisinformationEvent(strength=0.50, reality=0.42, disinformer_count=173),
        DisinformationEvent(strength=0.68, reality=0.38, disinformer_count=252),
        DisinformationEvent(strength=0.76, reality=0.25, disinformer_count=243),
        DisinformationEvent(strength=0.45, reality=0.19, disinformer_count=337),
    ]
    return {day: daily_events for day in range(14)}

def main():
    # Define NPC configuration with all available parameters
    npc_config = NPCConfig(
        total_npcs=50000,
        age_influence_sigma=15.0,    # default ~15.0
        age_network_sigma=20.0,      # default ~20.0
        max_influence_score=100.0,   # max influence score cap
        max_network_score=100.0,     # max network score cap
        num_groups=10,               # number of groups
        random_seed=42               # deterministic behavior
    )

    # Define Simulation configuration with all available parameters
    sim_config = SimulationConfig(
        total_ticks=14,                 # Number of simulation ticks (days)
        conversion_threshold=0.05,      # NPC trust threshold for conversion
        max_trust_delta_per_tick=0.07,  # max trust change applied per tick per NPC
        network_targeting_divisor=5,    # governs number of targets per sender: network_score/divisor
        ingroup_communication_bias=0.7  # probability bias to communicate within own group
    )

    # Create NPC generator
    generator = NPCGenerator(npc_config)

    # Generate NPCs with all optional parameters supported by generate_npcs()
    npcs = generator.generate_npcs(
        npc_count=npc_config.total_npcs,          # Number of NPCs (default: npc_config.total_npcs)
        age_based_ratio=0.6,                       # Ratio of NPCs with age-based scoring (default: 0.5)
        trust_range=(0.3, 0.9),                    # Initial trust range (default (0, 1))
        age_range=(18, 80),                        # Age range for age-based NPCs (default (18, 80))
        influence_range=(0.0, 100.0),              # Influence scores range for random NPCs (default (0, 100))
        network_range=(0.0, 100.0),                # Network scores range for random NPCs (default (0, 100))
        num_groups=npc_config.num_groups           # Number of groups (default: npc_config.num_groups)
    )

    # Create the simulator instance with NPC and simulation configs
    simulator = DisinformationSimulator(npcs, sim_config, npc_config)

    # Create sample external events (dict keyed by tick)
    external_events = create_sample_events()

    # Run simulation with all supported optional event processing parameters
    logs = simulator.run_simulation(
        external_events,
        trust_erosion_multiplier=0.05,  # Multiplier controlling speed of trust erosion (default ~0.1)
        group_trust_modifier=1.5,       # Multiplier for same-group trust effect (default ~1.2)
        ingroup_bias=0.8                # Overrides ingroup communication bias parameter in sim_config
    )

    # Get final summary stats (optional)
    final_stats = simulator._generate_final_summary()

    # Export simulation data with custom ID to directory "results"
    exported_files = simulator.export_simulation_data(
        custom_id="experiment_001",
        output_dir="results"
    )

    print("Exported files:", exported_files)

    # Optionally save logs as JSON
    with open("results/experiment_001_logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    logger.info("Simulation completed, logs and data exported.")

if __name__ == "__main__":
    main()