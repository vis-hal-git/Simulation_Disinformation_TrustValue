import numpy as np
import time
import json

# ---- NPC Generation ---- #
def generate_npc(count, include_age=True):
    npc_ids = np.arange(1, count + 1)
    trust_values = np.random.uniform(0, 1, count)

    if include_age:
        ages = np.random.randint(18, 80, count)
        return {
            'npc_id': npc_ids,
            'trust': trust_values,
            'age': ages
        }
    else:
        influential_scores = np.random.uniform(0, 100, count)
        network_scores = np.random.uniform(0, 100, count)
        return {
            'npc_id': npc_ids,
            'trust': trust_values,
            'influential_score': influential_scores,
            'network_score': network_scores
        }

def generate_npcs(count):
    npcs_with_age = generate_npc(count // 2, include_age=True)
    npcs_with_scores = generate_npc(count - count // 2, include_age=False)
    return [npcs_with_age, npcs_with_scores]

# ---- Processor ---- #
class Processor:
    def __init__(self, unit_time=10):
        self.unit_time = unit_time

    def calculate_scores_based_on_age(self, age):
        I_max = 100
        N_max = 100
        sigma = 15
        sigma_prime = 20
        f_age = np.exp(-((age - 60) / sigma) ** 2)
        g_age = np.exp(-((age - 60) / sigma_prime) ** 2)
        return I_max * f_age, N_max * g_age

    def load_npcs(self, all_npcs):
        npc_id = np.array([])
        trust = np.array([])
        influencial_score = np.array([])
        network_score = np.array([])

        for npc_group in all_npcs:
            npc_id = np.concatenate([npc_id, npc_group['npc_id']])
            trust = np.concatenate([trust, npc_group['trust']])

            if 'age' in npc_group:
                calc_influential, calc_network = self.calculate_scores_based_on_age(npc_group['age'])
                influencial_score = np.concatenate([influencial_score, calc_influential])
                network_score = np.concatenate([network_score, calc_network])
            else:
                influencial_score = np.concatenate([influencial_score, npc_group['influential_score']])
                network_score = np.concatenate([network_score, npc_group['network_score']])

        return {
            'npc_id': npc_id,
            'trust': trust,
            'influencial_score': influencial_score,
            'network_score': network_score
        }

# ---- Simulation ---- #
class DisinfoSimulator:
    def __init__(self, npc_data, disinformation_events, event_timeline, total_ticks=24, disinformers_per_tick=100, conversion_threshold=0.05):
        self.trust = npc_data['trust']
        self.influence = npc_data['influencial_score']
        self.network = npc_data['network_score']
        self.num_npcs = len(self.trust)
        self.events = disinformation_events
        self.timeline = event_timeline
        self.total_ticks = total_ticks
        self.disinformers_per_tick = disinformers_per_tick
        self.conversion_threshold = conversion_threshold
        self.logs = []

    def get_active_event(self, tick):
        for event in self.timeline:
            if event['start'] <= tick < event['start'] + event['duration']:
                return event['event_id']
        return None

    def compute_trust_delta(self, T_i, I_j, S, R):
        P_ij = np.clip(100 + (R - T_i * 100), 0, 100)
        delta_T = (P_ij / 100.0) * (S - T_i / 4.0) * I_j
        return delta_T

    def simulate_tick(self, tick):
        event_id = self.get_active_event(tick)

        if event_id is None:
            tick_log = {"tick": tick, "event_id": None, "note": "No active event."}
            self.logs.append(tick_log)
            print(json.dumps(tick_log, indent=2))
            return

        disinfo_batch = self.events.get(event_id, [])
        tick_log = {
            "tick": tick,
            "event_id": event_id,
            "disinformations": [],
        }

        for disinfo in disinfo_batch:
            S, R = disinfo['S'], disinfo['R']
            disinformers = np.random.choice(self.num_npcs, size=self.disinformers_per_tick, replace=False)
            trust_deltas = np.zeros_like(self.trust)

            for sender_id in disinformers:
                num_targets = max(1, int(self.network[sender_id] / 5))
                targets = np.random.choice(self.num_npcs, size=num_targets, replace=False)

                for receiver_id in targets:
                    T_i = self.trust[receiver_id]
                    I_j = self.influence[sender_id]
                    delta_T = self.compute_trust_delta(T_i, I_j, S, R)
                    trust_deltas[receiver_id] += delta_T

            self.trust -= trust_deltas
            self.trust = np.clip(self.trust, 0, 1)

            avg_trust = np.mean(self.trust)
            converted = int(np.sum(self.trust < self.conversion_threshold))
            delta_stats = {
                "mean": float(np.mean(trust_deltas)),
                "max": float(np.max(trust_deltas)),
                "min": float(np.min(trust_deltas))
            }

            tick_log["disinformations"].append({
                "S": S,
                "R": R,
                "avg_trust": float(avg_trust),
                "converted_npcs": converted,
                "delta_stats": delta_stats
            })

        self.logs.append(tick_log)
        print(json.dumps(tick_log, indent=2))

    def run(self):
        for tick in range(self.total_ticks):
            self.simulate_tick(tick)
            time.sleep(1)
        return self.logs

# ---- Simulation Config ---- #
disinformation_events = {
    0: [
        {"S": 0.9, "R": 0.2},
        {"S": 0.7, "R": 0.3},
    ],
    1: [
        {"S": 0.6, "R": 0.4},
        {"S": 0.5, "R": 0.5},
    ],
    2: [
        {"S": 0.3, "R": 0.6}
    ],
}

event_timeline = [
    {"event_id": 0, "start": 0,  "duration": 6},
    {"event_id": 1, "start": 6,  "duration": 10},
    {"event_id": 2, "start": 16, "duration": 8},
]

# ---- Entry Point ---- #
if __name__ == "__main__":
    processor = Processor()
    all_npcs = generate_npcs(10000)
    npc_data = processor.load_npcs(all_npcs)
    simulator = DisinfoSimulator(npc_data, disinformation_events, event_timeline)
    simulator.run()
