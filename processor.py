import numpy as np
import json
import time

def generate_npc(count, include_age=True):
    """
    Generate sample NPCs with random attributes.
    
    Args:
        count: Number of NPCs to generate
        include_age: If True, includes age. If False, includes influential_score and network_score
    
    Returns:
        dict: Dictionary containing arrays of NPC attributes
    """
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
    
    all_npcs = [npcs_with_age, npcs_with_scores]
    
    return all_npcs

class Processor:
    def __init__(self, unit_time=10):
        self.unit_time = unit_time

    def calculate_scores_based_on_age(self, age):
        # Parameters for the Gaussian functions
        I_max = 100  # Maximum influence score
        N_max = 100  # Maximum network score
        sigma = 15   # Standard deviation for influence function
        sigma_prime = 20  # Standard deviation for network function

        # Calculate influence score using Gaussian function centered at age 60
        f_age = np.exp(-((age - 60) / sigma) ** 2)
        influencial_score = I_max * f_age

        # Calculate network score using Gaussian function centered at age 60
        g_age = np.exp(-((age - 60) / sigma_prime) ** 2)
        network_score = N_max * g_age


        return influencial_score, network_score

    def _load_npcs(self, all_npcs):

        npc_id = np.array([])
        trust = np.array([])
        influencial_score = np.array([])
        network_score = np.array([])
        
        for npc_group in all_npcs:
            npc_id = np.concatenate([npc_id, npc_group['npc_id']])
            trust = np.concatenate([trust, npc_group['trust']])
            
            if 'age' in npc_group:
                # Calculate scores from age
                calc_influential, calc_network = self.calculate_scores_based_on_age(npc_group['age'])
                influencial_score = np.concatenate([influencial_score, calc_influential])
                network_score = np.concatenate([network_score, calc_network])
            else:
                # Use existing scores
                influencial_score = np.concatenate([influencial_score, npc_group['influential_score']])
                network_score = np.concatenate([network_score, npc_group['network_score']])
        
        print("NPC Details:")
        print(f"Total NPCs: {len(npc_id)}")
        # for i in range(len(npc_id)):
        #     print(f"NPC {int(npc_id[i])}: Trust={trust[i]:.3f}, Influential={influencial_score[i]:.2f}, Network={network_score[i]:.2f}")
        
        return {
            'npc_id': npc_id,
            'trust': trust,
            'influencial_score': influencial_score,
            'network_score': network_score
        }
    

if __name__ == "__main__":
    processor = Processor()
    all_npcs = generate_npcs(100000000)

    start_time = time.time()
    npc_data = processor._load_npcs(all_npcs)
    end_time = time.time()
    execution_time = end_time - start_time
    time_per_entry = execution_time / len(npc_data['npc_id'])
    print(npc_data)
    
    # print(json.dumps(npc_data, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))

    print("---------------------------------------------------------------")
    print(f"Processing time: {execution_time:.4f} seconds")
    print(f"Time per entry: {time_per_entry:.6f} seconds")
    print("---------------------------------------------------------------")


