import numpy as np
from typing import List
from src.core_engine import ClusterIndividual

def check_dominance(ind1: ClusterIndividual, ind2: ClusterIndividual) -> bool:
    """
    Determines if individual 1 strictly dominates individual 2.
    Assumes both objectives are structured for minimization.
    """
    f1_better = ind1.fitness[0] <= ind2.fitness[0] and ind1.fitness[1] <= ind2.fitness[1]
    strict_better = ind1.fitness[0] < ind2.fitness[0] or ind1.fitness[1] < ind2.fitness[1]
    return bool(f1_better and strict_better)

def fast_non_dominated_sort(population: List[ClusterIndividual]) -> List[List[ClusterIndividual]]:
    """
    Sorts a population of clustering solutions into hierarchical fronts based on dominance.
    Front 0 contains the absolute best non-dominated choices.
    """
    fronts = [[]]
    for p_idx, p in enumerate(population):
        p.domination_count = 0
        p.dominated_set = []
        
        for q_idx, q in enumerate(population):
            if p_idx == q_idx:
                continue
            if check_dominance(p, q):
                p.dominated_set.append(q)
            elif check_dominance(q, p):
                p.domination_count += 1
                
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_set:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    return [f for f in fronts if len(f) > 0]
def assign_crowding_distances(front: List[ClusterIndividual]):
    """
    Calculates the boundary spacing density around each solution within a front.
    Boundary solutions are given infinite distance values to preserve edge choices.
    """
    n_individuals = len(front)
    if n_individuals == 0:
        return
    if n_individuals <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return

    # Initialize all distances to zero
    for ind in front:
        ind.crowding_distance = 0.0

    n_objectives = len(front[0].fitness)
    
    for obj_idx in range(n_objectives):
        # Sort the front using the current objective score
        front.sort(key=lambda x: x.fitness[obj_idx])
        
        # Protect edge solutions
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        obj_range = front[-1].fitness[obj_idx] - front[0].fitness[obj_idx]
        if obj_range == 0:
            continue
            
        # Accumulate normalized spacing steps
        for i in range(1, n_individuals - 1):
            distance_contribution = (front[i+1].fitness[obj_idx] - front[i-1].fitness[obj_idx]) / obj_range
            front[i].crowding_distance += distance_contribution