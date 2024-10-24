"""
Aaron Dave A. Siapuatco
Ven Mark A.Recla
BSCS - 3
Firefly Optimization Algorithm
"""

import numpy as np
import pandas as pd
import random

def griewank(dimension) -> float:
    first_term, second_term = 0, 1
    for i, x in enumerate(dimension):
        first_term += x**2 / 4000
        second_term *= np.cos(x / np.sqrt(i + 1))
    return first_term - second_term + 1

def generate_population(dimension, population_size) -> list[list[float]]:
    lb, ub = -10, 10
    solutions = [np.random.uniform(lb, ub, dimension) for i in range(population_size)]
    return solutions

def firefly_operator(best_fly, worst_fly, attractiveness, levy, absorption, alpha=None, t=1) -> list[float]:
    alpha = random.random() if alpha is None else alpha
    r2 = sum((b - w) ** 2 for w, b in zip(worst_fly, best_fly))
    second_partial = attractiveness * np.exp(-absorption * r2)
    second_term = [second_partial * (b - w) for w, b in zip(worst_fly, best_fly)]
    third_term = [alpha * np.sign(random.random() - 0.5) * i * t ** (-levy) for i in worst_fly]
    return [float(a + b + c) for a, b, c in zip(worst_fly, second_term, third_term)]  # Convert to float here

def main() -> None:
    # Given Values
    dimension, population_size = 5, 10
    solutions = generate_population(dimension, population_size)
    
    # From solutions, generate fitness values through Griewank function
    fitness_arr = [griewank(solution) for solution in solutions]
    
    # Sort Fitness array get the first and last index (best and worst fitness)
    best_fitness, worst_fitness = min(fitness_arr), max(fitness_arr)
    best_fly, worst_fly = fitness_arr.index(best_fitness), fitness_arr.index(worst_fitness)
    
    # Create DataFrame
    data = []
    for solution, fitness in zip(solutions, fitness_arr):
        data.append([*solution, fitness])
    
    cols = [*(str(i) for i in range(1, dimension + 1)), "Fitness Values"]
    rows = [f"Solution {i}" for i in range(1, population_size + 1)]
    df = pd.DataFrame(data, index=rows, columns=cols)
    
    # Display the DataFrame
    print(df)
    
    # Display sorted DataFrame based on Fitness Values
    print("\nSorted DataFrame")
    print(df.sort_values(by="Fitness Values"))
    
    # Display worst and best firefly
    print("\nWorst and Best Firefly")
    print(f"{df.index[worst_fly]} (Worst): {df.iloc[worst_fly].values} -- Fitness: {worst_fitness}")
    print(f"{df.index[best_fly]} (Best): {df.iloc[best_fly].values} -- Fitness: {best_fitness}")
    
    
    # Store old position and fitness
    old_position = [float(x) for x in solutions[worst_fly]]  # Convert to float
    old_fitness = fitness_arr[worst_fly]
    """ 
    Parameters:
        Attractiveness Test:
            - attractiveness: 1, 10
            levy: 2
            absorption: 1
        Levy Flight Test:
            attractiveness: 1
            - levy: 2, 10
            absorption: 1
        Absorption Test:
            attractiveness: 1
            levy: 2
            - absorption: 1,5
    """
    # Calculate new position and fitness
    improved_worst_fly = firefly_operator(
        solutions[best_fly],
        solutions[worst_fly],
        attractiveness=1,
        levy=2,
        absorption=1
    )
    new_fitness = griewank(improved_worst_fly)
    
    # Display results with clean formatting
    print("\nResults")

    print(f"Old Fitness: {old_fitness:.6f}")
    print(f"Old Position: [{', '.join(f'{x:.6f}' for x in old_position)}]")
    print(f"New Fitness: {new_fitness:.6f}")
    print(f"New Position: [{', '.join(f'{x:.6f}' for x in improved_worst_fly)}]")
    print(f"Improvement: {new_fitness - old_fitness:.6f}")

if __name__ == "__main__":
    main()