import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.evaluate import plot_evolution
from src.loader import load_prep
from src.ga import ga_selector

def main():
    x, y, feature_names, df = load_prep("data/raw/TCGA_GBM_LGG_Mutations_all.csv")

    ga = ga_selector(
        x=x,
        y=y,
        feature_name=feature_names,
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        generations=50,
        random_state=228,
        elitism_ratio=0.1,
        min_features=3
    )

    best_chromosome, best_fitness, best_features = ga.run(verbose=True)
    print(f"Процесс эволюции")
    plot_evolution(ga.history)

    print(f"Обучение и оценка")
    selected_indices = np.where(best_chromosome == 1)[0]
    model,importance_df = ga.evaluate_model(x,y,selected_indices,feature_names,random_state=228)

    #Предсказание новенького
    new_patient = {
        'IDH1': 'MUTATED',
        'TP53': 'MUTATED',
        'ATRX': 'NOT_MUTATED',
        'PTEN': 'NOT_MUTATED',
        'EGFR': 'NOT_MUTATED',
        'CIC': 'NOT_MUTATED',
        'MUC16': 'NOT_MUTATED',
        'PIK3CA': 'NOT_MUTATED',
        'NF1': 'NOT_MUTATED',
        'PIK3R1': 'NOT_MUTATED',
        'FUBP1': 'MUTATED',
        'RB1': 'NOT_MUTATED',
        'NOTCH1': 'NOT_MUTATED',
        'BCOR': 'NOT_MUTATED',
        'CSMD3': 'NOT_MUTATED',
        'SMARCA4': 'NOT_MUTATED',
        'GRIN2A': 'NOT_MUTATED',
        'IDH2': 'NOT_MUTATED',
        'FAT4': 'NOT_MUTATED',
        'PDGFRA': 'NOT_MUTATED',
        'age_years': 51,
        'sex': 1
    }

    from src.evaluate import prob_mutation_impact
    result = prob_mutation_impact(model,new_patient,[feature_names[i] for i in selected_indices])
    print(f"Результаты предсказания:")
    for key, value in result.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()