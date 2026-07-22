import numpy as np
import sklearn as sl
from tqdm import tqdm

class ga_selector:
    def __init__(self,x,y,feature_name,
                 population_size=100,
                 mutation_rate=0.1,
                 crossover_rate=0.5,
                 generations=100,
                 random_state=42,
                 elitism_ratio=0.1,
                 min_features=3):
        self.x = x
        self.y = y
        self.feature_name = feature_name
        self.n_features = x.shape[1]
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.elitism_ratio = elitism_ratio
        self.min_features = min_features

        self.history ={
            "best_fitness": [],
            "mean_fitness": [],
            'best_features': []
        }

    #Создаем первых особей
    #Прикол, обновил вс код и подключил гитхаб и теперь работает автозаполнение с помощью иишки
    def _initialize_population(self):
        population = []

        for _ in range(self.population_size):
            chromosome = np.zeros(self.n_features, dtype=int)
            n_genes = self.rng.randint(self.min_features, self.n_features + 1)
            indices = self.rng.choice(self.n_features, size=n_genes, replace=False)
            chromosome[indices] = 1
            population.append(chromosome)

        return np.array(population)

    #Оценка 1 особи
    """
    Попробовать разные модели, градиентный бустинг, т.д.
    Также попробовать разные метрики, да и поиграться с параметрами модели, типа глубины деревьев, кол-ва деревьев и т.д.
    """
    def _fitness(self, chromosome):
        selected_indices = np.where(chromosome == 1)[0]

        if len(selected_indices) < self.min_features:
            return 0.0
        x_selected = self.x[:, selected_indices]

        model = sl.ensemble.RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            max_depth=5,
            n_jobs=-1)
        
        scores = sl.model_selection.cross_val_score(
            model,
            x_selected,
            self.y,
            cv=5,
            scoring='f1_macro'
        )
        return scores.mean()
    
    #Оценка популяции
    def _evaluate_population(self, population):
        fitness_value = np.array([self._fitness(chromosome) for chromosome in population])
        return fitness_value

    #Отбор родителей
    def _selection(self, population, fitness_values):
        def tournament_selection():
            indices = self.rng.choice(self.population_size, size=3, replace=False)
            best_index = indices[np.argmax(fitness_values[indices])]
            return population[best_index].copy()

        parent1 = tournament_selection()
        parent2 = tournament_selection()

        return parent1, parent2

    #Кроссовер, чо еще сказать
    def _crossover(self, parent1, parent2):
        if self.rng.random() < self.crossover_rate:
            point = self.rng.integers(1, self.n_features - 1)
            
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])

            return child1, child2
        return parent1.copy(), parent2.copy()

    #Мутации
    def _mutation(self, chromosome):
        mutated = chromosome.copy()

        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]

        if mutated.sum() < self.min_features:
            zero_indices = np.where(mutated == 0)[0]
            needed = self.min_features - mutated.sum()
            add_indices = self.rng.choice(zero_indices, 
                                          size=min(needed, len(zero_indices)), replace=False)
            mutated[add_indices] = 1

        return mutated

    def run(self,verbose=True):
        population = self._initialize_population()
        iterator = tqdm(range(self.generations)) if verbose else range(self.generations)

        for generation in iterator:
            fitness_values = self._evaluate_population(population)

            best_index = np.argmax(fitness_values)
            self.history['best_fitness'].append(fitness_values[best_index])
            self.history['mean_fitness'].append(fitness_values.mean())
            self.history['best_features'].append(population[best_index].copy())

            if verbose:
                selected_genes = np.where(population[best_index] == 1)[0]
                gene_names = [self.feature_name[i] for i in selected_genes]

                iterator.set_description(
                    f"Gen {generation+1}/{self.generations} \n Best Fitness: {fitness_values[best_index]:.4f} \n Mean Fitness: {fitness_values.mean():.4f} \n Selected Genes: {gene_names}")

            n_elite = max(1, int(self.elitism_ratio * self.population_size))
            elite_indices = np.argsort(fitness_values)[-n_elite:]
            elite = population[elite_indices].copy()

            new_population = []
            new_population.extend(elite)

            while len(new_population) < self.population_size:
                parent1, parent2 = self._selection(population, fitness_values)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population[:self.population_size])

        final_fitness = self._evaluate_population(population)
        best_final_index = np.argmax(final_fitness)

        self.best_chromosome = population[best_final_index]
        self.best_fitness = final_fitness[best_final_index]
        self.best_features = [self.feature_name[i] for i in np.where(self.best_chromosome == 1)[0]]

        if verbose:
            print(f"Дело сделано\nЛучший фитнесс {self.best_fitness:.4f}\nОтобрано генов {self.best_features}\nГены(букины) {self.best_chromosome}")

        return self.best_chromosome, self.best_fitness, self.best_features