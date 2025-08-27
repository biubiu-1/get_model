import random
import pickle
import pandas as pd
import pyranges as pr
from collections import Counter
from deap import base, creator, tools
import os
import multiprocessing
from multiprocessing import Manager
import uuid

os.chdir('/gpfs1/tangfuchou_pkuhpc/tangfuchou_coe/jiangzh/cellTransformer/code/evol/')
from pred import CellTransformer, load_msgpack, map_guide_to_peaks

os.chdir('/gpfs1/tangfuchou_pkuhpc/tangfuchou_coe/jiangzh/cellTransformer/GET/')

# =========================================
# Configuration
# =========================================
CONFIG = {
    "set_size": 6,
    "population_size": 120,
    "num_generations": 100,
    "crossover_prob": 0.85,
    "mutation_prob": 0.25,
    "tournament_size": 3,
    "immigrant_ratio": 0.08,
    "multi_point_mutation": True,
    "mutation_decay": True
}

KMER_CONFIG = {
    "max_global_hits": 1e4,
    "min_local_hits": 3,
    "min_effective_hits": 3
}

# Global references
kmer_to_peak = None
roi = None
fitness_cache = None
threads = 10  # Default number of threads
group_name = 'NG_global_1e4_local_3_eff_3'

index_name = "PAM_NG.hg38_CATlas_cCREs.9mer"
index_path = f"./data/index/{index_name}.kmer_to_peak_freq.msgpack"
ref_cCRE_bed = '../resource/CATlas_cCRE_hg38.bed.gz'
input_peaks = './data/Hu_2023_NGS_bulk.HEK293T.merged.aCPM.bed'

# =========================================
# Data loading
# =========================================
'''
def init_worker(kmer_data, roi_data, cache_dict):
    global kmer_to_peak, roi, fitness_cache
    kmer_to_peak = kmer_data
    roi = roi_data
    fitness_cache = cache_dict
'''

def load_kmer_index():
    return load_msgpack(index_path)

def load_data():
    """Load ROI and k-mer datasets for the GA search."""
    try:
        roi = pr.PyRanges(
            chromosomes=["chr19"],
            starts=[int(55115750 - 5e5)],
            ends=[int(55115750 + 5e5)]
        )

        kmer_cov_path = f'./data/index/{index_name}.kmer_cov_by_peak.pkl'
        if not os.path.exists(kmer_cov_path):
            kmer_cov = {kmer: len(peaks) for kmer, peaks in kmer_to_peak.items() if peaks}
            with open(kmer_cov_path, 'wb') as f:
                pickle.dump(kmer_cov, f)  # kmer coverage dictionary
        else:
            with open(kmer_cov_path, 'rb') as f:
                kmer_cov = pickle.load(f)

        roi_kmer_path = f'./data/intervention/{index_name}_HEK293T_AAVS1_1Mb_ROI_kmer.pkl'
        if not os.path.exists(roi_kmer_path):
            peak_to_kmer = load_msgpack(f'./data/index/{index_name}.peak_to_kmer_freq.msgpack')
            existing_peaks = pr.read_bed(input_peaks)
            
            roi_cCRE = pr.read_bed(ref_cCRE_bed).overlap(roi)
            roi_novel = roi_cCRE.overlap(existing_peaks, invert=True)
            roi_novel_name = roi_novel.as_df().apply(lambda x: f"{x['Chromosome']}:{x['Start']}-{x['End']}", axis=1).tolist()
            roi_kmer = [i for roi_name in roi_novel_name for i in list(peak_to_kmer[roi_name]['kmer_info'].keys())]

            with open(roi_kmer_path, 'wb') as f:
                pickle.dump(roi_kmer, f)
        else:
            with open(roi_kmer_path, 'rb') as f:
                roi_kmer = pickle.load(f)

        # Filter out extremely common or non-multiplexed k-mers
        kmer_cov_flt = {k: v for k, v in kmer_cov.items() if v <= KMER_CONFIG["max_global_hits"]}
        roi_kmer_counts = Counter(k for k in roi_kmer if k in kmer_cov_flt)
        roi_kmer_list = [k for k, count in roi_kmer_counts.items() if count >= KMER_CONFIG["min_local_hits"]]

        if not roi_kmer_list:
            raise ValueError("No valid k-mers found after filtering")

        print(f"Available k-mers: {len(roi_kmer_list)}")
        return roi, roi_kmer_list
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# =========================================
# Helper: fix duplicates
# =========================================
def fix_duplicates(individual, kmer_list):
    """Ensure no duplicate kmers in an individual."""
    seen = set()
    for i in range(len(individual)):
        if individual[i] in seen:
            choices = [k for k in kmer_list if k not in seen]
            individual[i] = random.choice(choices)
        seen.add(individual[i])
    return individual

def deduplicate_population(population):
    seen = set()
    unique_pop = []
    for ind in population:
        key = tuple(sorted(ind))
        if key not in seen:
            seen.add(key)
            unique_pop.append(ind)
    return unique_pop

# =========================================
# Fitness function with caching
# =========================================
def fitness(individual):
    """Evaluate individual fitness using CellTransformer, with caching."""

    key = tuple(individual)
    if key in fitness_cache:
        return (fitness_cache[key],)

    peak_hits = map_guide_to_peaks(
        individual, 
        kmer_to_peak, 
        hit_threshold=KMER_CONFIG["min_effective_hits"]
    )

    run_id = uuid.uuid4() 

    ct = CellTransformer(
        guide_list=list(individual),
        peak_hits=peak_hits,
        target_gene=["GFP"],
        output_dir="./data/get_tmp/",
        celltype="HEK293T",
        insert_transgene=True,
        prediction_scope=roi,
        run_id=run_id,
        motif_bed="../resource/hg38.archetype_motifs.v1.0.bed.gz",
        zarr_path="./data/zarr/HEK293T_hPGK1_AAVS1.zarr"
    )

    score = ct.predict()    

    fitness_cache[key] = score
    return (score,)

# =========================================
# Custom mutation
# =========================================
def custom_mutation(individual, indpb, kmer_list):
    """Randomly replace k-mers in the individual with probability indpb."""
    for i in range(len(individual)):
        if random.random() < indpb:
            new_kmer = random.choice(kmer_list)
            while new_kmer in individual:  # avoid duplicates
                new_kmer = random.choice(kmer_list)
            individual[i] = new_kmer
    return individual,

# =========================================
# Main GA procedure
# =========================================
def main():
    global kmer_to_peak, roi, fitness_cache, threads, group_name

    # Load data
    kmer_to_peak = load_kmer_index()
    roi, roi_kmer_list = load_data()

    manager = Manager()
    fitness_cache = manager.dict()

    # DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_kmer", lambda: random.choice(roi_kmer_list))
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: random.sample(roi_kmer_list, CONFIG["set_size"]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", custom_mutation, kmer_list=roi_kmer_list, indpb=CONFIG["mutation_prob"])
    toolbox.register("select", tools.selTournament, tournsize=CONFIG["tournament_size"])
    toolbox.register("evaluate", fitness)

    # Multiprocessing
    pool = multiprocessing.Pool(
        processes=threads,
        #initializer=init_worker,
        #initargs=(kmer_to_peak, roi, fitness_cache)
    )
    toolbox.register("map", pool.map)

    # Init population
    pop = toolbox.population(n=CONFIG["population_size"])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))
    logbook = tools.Logbook()
    logbook.header = ["gen", "best_fitness", "avg_fitness"]

    # Output CSV
    output_file = f"./results/{group_name}/ga_individuals_{uuid.uuid4()}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(columns=["generation", "individual", "fitness"]).to_csv(output_file, index=False)

    mutation_prob = CONFIG["mutation_prob"]

    # Evolutionary loop
    for gen in range(CONFIG["num_generations"]):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CONFIG["crossover_prob"]:
                toolbox.mate(child1, child2)
                fix_duplicates(child1, roi_kmer_list)
                fix_duplicates(child2, roi_kmer_list)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation (multi-point)
        for mutant in offspring:
            if random.random() < mutation_prob:
                if CONFIG.get("multi_point_mutation", False):
                    # Mutate multiple points
                    num_points = random.randint(1, CONFIG["set_size"])
                    for _ in range(num_points):
                        toolbox.mutate(mutant)
                else:
                    toolbox.mutate(mutant)
                fix_duplicates(mutant, roi_kmer_list)
                del mutant.fitness.values
                
        # 1. Immigration
        num_immigrants = int(CONFIG.get("immigrant_ratio", 0) * CONFIG["population_size"])
        for _ in range(num_immigrants):
            new_ind = toolbox.individual()
            offspring[random.randint(0, len(offspring) - 1)] = new_ind

        # 2. Deduplicate
        offspring = deduplicate_population(offspring)

        # 3. Fill to full size
        while len(offspring) < CONFIG["population_size"]:
            new_ind = toolbox.individual()
            offspring.append(new_ind)

        # 4. Evaluate all invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 5. Save CSV
        df = pd.DataFrame([
            [gen, str(ind), ind.fitness.values[0]]
            for ind in offspring
        ], columns=["generation", "individual", "fitness"])
        df.to_csv(output_file, mode='a', header=False, index=False)

        pop[:] = offspring
        hof.update(pop)

        # Logging
        record = stats.compile(pop)
        logbook.record(gen=gen, best_fitness=hof[0].fitness.values[0], avg_fitness=record["avg"])
        print(logbook.stream)

        # Mutation decay
        if CONFIG.get("mutation_decay", False):
            mutation_prob = max(0.08, mutation_prob * 0.95)

    pool.close()
    pool.join()
    print(f"Individual data saved to {output_file}")
    return hof[0]

if __name__ == "__main__":
    best_individual = main()
    print(f"Final best individual: {best_individual}, Fitness: {best_individual.fitness.values[0]}")