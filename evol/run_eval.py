import os

os.chdir('/gpfs1/tangfuchou_pkuhpc/tangfuchou_coe/jiangzh/cellTransformer/code/evol/')
from pred import CellTransformer, load_msgpack, map_guide_to_peaks

import pickle
import pyranges as pr
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir('/gpfs1/tangfuchou_pkuhpc/tangfuchou_coe/jiangzh/cellTransformer/GET/')

def predict(args):
    guide_list, peak_hits = args
    ct = CellTransformer(
        guide_list=guide_list,
        peak_hits=peak_hits,
        target_gene=["GFP"],
        output_dir="./data/get_tmp/",
        celltype="HEK293T",
        insert_transgene=True,
        prediction_scope=roi,
        motif_bed="../resource/hg38.archetype_motifs.v1.0.bed.gz",
        zarr_path="./data/zarr/HEK293T_hPGK1_AAVS1.zarr"
    )
    score = ct.predict()
    return (guide_list, score)


if __name__ == "__main__":
    threads = 4

    roi = pr.PyRanges(
        chromosomes=["chr19"],
        starts=[int(55115750 - 5e5)],
        ends=[int(55115750 + 5e5)]
    )

    kmer_to_peak = load_msgpack("./data/index/hg38_CATlas_cCREs.9mer.kmer_to_peak_freq.msgpack")

    # select valid kmer only with coverage < 1e4
    with open('./data/intervention/hg38_CATlas_cCREs.9mer.kmer_cov_by_peak.pkl', 'rb') as f:
        kmer_cov = pickle.load(f)

    with open('./data/intervention/AAVS1_1Mb_9mer_HEK293T_novel.pkl', 'rb') as f:
        roi_kmer = pickle.load(f)

    kmer_cov_flt = {k: v for k, v in kmer_cov.items() if v < 1e4}   # filter kmer with global hits more than 10k
    roi_kmer_counts = Counter(k for k in roi_kmer if k in kmer_cov_flt)
    input_list = [[kmer] for kmer,count in roi_kmer_counts.items() if count >= 3]  # minimal peak hits
    peak_hits = [map_guide_to_peaks(guide_list, kmer_to_peak, hit_threshold=1) for guide_list in input_list]

    args_list = list(zip(input_list, peak_hits))

    save_path = 'non_repetitive_evaluation_results.pkl'
    done_set = set()

    # Load existing results if file exists
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            existing_results = pickle.load(f)
        done_set = {tuple(x[0]) for x in existing_results}  # unique key from guide_list
        print(f"Loaded {len(existing_results)} previous results. Skipping completed tasks.")
    else:
        existing_results = []

    # Filter out already completed tasks
    remaining_args = [args for args in args_list if tuple(args[0]) not in done_set]

    batch_size = 10  # save every 10 results
    batch_results = []  # temporary storage for batch saving

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(predict, args): args for args in remaining_args}

        for future in as_completed(futures):
            res = future.result()
            existing_results.append(res)
            batch_results.append(res)  # collect in batch

            # Save only when batch is full
            if len(batch_results) >= batch_size:
                with open(save_path, 'wb') as f:
                    pickle.dump(existing_results, f)
                batch_results.clear()
                print(f"Progress: {len(existing_results)}/{len(args_list)} tasks completed (batch saved)")

    # Save any leftover results after loop
    if batch_results:
        with open(save_path, 'wb') as f:
            pickle.dump(existing_results, f)
        print("Final batch saved.")
