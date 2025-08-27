import os
import zarr
import msgpack
import hashlib
import pyranges as pr
import pandas as pd
from collections import Counter
from get_model.config.config import load_config, export_config, load_config_from_yaml
from get_model.run_region import run_zarr_delta
from get_model.preprocess_utils import query_motif, get_motif, add_activation_to_zarr

def load_msgpack(idx_path: str):
    with open(idx_path, "rb") as f:
        print(f"Loading {idx_path}...")
        x = msgpack.unpackb(f.read(), raw=False)
    return x

def map_guide_to_peaks(guide_list, kmer_to_peak, hit_threshold=1):
    peak_counter = Counter()
    for kmer in guide_list:
        target_peaks = kmer_to_peak.get(kmer, {})
        for coord, info in target_peaks.items():
            if ":" in coord and "-" in coord:
                chrom = coord.split(":")[0]
                start, end = map(int, coord.split(":")[1].split("-"))
                count = info.get("count", 1)
                peak_counter[(chrom, start, end)] += count

    peak_hits = Counter({k: v for k, v in peak_counter.items() if v >= hit_threshold})
    return peak_hits

class CellTransformer:
    def __init__(
        self,
        guide_list,
        peak_hits,
        target_gene=["GFP"],
        prediction_scope=None,  # pyRanges object for region of interest
        insert_transgene=True,
        output_dir="./data/get_tmp/",
        run_id=None,
        celltype="HEK293T",
        num_region_per_sample=200,
        motif_bed="../resource/hg38.archetype_motifs.v1.0.bed.gz",
        zarr_path="./data/zarr/HEK293T_hPGK1_AAVS1.zarr",
    ):
        self.guide_list = guide_list
        self.peak_hits = peak_hits 
        self.target_gene = target_gene
        self.output_dir = output_dir

        self.run_id = run_id or self.guide_list_hash()

        self.celltype = celltype
        self.run_name = f"intervention_{self.run_id}"

        self.num_region_per_sample = num_region_per_sample
        self.prediction_scope = prediction_scope
        self.insert_transgene = insert_transgene

        self.motif_bed = motif_bed
        self.zarr_path = zarr_path

        self.query_motif_bed = f"./data/tmp/query_motif_{self.run_id}"
        self.delta_peak_bed = f"./data/tmp/delta_peaks_{self.run_id}.bed"
        self.get_motif_bed = f"./data/tmp/get_motif_{self.run_id}"
        self.motif_bed_out = f"./data/tmp/out_get_motif_{self.run_id}.bed"
        self.peak_bed_out = f"./data/tmp/out_peak_{self.run_id}.bed"
        self.output_csv = os.path.join(self.output_dir, celltype, self.run_name + ".csv")

        self.config_file = f"{self.output_dir}/{celltype}/config_inference_{self.run_id}.yaml"

        os.makedirs(os.path.join(self.output_dir, celltype), exist_ok=True)

    def guide_list_hash(self):
        joined = ','.join(sorted(self.guide_list))
        return hashlib.md5(joined.encode()).hexdigest()

    def build_config(self):
        cfg = load_config('finetune_tutorial_pbmc')
        export_config(cfg, self.config_file)
        cfg = load_config_from_yaml(self.config_file)

        cfg.run.project_name = self.celltype
        cfg.dataset.zarr_path = self.zarr_path
        cfg.dataset.celltypes = self.celltype
        cfg.dataset.quantitative_atac = False
        cfg.dataset.num_region_per_sample = self.num_region_per_sample
        cfg.finetune.checkpoint = "./data/checkpoints/finetune_fetal_adult_leaveout_astrocyte.checkpoint-best.pth"
        cfg.machine.num_devices = 0
        cfg.machine.num_workers = 0
        cfg.machine.batch_size = 4
        cfg.machine.output_dir = self.output_dir
        cfg.run.run_name = self.run_name
        cfg.stage = 'predict'
        cfg.task.test_mode = 'inference'
        cfg.finetune.resume_ckpt = f"./data/get_output/{self.celltype}/regionEmb_head_finetune_binary/checkpoints/best.ckpt"
        cfg.run.use_wandb = False
        cfg.task.layer_names = []
        cfg.task.gene_list = self.target_gene
        cfg.task.run_id = f"activation_{self.run_id}"
        export_config(cfg, self.config_file)
    
    def load_original_peak(self):
        z = zarr.open(self.zarr_path, mode='r')
        peak_names = z['peak_names'][:]

        if isinstance(peak_names[0], bytes):
            peak_names = [x.decode('utf-8') for x in peak_names]

        chroms, starts, ends = [], [], []
        for name in peak_names:
            chrom, rest = name.split(":")
            start, end = rest.split("-")
            chroms.append(chrom)
            starts.append(int(start))
            ends.append(int(end))

        peak_df = pd.DataFrame({
            "Chromosome": chroms,
            "Start": starts,
            "End": ends,
        })

        peak = pr.PyRanges(peak_df)

        return peak
    
    def generate_delta_peaks(self):
        filtered = [(chrom, start, end) for (chrom, start, end), count in self.peak_hits.items()]
        df = pd.DataFrame(filtered, columns=["Chromosome", "Start", "End"])
        df["aCPM"] = 1
        query_peaks = pr.PyRanges(df)

        existing_peaks = self.load_original_peak()
        delta_peaks = query_peaks.overlap(existing_peaks, invert=True)

        if self.prediction_scope:
            delta_peaks = delta_peaks.overlap(self.prediction_scope)

        if self.insert_transgene:
            # HDR deletion of AAVS1
            exclude_region = pr.PyRanges(chromosomes=["chr19"], starts=[55116169], ends=[55116569])
            delta_peaks = delta_peaks.overlap(exclude_region, invert=True)

        delta_peaks_df = delta_peaks.as_df()
        if delta_peaks_df.empty:
            pd.DataFrame(columns=["Chromosome", "Start", "End"]).to_csv(self.delta_peak_bed, sep='\t', header=False, index=False)
            return False
        else:
            delta_peaks_df.to_csv(self.delta_peak_bed, sep='\t', header=False, index=False)
            return True
        
    def prepare_delta_zarr(self):
        # get motif annotation for delta peaks
        query_motif(self.delta_peak_bed, self.motif_bed, base_name = self.query_motif_bed)
        get_motif(self.delta_peak_bed, f'{self.query_motif_bed}.bed', base_name = self.get_motif_bed)

        # If inserting AAVS1-GFP transgene, add hPGK1 promoter
        if self.insert_transgene:
            motif_df = pd.read_csv(f"{self.get_motif_bed}.bed", sep="\t", header=None,
                                names=["Chromosome", "Start", "End", "Motif", "Score"])
            hPGK1_bed = "./data/intervention/hPGK1-AAVS1-insertion.bed"
            hPGK_df = pd.read_csv(hPGK1_bed, sep="\t", header=None,
                                  names=["Chromosome", "Start", "End", "Motif", "Score"])
            motif_df = pd.concat([motif_df, hPGK_df], ignore_index=True)

            motif_df = motif_df.sort_values(by=["Chromosome", "Start"])
            motif_df.to_csv(self.motif_bed_out, sep="\t", header=False, index=False)

            df_peak = pd.read_csv(self.delta_peak_bed, sep="\t", header=None,
                names=["Chromosome", "Start", "End", "aCPM"])
        
            transgene_peak = pd.DataFrame({
                "Chromosome": ["chr19"],
                "Start": [55115750],
                "End": [55115751],
                "aCPM": [1.0]
            })
            df_peak = pd.concat([df_peak, transgene_peak], ignore_index=True)
            df_peak = df_peak.sort_values(by=["Chromosome", "Start"])
            df_peak.to_csv(self.peak_bed_out, sep="\t", header=False, index=False)

            df_AAVS1_GFP = pd.DataFrame({
                "Chromosome": ["chr19"],
                "Start": [55115750],
                "End": [55115751],
                "Strand": ["-"],
                "gene_name": ['GFP']
            })
            anno = pr.PyRanges(df_AAVS1_GFP)
            version = None
        else:
            self.motif_bed_out = f"{self.get_motif_bed}.bed"
            self.peak_bed_out = self.delta_peak_bed
            anno = None
            version = 44

        add_activation_to_zarr(
            zarr_file=self.zarr_path,
            peak_motif_bed=self.motif_bed_out,
            atac_file=self.peak_bed_out,
            name=f'activation_{self.run_id}',
            celltype=self.celltype,
            gene_anno=anno,
            assembly='hg38',
            version=version,
            extend_bp=300
        )

    def predict(self):
        print(f"Running prediction for {self.celltype} with run ID {self.run_id}...")

        self.build_config()
        has_delta_peaks = self.generate_delta_peaks()
        if not has_delta_peaks:
            print("No delta peaks found, returning invalid fitness.")
            return -1.0
        self.prepare_delta_zarr()

        # Run GET model to predict fitness
        cfg = load_config_from_yaml(self.config_file)
        cfg.dataset.leave_out_celltypes = self.celltype
        run_zarr_delta(cfg)

        # Read the output CSV to get the final predicted expression value
        df = pd.read_csv(self.output_csv, header=None)
        return float(df.iloc[-1, 2])
