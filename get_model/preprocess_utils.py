# This file contains some utility functions for preprocessing the data for the region-level GET model.
# You can use this instead of modules/atac_rna_data_processing to produce the zarr-based data

import os
import subprocess

import numpy as np
import pandas as pd
import zarr
from gcell.rna.gencode import Gencode
import pyranges as pr
from numcodecs import Blosc
from typing import Union, Optional

# required softwares:
# bedtools
# tabix
# wget

def download_motif(motif_url, index_url, motif_dir="data"):
    """
    Download motif data from a URL and its index.

    This function uses the `wget` command-line tool to download the motif file and its index.

    Args:
        motif_url (str): URL to the motif file.
        index_url (str): URL to the index file.
    """
    subprocess.run(
        ["wget", "-O", os.path.join(motif_dir, "hg38.archetype_motifs.v1.0.bed.gz"), motif_url], check=True
    )
    subprocess.run(
        ["wget", "-O", os.path.join(motif_dir, "hg38.archetype_motifs.v1.0.bed.gz.tbi"), index_url], check=True
    )


def join_peaks(peak_bed, reference_peaks=None):
    """
    Join peaks from a peak file and a reference peak file.

    This function uses the `bedtools` command-line tool to perform the following operations:
    1. Intersect the peak file with the reference peak file.
    2. Save the final result to a bed file.

    Args:
        peak_bed (str): Path to the peak file.
        reference_peaks (str): Path to the reference peak file.

    Returns:
        str: Path to the output bed file containing the joined peaks.
    """
    if reference_peaks:
        subprocess.run(
            [
                "bedtools",
                "intersect",
                "-a",
                peak_bed,
                "-b",
                reference_peaks,
                "-wa",
                "-u",
                "-o",
                "joint_peaks.bed",
            ],
            check=True,
        )

    else:
        if os.path.exists("join_peaks.bed"):
            os.remove("join_peaks.bed")
        os.symlink(peak_bed, "join_peaks.bed")
    return "join_peaks.bed"


def query_motif(peak_bed, motif_bed, base_name = 'query_motif'):
    """
    Query motif data from a peak file and a motif file.

    This function uses the `tabix` command-line tool to perform the following operations:
    1. Intersect the peak file with the motif file.
    2. Save the final result to a bed file.

    Args:
        peak_bed (str): Path to the peak file.
        motif_bed (str): Path to the motif file.

    Returns:
        str: Path to the output bed file containing the peak motif data.
    """
    subprocess.run(
        ["tabix", motif_bed, "-R", peak_bed],
        stdout=open(base_name+".bed", "w"),
        check=True,
    )
    return base_name+".bed"

def get_motif(peak_file, motif_file, base_name='get_motif'):
    # 8th Aug 2025: No chromosome splitting, no loop, single process.

    cmd = f'''
    ATAC_PEAK_FILE="{peak_file}"
    ATAC_MOTIF_FILE="{motif_file}"
    OUTPUT_BASE="{base_name}"

    # Step 1: extract columns 1-3 from peak file and filter unwanted chromosomes
    awk '{{OFS="\\t"; print $1,$2,$3}}' "$ATAC_PEAK_FILE" | \
        grep -Ev "random|alt|KI|chrY" > "$OUTPUT_BASE.filtered_peak.bed"

    # Step 2: keep all columns from motif file, filter unwanted chromosomes
    grep -Ev "random|alt|KI|chrY" "$ATAC_MOTIF_FILE" > "$OUTPUT_BASE.filtered_motif.bed"

    # Step 3: intersect, sort, and group
    bedtools intersect -a "$OUTPUT_BASE.filtered_peak.bed" -b "$OUTPUT_BASE.filtered_motif.bed" -wa -wb |
    cut -f1,2,3,7,8,10 |
    sort -k1,1 -k2,2n -k3,3n -k4,4 |
    bedtools groupby -i - -g 1-4 -c 5 -o sum \
    > "$OUTPUT_BASE.bed"

    # Step 4: clean up
    rm "$OUTPUT_BASE.filtered_peak.bed" "$OUTPUT_BASE.filtered_motif.bed"
    '''
    
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
    return base_name + '.bed'

def _get_motif(peak_file, motif_file, threads=4, base_name='get_motif'):
    # 4th, June, 2025; Revised by Zhenhuan Jiang.
    # Output equivalence has been verfied by md5sum. 
    # After that, `-k1,1V` was changed to `-k1,1` for general consistency.
    
    """
    Get motif data from a peak file and a motif file.

    This function uses the `bedtools` command-line tool to perform the following operations:
    1. Intersect the peak file with the motif file.
    2. Group the intersected data by peak and motif.
    3. Sum the scores for overlapping peaks and motifs.
    4. Sort the resulting data by chromosome, start, end, and motif.
    5. Save the final result to a bed file.

    Args:
        peak_file (str): Path to the peak file.
        motif_file (str): Path to the motif file.

    Returns:
        str: Path to the output bed file containing the peak motif data.
    """

    cmd = f'''
    ATAC_PEAK_FILE="{peak_file}"
    ATAC_MOTIF_FILE="{motif_file}"
    CPU="{threads}"
    OUTPUT_BASE="{base_name}"

    awk -v OUTPUT_BASE="$OUTPUT_BASE" '{{OFS="\\t"; print $1,$2,$3 > OUTPUT_BASE"."$1".peak.bed"}}' "$ATAC_PEAK_FILE"
    awk -v OUTPUT_BASE="$OUTPUT_BASE" '{{print > OUTPUT_BASE"."$1".motif.bed"}}' "$ATAC_MOTIF_FILE"

    chr_list=$(cut -f1 $ATAC_PEAK_FILE |  grep -Ev "random|alt|KI|chrY" | sort -k1,1 | uniq)

    export OUTPUT_BASE
    rm -f "$OUTPUT_BASE.*.peak_motif.bed"
    echo "$chr_list" | parallel -j "$CPU" '
        CHR={{}}
        if [[ -f "$OUTPUT_BASE.$CHR.peak.bed" && -f "$OUTPUT_BASE.$CHR.motif.bed" ]]; then
            bedtools intersect -a "$OUTPUT_BASE.$CHR.peak.bed" -b "$OUTPUT_BASE.$CHR.motif.bed" -wa -wb |
            cut -f1,2,3,7,8,10 |
            sort -k1,1 -k2,2n -k3,3n -k4,4 |
            bedtools groupby -i - -g 1-4 -c 5 -o sum \
            > "$OUTPUT_BASE.$CHR.peak_motif.bed"
        fi
    '

    rm -f "$OUTPUT_BASE.bed"
    for CHR in $chr_list
    do
        cat "$OUTPUT_BASE.$CHR.peak_motif.bed" >> "$OUTPUT_BASE.bed"
    done

    # Clean up temporary files
    rm "$OUTPUT_BASE."*".peak.bed" "$OUTPUT_BASE."*".motif.bed" "$OUTPUT_BASE."*".peak_motif.bed"
    '''
    
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
    return base_name + '.bed'


def create_peak_motif(peak_motif_bed, output_zarr, peak_bed):
    """
    Create a peak motif zarr file from a peak motif bed file.

    This function reads a peak motif bed file, pivots the data, and saves it to a zarr file.
    The zarr file contains three datasets: 'data', 'peak_names', 'motif_names', and 'accessibility'.
    The 'data' dataset is a sparse matrix containing the peak motif data.
    The 'peak_names' dataset contains the peak names.
    The 'motif_names' dataset contains the motif names.

    Args:
        peak_motif_bed (str): Path to the peak motif bed file.
        output_zarr (str): Path to the output zarr file.
    """
    # Read the peak motif bed file
    peak_motif = pd.read_csv(
        peak_motif_bed,
        sep="\t",
        header=None,
        names=["Chromosome", "Start", "End", "Motif_cluster", "Score"],
    )

    # Pivot the data
    peak_motif_pivoted = peak_motif.pivot_table(
        index=["Chromosome", "Start", "End"],
        columns="Motif_cluster",
        values="Score",
        fill_value=0,
    )
    peak_motif_pivoted.reset_index(inplace=True)

    # Create the 'Name' column
    peak_motif_pivoted["Name"] = peak_motif_pivoted.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )
    peak_motif_pivoted = peak_motif_pivoted.drop(columns=["Chromosome", "Start", "End"])
    # Read the original peak bed file
    original_peaks = pd.read_csv(
        peak_bed, sep="\t", header=None, names=["Chromosome", "Start", "End", "Score"]
    )
    # exclude chrM and chrY
    original_peaks = original_peaks[~original_peaks.Chromosome.isin(["chrM", "chrY"])]
    original_peaks["Name"] = original_peaks.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )

    # Merge the pivoted data with the original peaks
    merged_data = pd.merge(original_peaks, peak_motif_pivoted, on="Name", how="left")

    # Fill NaN values with 0 for motif columns
    motif_columns = [
        col
        for col in merged_data.columns
        if col not in ["Chromosome", "Start", "End", "Score", "Name"]
    ]
    merged_data[motif_columns] = merged_data[motif_columns].fillna(0)
    # Prepare data for zarr storage
    name_values = list(merged_data["Name"].values)
    motif_values = motif_columns

    # Create sparse matrix
    motif_data_matrix = merged_data[motif_columns].values
    # Open zarr store and save data

    z = zarr.open(output_zarr, mode="w")
    z.create_dataset(
        "data",
        data=motif_data_matrix.data,
        chunks=(1000, motif_data_matrix.shape[1]),
        dtype=np.float32,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        shape=motif_data_matrix.shape,
    )
    z.create_dataset("peak_names", data=name_values)
    z.create_dataset("motif_names", data=motif_values)

    print(f"Peak motif data saved to {output_zarr}")

# Zhenhuan Jiang, 14th July, 2025
# A zarr processing method for epigenetic activation
def add_activated_peaks_to_zarr(
    zarr_file: str,
    peak_motif_bed: str,
    name: str = "activation",
):
    """
    Write a peak-motif matrix into the 'added/{name}' group of a zarr dataset,
    ensuring motif columns align exactly with the original motif list in the zarr.

    Args:
        zarr_file: Path to the main zarr file.
        peak_motif_bed: Path to added motif-annotated BED file.
        name: Increment group name under 'added/'.
    """

    # 1. Open the main zarr file and read the complete motif list
    z_main = zarr.open(zarr_file, mode="r")
    full_motif_list = list(z_main["motif_names"][:].astype(str))

    print(f"Read {len(full_motif_list)} motifs from main zarr.")

    # 2. Read the added motif BED file
    df = pd.read_csv(
        peak_motif_bed,
        sep="\t",
        header=None,
        names=["Chromosome", "Start", "End", "Motif_cluster", "Score"],
    )

    # 3. Pivot the data into a peak x motif matrix (only existing motifs)
    pivot = df.pivot_table(
        index=["Chromosome", "Start", "End"],
        columns="Motif_cluster",
        values="Score",
        fill_value=0,
    ).reset_index()

    # 4. Generate peak names in 'chr:start-end' format
    peak_names = pivot.apply(
        lambda row: f"{row['Chromosome']}:{row['Start']}-{row['End']}", axis=1
    )

    # 5. Create a complete motif matrix with all motifs from the main motif list,
    #    filling missing motifs with zeros and preserving motif order
    motif_data_complete = np.zeros((len(pivot), len(full_motif_list)), dtype=np.float32)
    present_motifs = pivot.columns[3:].tolist()
    for i, motif in enumerate(full_motif_list):
        if motif in present_motifs:
            col_idx = present_motifs.index(motif)
            motif_data_complete[:, i] = pivot[motif].values
        # Missing motifs remain zeros

    # 6. Write the complete motif matrix and peak names into the zarr 'added/{name}' group,
    #    overwriting any existing group with the same name
    z = zarr.open(zarr_file, mode="a")
    if f"added/{name}" in z:
        del z[f"added/{name}"]
    group = z.require_group(f"added/{name}")
    group.create_dataset(
        "data",
        data=motif_data_complete,
        chunks=(1000, motif_data_complete.shape[1]),
        dtype=np.float32,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    group.create_dataset("peak_names", data=np.array(peak_names, dtype=str))
    group.create_dataset("motif_names", data=np.array(full_motif_list, dtype=str))

    print(f"Written increment '{name}' to {zarr_file}/added/{name} with shape {motif_data_complete.shape}")

# Zhenhuan Jiang, 14th July, 2025
# A zarr processing method for epigenetic inhibition
def add_deletion_to_zarr(
    zarr_file: str,
    peak_bed: str,
    name: str = "inhibition",
):
    """
    Add deleted peaks from a BED file into a 'deleted/{name}' group in a zarr dataset.

    This function converts BED peaks into 'chr:start-end' format strings and stores
    them in a dataset called 'deleted_peak_names' within the specified group.

    If the dataset already exists, new peaks are appended and duplicates removed.

    Args:
        zarr_file (str): Path to the zarr dataset.
        peak_bed (str): BED file containing peaks to delete (chr, start, end).
        name (str): Name of the deletion group inside 'deleted/'.

    Example:
        add_deletion_to_zarr(
            zarr_file="original.zarr",
            peak_bed="peaks_to_delete.bed",
            name="inhibition"
        )
    """
    # Read BED file (only first 3 columns)
    df = pd.read_csv(
        peak_bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["Chromosome", "Start", "End"]
    )

    # Format peaks as strings
    peak_names = df.apply(lambda row: f"{row.Chromosome}:{row.Start}-{row.End}", axis=1).values.astype(str)

    # Open zarr group for deletion
    z = zarr.open(zarr_file, mode="a")

    # If the group exists, delete it entirely (overwrite)
    if f"deleted/{name}" in z:
        del z[f"deleted/{name}"]

    group = z.require_group(f"deleted/{name}")
    group.create_dataset("deleted_peak_names", data=peak_names)

    print(f"Written deletion '{name}' to {zarr_file}/deleted/{name}")


def zip_zarr(zarr_file):
    subprocess.run(["zip", "-r", f"{zarr_file}.zip", zarr_file], check=True)

def unzip_zarr(zarr_file):
    subprocess.run(["unzip", f"{zarr_file}.zip"], check=True)

def add_atpm(zarr_file, bed_file, celltype):
    """
    Add aTPM (ATAC-seq 'Transcript'/Count Per Million) data for a specific cell type to the zarr file.

    This function reads aTPM data from a BED file and adds it to the zarr file under the 'atpm' group.
    The aTPM values are associated with peak names in the zarr file.

    Args:
        zarr_file (str): Path to the zarr file.
        bed_file (str): Path to the BED file containing aTPM data.
        celltype (str): Name of the cell type for which the aTPM data is being added.

    The BED file should have the following columns:
    1. Chromosome
    2. Start
    3. End
    4. aTPM value

    The function creates an 'atpm' group in the zarr file if it doesn't exist and adds a dataset
    for the specified cell type under this group.
    """
    df = pd.read_csv(
        bed_file, sep="\t", header=None, names=["Chromosome", "Start", "End", "aTPM"]
    )
    df["Name"] = df.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )
    print(df)
    name_to_atpm = dict(zip(df["Name"], df["aTPM"]))
    z = zarr.open(zarr_file, mode="a")

    # Create the atpm group if it doesn't exist
    if "atpm" not in z:
        z.create_group("atpm")

    # Save aTPM data for the specific cell type
    z["atpm"].create_dataset(
        celltype,
        data=np.array([name_to_atpm[name] for name in z["peak_names"]]),
        overwrite=True,
        chunks=(1000,),
        dtype=np.float32,
    )


def add_exp(
    zarr_file, rna_file, atac_file, celltype, assembly="hg38", version=40, extend_bp=300, id_or_name="gene_id",
):
    """
    Add expression and TSS data for a specific cell type to the zarr file.

    Args:
        zarr_file (str): Path to the zarr file.
        rna_file (str): Path to the RNA file.
        atac_file (str): Path to the ATAC file.
        celltype (str): Name of the cell type for which the expression data is being added.
        assembly (str): Genome assembly (e.g., 'hg38', 'mm10').
        version (int): Version of the genome assembly.
        extend_bp (int): Number of base pairs to extend the promoter region.
        id_or_name (str): Whether to use 'gene_id' or 'gene_name' for the gene identifier.
    """
    # Initialize Gencode
    gencode = Gencode(assembly=assembly, version=version)

    # Read RNA data
    gene_exp = pd.read_csv(rna_file)
    if id_or_name == "gene_id":
        gene_exp["gene_id"] = gene_exp["gene_id"].apply(lambda x: x.split(".")[0])
        promoter_exp = pd.merge(
            gencode.gtf, gene_exp, left_on="gene_id", right_on="gene_id"
        )
    elif id_or_name == "gene_name":
        promoter_exp = pd.merge(
            gencode.gtf, gene_exp, left_on="gene_name", right_on="gene_name"
        )
    else:
        raise ValueError(f"Invalid value for id_or_name: {id_or_name}")


    # Read ATAC data
    if atac_file.endswith(".bed"):
        atac = pr.PyRanges(
            pd.read_csv(
                atac_file,
                sep="\t",
                header=None,
                names=["Chromosome", "Start", "End", "aTPM"],
            ).reset_index(),
            int64=True,
        )
    else:
        atac = pr.PyRanges(pd.read_csv(atac_file, index_col=0).reset_index(), int64=True)

    # Join ATAC and RNA data
    exp = atac.join(pr.PyRanges(promoter_exp, int64=True).extend(extend_bp), how="left").as_df()

    # Save to exp.feather for getting gene name to index
    gene_idx_info = exp.query('index_b!=-1')[['index', 'gene_name', 'Strand']].values

    
    # Process expression data
    exp = (
        exp[["index", "Strand", "TPM"]]
        .groupby(["index", "Strand"])
        .mean()
        .reset_index()
    )

    # Calculate expression and TSS
    exp_n = exp[exp.Strand == "-"].set_index("index")["TPM"].fillna(0)
    exp_p = exp[exp.Strand == "+"].set_index("index")["TPM"].fillna(0)
    exp_n[exp_n < 0] = 0
    exp_p[exp_p < 0] = 0

    exp_n_tss = (exp[exp.Strand == "-"].set_index("index")["TPM"] >= 0).fillna(False)
    exp_p_tss = (exp[exp.Strand == "+"].set_index("index")["TPM"] >= 0).fillna(False)

    tss = np.stack([exp_p_tss, exp_n_tss]).T
    exp_data = np.stack([exp_p, exp_n]).T

    # Open zarr file
    z = zarr.open(zarr_file, mode="a")

    # Create groups if they don't exist
    for group in ["expression_positive", "expression_negative", "tss"]:
        if group not in z:
            z.create_group(group)

    # Save data for the specific cell type
    peak_names = z["peak_names"][:]
    z["expression_positive"].create_dataset(
        celltype,
        data=exp_data[:, 0].astype(np.float32),
        overwrite=True,
        chunks=(1000,),
        dtype=np.float32,
    )
    z["expression_negative"].create_dataset(
        celltype,
        data=exp_data[:, 1].astype(np.float32),
        overwrite=True,
        chunks=(1000,),
        dtype=np.float32,
    )
    z["tss"].create_dataset(
        celltype,
        data=tss.astype(np.int8),
        overwrite=True,
        chunks=(1000,),
        dtype=np.int8,
    )
    z["gene_idx_info_index"] = gene_idx_info[:, 0].astype(int)
    z["gene_idx_info_name"] = gene_idx_info[:, 1].astype(str)
    z["gene_idx_info_strand"] = gene_idx_info[:, 2].astype(str)

# Zhenhuan Jiang, 4th, Aug, 2025
# A zarr processing method for ATAC-seq TSS annotation
# This function annotates TSS and dummy expression for peaks in 'added/{added_name}'
# of a Zarr dataset. It uses Gencode annotation and overlaps ATAC peaks with promoter
# regions. Expression is filled with zeros.
def add_activated_tss_to_zarr(
    zarr_file: str,
    name: str,
    atac_file: str,
    celltype: str,
    assembly: str = "hg38",
    version: int = None,
    gene_anno: Optional[Union[str, pr.PyRanges]] = None,
    extend_bp: int = 300,
):
    """
    Annotate TSS and dummy expression for peaks in 'added/{name}' of a Zarr dataset.
    Supports:
        - Gencode version/assembly (if version is provided)
        - GTF file path
        - PyRanges object

    Args:
        zarr_file (str): Path to the Zarr dataset.
        name (str): Name of the 'added' group inside Zarr.
        atac_file (str): ATAC peak file (BED or CSV).
        celltype (str): Cell type name.
        gene_anno (str or PyRanges): Gencode GTF path or PyRanges object.
        assembly (str): Genome assembly (default: "hg38").
        version (int): Gencode annotation version. If specified, overrides gene_anno.
        extend_bp (int): Promoter extension in base pairs.
    """

    if version is None and gene_anno is None:
        raise ValueError("Must specify either `gene_anno` or `version`.")
    if version is not None and gene_anno is not None:
        raise ValueError("Specify only one of `gene_anno` or `version`, not both.")

    # Load Gencode annotation
    if version is not None:
        gencode = Gencode(assembly=assembly, version=version)
        gtf_df = gencode.gtf
    elif isinstance(gene_anno, str):
        gtf_df = pr.read_gtf(gene_anno).df
    elif isinstance(gene_anno, pr.PyRanges):
        gtf_df = gene_anno.df
    else:
        raise ValueError("Please provide either a valid `version` or `gene_anno` (GTF path or PyRanges).")

    # Extract transcript TSS info
    if "Feature" in gtf_df.columns:
        gtf_df = gtf_df[gtf_df["Feature"] == "transcript"]
    gtf_df = gtf_df[["Chromosome", "Start", "End", "Strand", "gene_name"]].copy()
    gtf_df["TSS"] = gtf_df.apply(lambda row: row["Start"] if row["Strand"] == "+" else row["End"], axis=1)
    gtf_df["Start"] = gtf_df["TSS"] - 1
    gtf_df["End"] = gtf_df["TSS"]
    promoter_regions = pr.PyRanges(gtf_df[["Chromosome", "Start", "End", "Strand", "gene_name"]], int64=True).extend(extend_bp)

    # Load ATAC data
    if atac_file.endswith(".bed"):
        atac = pr.PyRanges(
            pd.read_csv(
                atac_file,
                sep="\t",
                header=None,
                names=["Chromosome", "Start", "End", "aTPM"],
            ).reset_index(),
            int64=True,
        )
    else:
        atac = pr.PyRanges(pd.read_csv(atac_file, index_col=0).reset_index(), int64=True)

    # Overlap ATAC peaks with promoters
    exp = atac.join(promoter_regions, how="left", apply_strand_suffix=False).as_df()

    # Extract gene info
    #gene_idx_info = exp.query('index_b != -1')[['index', 'gene_name', 'Strand']].drop_duplicates().values
    gene_idx_info = exp[exp["gene_name"].notnull()][['index', 'gene_name', 'Strand']].drop_duplicates().values

    # Dummy expression arrays
    all_indexes = exp["index"].unique()
    all_indexes.sort()
    exp_data = np.zeros((len(all_indexes), 2), dtype=np.float32)

    # TSS annotation arrays
    tss_p_series = pd.Series(False, index=all_indexes)
    tss_p_series.loc[exp[exp.Strand == "+"].set_index("index").index.unique()] = True

    tss_n_series = pd.Series(False, index=all_indexes)
    tss_n_series.loc[exp[exp.Strand == "-"].set_index("index").index.unique()] = True

    tss = np.stack([
        tss_p_series.sort_index().astype(np.int8).values,
        tss_n_series.sort_index().astype(np.int8).values
    ]).T

    # Write into Zarr
    z = zarr.open(zarr_file, mode="a")
    group = z[f"added/{name}"]

    for ds in [f"expression_positive/{celltype}", f"expression_negative/{celltype}", f"tss/{celltype}"]:
        if ds in group:
            del group[ds]

    group.create_dataset(f"expression_positive/{celltype}", data=exp_data[:, 0], dtype=np.float32)
    group.create_dataset(f"expression_negative/{celltype}", data=exp_data[:, 1], dtype=np.float32)
    group.create_dataset(f"tss/{celltype}", data=tss, dtype=np.int8)

    group["gene_idx_info_index"] = gene_idx_info[:, 0].astype(int)
    group["gene_idx_info_name"] = gene_idx_info[:, 1].astype(str)
    group["gene_idx_info_strand"] = gene_idx_info[:, 2].astype(str)

    print(f"TSS and dummy expression annotated for 'added/{name}' with celltype '{celltype}'.")


def add_activation_to_zarr(
    zarr_file: str,
    peak_motif_bed: str,
    atac_file: str,
    name: str,
    celltype: str,
    assembly: str = "hg38",
    version: int = None,
    gene_anno: Optional[Union[str, pr.PyRanges]] = None,
    extend_bp: int = 300,
):
    add_activated_peaks_to_zarr(
        zarr_file=zarr_file,
        peak_motif_bed=peak_motif_bed,
        name=name
    )

    add_activated_tss_to_zarr(
        zarr_file=zarr_file,
        name=name,
        atac_file=atac_file,
        celltype=celltype,
        assembly=assembly,
        version=version,
        gene_anno=gene_anno,
        extend_bp=extend_bp
    )