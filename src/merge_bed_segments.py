#!/usr/bin/env python3
"""
BED Segment Merger (No Genetic Map Version)

This standalone script takes a raw BED file (from ArchaicSeeker3 with merge_distance=0)
and merges nearby segments of the same ancestry label.

This version is designed for BED files WITHOUT genetic map (NO cM columns).

Usage:
    python merge_bed_segments.py --input raw.bed --output merged.bed --merge-distance 5000

Features:
    - Merges segments of the same label within specified distance
    - Uses 10-column BED format (NO cM columns)
    - Recalculates statistics (num_snps, avg_prob) after merging
    - Supports exact merge mode using SNP details file
"""

import argparse
import pandas as pd
import numpy as np
import sys
import logging
import gzip
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXACT MERGE: SNP-level detail functions for exact merge replication
# ============================================================================

def load_snp_details(snp_details_path):
    """
    Load SNP details file (gzipped TSV)

    Args:
        snp_details_path: Path to .raw.snps.gz file

    Returns:
        dict: {(chr, sample_hap_id, start_pos, end_pos): snp_data_dict}
    """
    if not os.path.exists(snp_details_path):
        raise FileNotFoundError(f"SNP details file not found: {snp_details_path}")

    logger.info(f"Loading SNP details from {snp_details_path}")

    with gzip.open(snp_details_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t')

    # Build lookup dictionary using composite key
    snp_details_dict = {}
    for _, row in df.iterrows():
        key = (row['chr'], row['sample_hap_id'], row['start_pos'], row['end_pos'])
        snp_details_dict[key] = {
            'positions': np.array([int(x) for x in row['snp_positions'].split(',')]),
            'states': np.array([int(x) for x in row['snp_states'].split(',')]),
            'prob_label1': np.array([float(x) for x in row['snp_prob_label1'].split(',')]),
            'prob_label2': np.array([float(x) for x in row['snp_prob_label2'].split(',')]),
        }

    logger.info(f"Loaded SNP details for {len(snp_details_dict)} segments")
    return snp_details_dict


def merge_segments_exact(df, snp_details_dict, merge_distance,
                         min_snps_per_segment=2,
                         mosaic_minority_threshold=0.20):
    """
    Exact merge replication using SNP-level details

    This function 100% replicates the main program's merge logic:
    1. Merges adjacent segments regardless of label (not grouped by label!)
    2. Recalculates label based on combined SNP counts
    3. Recalculates avg_prob from original SNP probabilities
    4. Applies min_snps_per_segment filter after merging

    Args:
        df: Raw BED DataFrame
        snp_details_dict: SNP details lookup dictionary
        merge_distance: Maximum distance (bp) to merge segments
        min_snps_per_segment: Minimum SNPs after merging
        mosaic_minority_threshold: Threshold for Mosaic classification

    Returns:
        DataFrame: Merged segments
    """
    if df.empty:
        return df

    merged_segments = []

    # Group by sample_hap_id only (NOT by label!)
    for sample_hap_id, group in df.groupby('sample_hap_id'):
        group = group.sort_values('start_pos').copy()

        if len(group) == 0:
            continue

        # Start with first segment
        current_seg_rows = [group.iloc[0]]

        for i in range(1, len(group)):
            next_seg = group.iloc[i]
            current_end = current_seg_rows[-1]['end_pos']
            gap = next_seg['start_pos'] - current_end

            if gap <= merge_distance:
                # Merge: add to current accumulated segments
                current_seg_rows.append(next_seg)
            else:
                # Don't merge: process accumulated segments
                merged_seg = _process_merged_segment_exact(
                    current_seg_rows, snp_details_dict,
                    min_snps_per_segment, mosaic_minority_threshold
                )
                if merged_seg is not None:
                    merged_segments.append(merged_seg)

                # Start new accumulation
                current_seg_rows = [next_seg]

        # Process last accumulated segments
        merged_seg = _process_merged_segment_exact(
            current_seg_rows, snp_details_dict,
            min_snps_per_segment, mosaic_minority_threshold
        )
        if merged_seg is not None:
            merged_segments.append(merged_seg)

    if not merged_segments:
        return pd.DataFrame()

    result_df = pd.DataFrame(merged_segments)
    result_df = result_df.sort_values(['sample_hap_id', 'chr', 'start_pos'])

    logger.info(f"Exact merge: {len(df)} -> {len(result_df)} segments")
    return result_df


def _process_merged_segment_exact(seg_rows, snp_details_dict,
                                   min_snps_per_segment,
                                   mosaic_minority_threshold):
    """
    Process merged segment - exact replication of main program logic

    Args:
        seg_rows: List of segment rows (pd.Series) to be merged
        snp_details_dict: SNP details lookup
        min_snps_per_segment: Min SNPs threshold
        mosaic_minority_threshold: Mosaic threshold

    Returns:
        dict or None: Merged segment data
    """
    # Collect all SNPs from all segments being merged
    all_positions = []
    all_states = []
    all_prob1 = []
    all_prob2 = []

    for seg_row in seg_rows:
        # Lookup SNP details using composite key
        key = (seg_row['chr'], seg_row['sample_hap_id'],
               seg_row['start_pos'], seg_row['end_pos'])

        if key not in snp_details_dict:
            logger.warning(f"SNP details not found for segment: {key}")
            continue

        snp_data = snp_details_dict[key]
        all_positions.extend(snp_data['positions'].tolist())
        all_states.extend(snp_data['states'].tolist())
        all_prob1.extend(snp_data['prob_label1'].tolist())
        all_prob2.extend(snp_data['prob_label2'].tolist())

    if len(all_positions) == 0:
        return None

    # Convert to numpy arrays
    all_positions = np.array(all_positions)
    all_states = np.array(all_states)
    all_prob1 = np.array(all_prob1)
    all_prob2 = np.array(all_prob2)

    # Count SNPs by state
    n1 = np.sum(all_states == 1)
    n2 = np.sum(all_states == 2)
    total_snps = n1 + n2

    # Apply min_snps filter (EXACTLY as in main program)
    if total_snps < min_snps_per_segment:
        return None

    # Re-judge label (EXACTLY as in main program)
    label = 0
    if n1 > 0 and n2 > 0:
        minority_ratio = min(n1, n2) / total_snps
        if minority_ratio >= mosaic_minority_threshold:
            label = 3  # Mosaic
        else:
            label = 1 if n1 >= n2 else 2
    elif n1 > 0:
        label = 1
    elif n2 > 0:
        label = 2

    if label == 0:
        return None

    # Recalculate avg_prob (EXACTLY as in main program)
    prob_values = []
    if label == 1:
        # Only use prob_label1 for SNPs with state==1
        prob_values = all_prob1[all_states == 1]
    elif label == 2:
        # Only use prob_label2 for SNPs with state==2
        prob_values = all_prob2[all_states == 2]
    elif label == 3:
        # Use both: prob_label1 for state==1, prob_label2 for state==2
        prob_values = np.concatenate([
            all_prob1[all_states == 1],
            all_prob2[all_states == 2]
        ])

    avg_prob = np.mean(prob_values) if len(prob_values) > 0 else np.nan

    # Build merged segment (10 columns, NO cM)
    first_seg = seg_rows[0]
    last_seg = seg_rows[-1]

    return {
        'chr': first_seg['chr'],
        'start_pos': int(np.min(all_positions)),
        'end_pos': int(np.max(all_positions)),
        'haplotype': first_seg['haplotype'],
        'ancestry_label': label,
        'num_snps': total_snps,
        'avg_prob': avg_prob,
        'archaic_snps': n1,
        'african_snps': n2,
        'sample_hap_id': first_seg['sample_hap_id']
    }


# ============================================================================
# Approximate merge functions (used when SNP details file is not available)
# ============================================================================

def read_bed_file(filepath):
    """
    Read a 10-column BED file without header (NO cM columns)

    Args:
        filepath: Path to BED file

    Returns:
        DataFrame with named columns
    """
    column_names = [
        'chr', 'start_pos', 'end_pos',
        'haplotype', 'ancestry_label', 'num_snps', 'avg_prob',
        'archaic_snps', 'african_snps', 'sample_hap_id'
    ]

    try:
        df = pd.read_csv(filepath, sep='\t', header=None, names=column_names)
        logger.info(f"Read {len(df)} segments from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error reading BED file: {e}")
        sys.exit(1)


def filter_segments(df, min_snps=None, min_prob=None, max_label=None):
    """
    Filter segments by various criteria

    Args:
        df: Input DataFrame
        min_snps: Minimum number of SNPs (optional)
        min_prob: Minimum average probability (optional)
        max_label: Maximum ancestry label to keep (optional, e.g., 2 to exclude Mosaic)

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)

    if min_snps is not None:
        df = df[df['num_snps'] >= min_snps]
        logger.info(f"Filtered by min_snps={min_snps}: {original_count} -> {len(df)}")
        original_count = len(df)

    if min_prob is not None:
        df = df[df['avg_prob'] >= min_prob]
        logger.info(f"Filtered by min_prob={min_prob}: {original_count} -> {len(df)}")
        original_count = len(df)

    if max_label is not None:
        df = df[df['ancestry_label'] <= max_label]
        logger.info(f"Filtered by max_label={max_label}: {original_count} -> {len(df)}")

    return df


def merge_segments(df, merge_distance, recalculate_stats=True):
    """
    Merge nearby segments of the same ancestry label (approximate method)

    Args:
        df: Input DataFrame with segments
        merge_distance: Maximum distance (bp) to merge segments
        recalculate_stats: Whether to recalculate num_snps and avg_prob

    Returns:
        DataFrame with merged segments
    """
    if df.empty:
        return df

    # Group by sample_hap_id and ancestry_label
    merged_segments = []

    for (sample_hap_id, label), group in df.groupby(['sample_hap_id', 'ancestry_label']):
        # Sort by start position
        group = group.sort_values('start_pos').copy()

        if len(group) == 0:
            continue

        # Initialize first segment
        current_seg = group.iloc[0].to_dict()

        for i in range(1, len(group)):
            next_seg = group.iloc[i].to_dict()
            gap = next_seg['start_pos'] - current_seg['end_pos']

            if gap <= merge_distance:
                # Merge: extend current segment
                current_seg['end_pos'] = max(current_seg['end_pos'], next_seg['end_pos'])

                if recalculate_stats:
                    # Sum SNP counts
                    current_seg['num_snps'] += next_seg['num_snps']
                    current_seg['archaic_snps'] += next_seg['archaic_snps']
                    current_seg['african_snps'] += next_seg['african_snps']

                    # Weighted average of probabilities
                    total_snps = current_seg['num_snps']
                    current_seg['avg_prob'] = (
                        current_seg['avg_prob'] * (total_snps - next_seg['num_snps']) +
                        next_seg['avg_prob'] * next_seg['num_snps']
                    ) / total_snps if total_snps > 0 else current_seg['avg_prob']
            else:
                # Gap too large, save current and start new
                merged_segments.append(current_seg)
                current_seg = next_seg.copy()

        # Add the last segment
        merged_segments.append(current_seg)

    if not merged_segments:
        return pd.DataFrame()

    # Convert back to DataFrame
    merged_df = pd.DataFrame(merged_segments)

    # Ensure correct column order (10 columns, NO cM)
    column_order = [
        'chr', 'start_pos', 'end_pos',
        'haplotype', 'ancestry_label', 'num_snps', 'avg_prob',
        'archaic_snps', 'african_snps', 'sample_hap_id'
    ]
    merged_df = merged_df[column_order]

    # Sort by sample_hap_id, chr, start_pos
    merged_df = merged_df.sort_values(['sample_hap_id', 'chr', 'start_pos'])

    logger.info(f"Merged segments: {len(df)} -> {len(merged_df)} (merge_distance={merge_distance})")

    return merged_df


def write_bed_file(df, filepath):
    """
    Write BED file without header

    Args:
        df: DataFrame to write
        filepath: Output file path
    """
    try:
        df.to_csv(filepath, sep='\t', index=False, header=False)
        logger.info(f"Written {len(df)} segments to {filepath}")
    except Exception as e:
        logger.error(f"Error writing BED file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Merge BED segments from ArchaicSeeker3 output (No Genetic Map version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic merge with 5kb distance
  python merge_bed_segments.py -i raw.bed -o merged.bed -d 5000

  # Filter before merging
  python merge_bed_segments.py -i raw.bed -o merged.bed -d 5000 --min-snps 5 --min-prob 0.8

  # Exclude Mosaic segments (label=3)
  python merge_bed_segments.py -i raw.bed -o merged.bed -d 5000 --max-label 2

  # No merge (just filter and reformat)
  python merge_bed_segments.py -i raw.bed -o filtered.bed -d 0 --min-snps 10
        '''
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input BED file (raw segments)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output BED file (merged segments)')
    parser.add_argument('-d', '--merge-distance', type=int, default=5000,
                        help='Maximum distance (bp) to merge segments (default: 5000)')

    # Filtering options
    parser.add_argument('--min-snps', type=int, default=None,
                        help='Filter: minimum number of SNPs per segment')
    parser.add_argument('--min-prob', type=float, default=None,
                        help='Filter: minimum average probability per segment')
    parser.add_argument('--max-label', type=int, default=None,
                        help='Filter: maximum ancestry label (e.g., 2 to exclude Mosaic=3)')

    # Advanced options
    parser.add_argument('--no-recalculate', action='store_true',
                        help='Do not recalculate statistics after merging')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("BED Segment Merger (No Genetic Map Version)")
    logger.info("=" * 60)

    # Read input
    df = read_bed_file(args.input)

    # Check for SNP details file for exact merge mode
    snp_details_path = args.input.replace('.raw.bed', '.raw.snps.gz')
    use_exact_merge = False
    snp_details_dict = None

    if os.path.exists(snp_details_path):
        logger.info("=" * 60)
        logger.info("EXACT MERGE MODE ENABLED")
        logger.info(f"Found SNP details file: {snp_details_path}")
        logger.info("Will use 100% exact merge replication")
        logger.info("=" * 60)
        use_exact_merge = True
        try:
            snp_details_dict = load_snp_details(snp_details_path)
        except Exception as e:
            logger.error(f"Error loading SNP details: {e}")
            logger.warning("Falling back to approximate merge mode")
            use_exact_merge = False
    else:
        logger.info("=" * 60)
        logger.info("APPROXIMATE MERGE MODE")
        logger.info(f"SNP details file not found: {snp_details_path}")
        logger.info("Using approximate merge (results may differ slightly from main program)")
        logger.info("=" * 60)

    # Filter if requested (only in approximate mode)
    if not use_exact_merge:
        if args.min_snps or args.min_prob or args.max_label:
            logger.info("Applying filters...")
            df = filter_segments(df, args.min_snps, args.min_prob, args.max_label)
    else:
        if args.min_snps or args.min_prob or args.max_label:
            logger.warning("Filtering options are ignored in exact merge mode")
            logger.warning("Exact merge applies fixed filters as in main program")

    # Merge
    if args.merge_distance > 0:
        logger.info(f"Merging segments (merge_distance={args.merge_distance})...")
        if use_exact_merge:
            # EXACT MERGE: Use exact replication
            df = merge_segments_exact(
                df, snp_details_dict, args.merge_distance,
                min_snps_per_segment=2,
                mosaic_minority_threshold=0.20
            )
        else:
            # Approximate merge
            df = merge_segments(df, args.merge_distance, recalculate_stats=not args.no_recalculate)
    else:
        logger.info("Merge distance is 0, skipping merge step")

    # Write output
    write_bed_file(df, args.output)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)

    # Summary
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Total segments: {len(df)}")


if __name__ == '__main__':
    main()
