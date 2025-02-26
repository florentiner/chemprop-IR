"""Computes the similarity of molecular scaffolds between two datasets."""

from argparse import ArgumentParser
import os
import sys

from itertools import product
from typing import List

import math
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import scaffold_to_smiles
from chemprop.data.utils import get_data


def scaffold_similarity(smiles_1: List[str], smiles_2: List[str]):
    """
    Determines the similarity between the scaffolds of two lists of smiles strings.

    :param smiles_1: A list of smiles strings.
    :param smiles_2: A list of smiles strings.
    """
    # Get scaffolds
    scaffold_to_smiles_1 = scaffold_to_smiles(smiles_1)
    scaffold_to_smiles_2 = scaffold_to_smiles(smiles_2)

    scaffolds_1, smiles_sets_1 = zip(*scaffold_to_smiles_1.items())
    scaffolds_2, smiles_sets_2 = zip(*scaffold_to_smiles_2.items())

    smiles_to_scaffold = {smiles: scaffold for scaffold, smiles_set in scaffold_to_smiles_1.items() for smiles in smiles_set}
    smiles_to_scaffold.update({smiles: scaffold for scaffold, smiles_set in scaffold_to_smiles_2.items() for smiles in smiles_set})


    # Determine similarity
    scaffolds_1, scaffolds_2 = set(scaffolds_1), set(scaffolds_2)
    smiles_1, smiles_2 = set(smiles_1), set(smiles_2)

    all_scaffolds = scaffolds_1 | scaffolds_2
    all_smiles = smiles_1 | smiles_2

    scaffolds_intersection = scaffolds_1 & scaffolds_2
    # smiles_intersection is smiles with a scaffold that appears in both datasets
    smiles_intersection = {smiles for smiles in all_smiles if smiles_to_scaffold[smiles] in scaffolds_intersection}

    smiles_in_1_with_scaffold_in_2 = {smiles for smiles in smiles_1 if smiles_to_scaffold[smiles] in scaffolds_2}
    smiles_in_2_with_scaffold_in_1 = {smiles for smiles in smiles_2 if smiles_to_scaffold[smiles] in scaffolds_1}

    sizes_1 = np.array([len(smiles_set) for smiles_set in smiles_sets_1])
    sizes_2 = np.array([len(smiles_set) for smiles_set in smiles_sets_2])

    # Print results
    print()
    print(f'Number of molecules = {len(all_smiles):,}')
    print(f'Number of scaffolds = {len(all_scaffolds):,}')
    print()
    print(f'Number of scaffolds in both datasets = {len(scaffolds_intersection):,}')
    print(f'Scaffold intersection over union = {len(scaffolds_intersection) / len(all_scaffolds):.4f}')
    print()
    print(f'Number of molecules with scaffold in both datasets = {len(smiles_intersection):,}')
    print(f'Molecule intersection over union = {len(smiles_intersection) / len(all_smiles):.4f}')
    print()
    print(f'Number of molecules in dataset 1 = {np.sum(sizes_1):,}')
    print(f'Number of scaffolds in dataset 1 = {len(scaffolds_1):,}')
    print()
    print(f'Number of molecules in dataset 2 = {np.sum(sizes_2):,}')
    print(f'Number of scaffolds in dataset 2 = {len(scaffolds_2):,}')
    print()
    print(f'Percent of scaffolds in dataset 1 which are also in dataset 2 = {100 * len(scaffolds_intersection) / len(scaffolds_1):.2f}%')
    print(f'Percent of scaffolds in dataset 2 which are also in dataset 1 = {100 * len(scaffolds_intersection) / len(scaffolds_2):.2f}%')
    print()
    print(f'Number of molecules in dataset 1 with scaffolds in dataset 2 = {len(smiles_in_1_with_scaffold_in_2):,}')
    print(f'Percent of molecules in dataset 1 with scaffolds in dataset 2 = {100 * len(smiles_in_1_with_scaffold_in_2) / len(smiles_1):.2f}%')
    print()
    print(f'Number of molecules in dataset 2 with scaffolds in dataset 1 = {len(smiles_in_2_with_scaffold_in_1):,}')
    print(f'Percent of molecules in dataset 2 with scaffolds in dataset 1 = {100 * len(smiles_in_2_with_scaffold_in_1) / len(smiles_2):.2f}%')
    print()
    print(f'Average number of molecules per scaffold in dataset 1 = {np.mean(sizes_1):.4f} +/- {np.std(sizes_1):.4f}')
    print('Percentiles for molecules per scaffold in dataset 1')
    print(' | '.join([f'{i}% = {int(np.percentile(sizes_1, i)):,}' for i in range(0, 101, 10)]))
    print()
    print(f'Average number of molecules per scaffold in dataset 2 = {np.mean(sizes_2):.4f} +/- {np.std(sizes_2):.4f}')
    print('Percentiles for molecules per scaffold in dataset 2')
    print(' | '.join([f'{i}% = {int(np.percentile(sizes_2, i)):,}' for i in range(0, 101, 10)]))


def morgan_similarity(smiles_1: List[str], smiles_2: List[str], radius: int, sample_rate: float):
    """
    Determines the similarity between the morgan fingerprints of two lists of smiles strings.

    :param smiles_1: A list of smiles strings.
    :param smiles_2: A list of smiles strings.
    :param radius: The radius of the morgan fingerprints.
    :param sample_rate: Rate at which to sample pairs of molecules for Morgan similarity (to reduce time).
    """
    # Compute similarities
    similarities = []
    num_pairs = len(smiles_1) * len(smiles_2)

    # Sample to improve speed
    if sample_rate < 1.0:
        sample_num_pairs = sample_rate * num_pairs
        sample_size = math.ceil(math.sqrt(sample_num_pairs))
        sample_smiles_1 = np.random.choice(smiles_1, size=sample_size, replace=True)
        sample_smiles_2 = np.random.choice(smiles_2, size=sample_size, replace=True)
    else:
        sample_smiles_1, sample_smiles_2 = smiles_1, smiles_2

    sample_num_pairs = len(sample_smiles_1) * len(sample_smiles_2)

    for smile_1, smile_2 in product(sample_smiles_1, sample_smiles_2):
        mol_1, mol_2 = Chem.MolFromSmiles(smile_1), Chem.MolFromSmiles(smile_2)
        fp_1, fp_2 = AllChem.GetMorganFingerprint(mol_1, radius), AllChem.GetMorganFingerprint(mol_2, radius)
        similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
        similarities.append(similarity)
    similarities = np.array(similarities)

    # Print results
    print()
    print(f'Average dice similarity = {np.mean(similarities):.4f} +/- {np.std(similarities):.4f}')
    print(f'Minimum dice similarity = {np.min(similarities):.4f}')
    print(f'Maximum dice similarity = {np.max(similarities):.4f}')
    print()
    print('Percentiles for dice similarity')
    print(' | '.join([f'{i}% = {np.percentile(similarities, i):.4f}' for i in range(0, 101, 10)]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path_1', type=str, required=True,
                        help='Path to first data CSV file')
    parser.add_argument('--data_path_2', type=str, required=True,
                        help='Path to second data CSV file')
    parser.add_argument('--use_compound_names_1', action='store_true', default=False,
                        help='Whether data_path_1 has compound names in addition to smiles')
    parser.add_argument('--use_compound_names_2', action='store_true', default=False,
                        help='Whether data_path_2 has compound names in addition to smiles')
    parser.add_argument('--similarity_measure', type=str, required=True, choices=['scaffold', 'morgan'],
                        help='Similarity measure to use to compare the two datasets')
    parser.add_argument('--radius', type=int, default=3,
                        help='Radius of Morgan fingerprint')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Rate at which to sample pairs of molecules for Morgan similarity (to reduce time)')
    args = parser.parse_args()

    data_1 = get_data(path=args.data_path_1, use_compound_names=args.use_compound_names_1)
    data_2 = get_data(path=args.data_path_2, use_compound_names=args.use_compound_names_2)

    if args.similarity_measure == 'scaffold':
        scaffold_similarity(data_1.smiles(), data_2.smiles())
    elif args.similarity_measure == 'morgan':
        morgan_similarity(data_1.smiles(), data_2.smiles(), args.radius, args.sample_rate)
    else:
        raise ValueError(f'Similarity measure "{args.similarity_measure}" not supported.')
