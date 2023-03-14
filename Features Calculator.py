# %% libraries
import os
import logging
from tqdm import tqdm
from collections import OrderedDict
from Bio.PDB import PDBList, PDBParser
from PyBioMed.PyProtein import CTD
import numpy as np
import pandas as pd
from utils import (
    get_baricentro,
    get_atoms_coord,
    get_covariance,
    inizializza_dict_amm,
    feature_conteggio,
    specific_feature,
)

# initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create a dir to save the pdb filese
if "pdb-files" not in os.listdir():
    os.mkdir("pdb-files")
path_pdb = "pdb-files"

# %%
list_Bar = list(np.arange(8, 17))  # set range for sampling r1
list_Ring = list(np.arange(3, 7))  # set range for sampling r2

amm_names = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]  # amino acids

# You can use a dict to convert three letter code to one letter code
d3to1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

path_dir = ""
# %%
# read file
dataset = pd.read_excel(os.path.join(path_dir, "data/dataset.xlsx")).set_index(
    "Unnamed: 0"
)[["PDB", "Em", "pH"]]
proteins_PDB = list(OrderedDict.fromkeys(dataset["PDB"]))  # list of PDB ID used

# read file with amino acids features
table_amm = pd.read_csv(
    os.path.join(path_dir, "data/tableAmm.txt"), sep="\t", index_col=1
)
table_amm.index = [el.upper() for el in table_amm.index]
table_amm = table_amm.iloc[:, 1:]

# %% start "for cycle" to consider different combination of radii
for bar in list_Bar:
    for ring in list_Ring:
        logger.info(
            f"Starting feature computation for dataset of barycenter: {str(bar)} $\AA$ and ring_radius: {str(ring)} $\AA$"
        )

        # %%
        df_total = pd.DataFrame()  # initialize pandas dataframe to save results
        names = list()  # initialize list to save name for dataframe columns
        # %% access to pdb database
        pdbl = PDBList()

        # %% "for cycle" on each protein

        for idx, name_protein in tqdm(enumerate(proteins_PDB), desc="Flavoproteins"):
            # download pdb file
            pdbl.retrieve_pdb_file(name_protein, pdir=path_pdb, file_format="pdb")
            parser = PDBParser(PERMISSIVE=True, QUIET=True)
            structure = parser.get_structure(
                os.path.join(path_pdb, name_protein.lower()),
                os.path.join(path_pdb, f"pdb{name_protein.lower()}.ent"),
            )

            # generation dict
            dict_residues = dict()  # inizialize dict for residues
            Cof_coord = dict()  # inizialize dict for barycenter coordinate
            Cof_coords = (
                dict()
            )  # inizialize dict for coordinate for each atom of the isoalloxazine ring
            N1_coord = dict()  # inizialize dict N1 coordinate
            N3_coord = dict()  # inizialize dict N3 coordinate
            N5_coord = dict()  # inizialize dict N5 coordinate

            # % start for cycle
            for model in structure:
                # header of the pdb file
                header = structure.header
                chains = model.get_chains()

                # scan on chains
                for chain in chains:
                    residue_names = [
                        residue.resname for residue in chain.get_residues()
                    ]  # check on FMN and FAD

                    if "FAD" not in residue_names and "FMN" not in residue_names:
                        logger.info("NO FAD e/o FMN found")
                        continue
                    else:
                        names.append(name_protein + "chain_" + chain.id)

                    dict_residues[chain.id] = dict()

                    # scan on residues
                    for residue in chain.get_residues():
                        if residue.resname in amm_names:
                            # dizionario con chiavi gli id e valori i nomi degli amminoacidi
                            dict_residues[chain.id][residue.id[1]] = residue.resname
                        elif residue.resname == "FMN" or residue.resname == "FAD":
                            # save info about the cofactor
                            for el in residue.get_atoms():
                                if el.id == "N1":
                                    N1_el = el.coord
                                elif el.id == "N3":
                                    N3_el = el.coord
                                elif el.id == "N5":
                                    N5_el = el.coord

                            if residue.resname == "FMN":
                                FAD = 0
                                # FMN cofactor
                                ind1 = 0
                                ind2 = 18
                            else:
                                FAD = 1
                                # FAD cofactor
                                ind1 = 23
                                ind2 = 40

                            # calculate barycenter coordinate
                            Cof_coord_el = get_baricentro(residue, ind1, ind2)
                            # calculate ring's atoms coordinate
                            Cof_coords_el = get_atoms_coord(residue, ind1, ind2)

                    # features about amino acids count
                    dict_cont = inizializza_dict_amm(amm_names)

                    dict_cont, N5_nearest_res, N5_3_nearest_res = feature_conteggio(
                        dict_cont,
                        chain,
                        Cof_coord_el,
                        Cof_coords_el,
                        N5_el,
                        bar,
                        ring,
                        dict_residues,
                        amm_names,
                    )

                    # count number of total amino acids
                    total_bar = sum(
                        [dict_cont[f"Bar.{nome}"] for nome in amm_names]
                    )  # respect the r1 sphere
                    total_protein = sum(
                        [dict_cont[f"Protein.{nome}"] for nome in amm_names]
                    )  # respect the entire aa sequence
                    total_ring = sum(
                        [dict_cont[f"Ring.{nome}"] for nome in amm_names]
                    )  # respect the r2 sphere

                    # rows to avoid any null divisions later
                    if total_bar == 0:
                        total_bar = 1
                    if total_protein == 0:
                        total_protein = 1
                    if total_ring == 0:
                        total_ring = 1

                    # r1 features calculation
                    for col in table_amm.columns:  # 28+28
                        values = table_amm[col]
                        val_feature = np.sum(
                            [
                                values[nome] * dict_cont[f"Bar.{nome}"]
                                for nome in amm_names
                            ]
                        )
                        dict_cont[f"Bar.{col}"] = val_feature

                    # protein features calculation
                    for col in table_amm.columns:
                        values = table_amm[col]
                        val_feature = sum(
                            [
                                values[nome] * dict_cont[f"Protein.{nome}"]
                                for nome in amm_names
                            ]
                        )
                        dict_cont[f"Protein.{col}"] = val_feature

                    # r2 features calculation
                    for col in table_amm.columns:
                        values = table_amm[col]
                        val_feature = np.sum(
                            [
                                values[nome] * dict_cont[f"Ring.{nome}"]
                                for nome in amm_names
                            ]
                        )
                        dict_cont[f"Ring.{col}"] = val_feature

                    # nearest amino acid respect N5
                    if N5_nearest_res:
                        for col in table_amm.columns:  # 28
                            value = table_amm.loc[N5_nearest_res, col]
                            dict_cont[f"N5_nearest.{col}"] = value

                    # 3 nearest amino acid respect N5
                    if N5_3_nearest_res:
                        for col in table_amm.columns:  # 28
                            value = table_amm.loc[N5_3_nearest_res, col].sum()
                            dict_cont[f"Around_N5.{col}"] = value

                    # add some specific feature
                    dict_cont = specific_feature(
                        dict_cont, prefix="Bar.", mean=True, total=total_bar
                    )
                    dict_cont = specific_feature(
                        dict_cont, prefix="Protein.", mean=True, total=total_protein
                    )
                    dict_cont = specific_feature(
                        dict_cont, prefix="Ring.", mean=True, total=total_bar
                    )

                    dict_cont["PDB"] = name_protein

                    # add some information from the pdb file header
                    dict_cont["organism"] = header["source"]["1"]["organism_scientific"]

                    # features by PyBioMed
                    lista_fasta = ""
                    for residue_name in residue_names:
                        if residue_name in d3to1.keys():
                            lista_fasta = lista_fasta + d3to1[residue_name]
                    protein_descriptor = CTD.CalculateC(lista_fasta)

                    # dicts merge between "aa count features" and "PyBioMed features"
                    dict_cont = {**dict_cont, **protein_descriptor}

                    ###end features calculation !!
                    df = pd.DataFrame.from_dict(dict_cont, orient="index")

                    # df_total update !!
                    df_total = pd.concat([df_total, df], axis=1)

        # %% end of for cycle on proteins list
        df_total = df_total.fillna(0)
        # columns name update
        df_total.columns = names
        df_total = df_total.transpose()

        cols = df_total.columns.tolist()
        df_total = df_total[cols]
        logger.info(
            f"Saving Features for dataset of barycenter: {str(bar)} $\AA$ and ring_radius: {str(ring)} $\AA$"
        )

        df_total = df_total.groupby("PDB").agg(
            lambda x: np.round(np.mean(x), 2)
        )  # groupby for PDB ID if a protein has 2+ chains

        dataset2 = dataset.join(
            df_total, on="PDB"
        )  # join pandas function to add information about Em and pH to the features dataset
        dataset2.to_excel(
            os.path.join(
                path_dir,
                f"dataset_features/dataset_protein_{str(bar)}_{str(ring)}.xlsx",
            )
        )  # save final dataset
