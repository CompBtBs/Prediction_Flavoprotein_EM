# function used in the main 
from sklearn.pipeline import Pipeline, TransformerMixin
import numpy as np
from numpy.linalg import norm as norm2
from sklearn.ensemble import IsolationForest


def RemoveOutliar(X, y):
    clf = IsolationForest(random_state=0).fit(X)
    X_ = X.copy()
    y_ = y.copy()
    X_ = X_[clf.predict(X) == 1, :]
    y_ = y_[clf.predict(X) == 1]
    return X_, y_


class RHCF:
    def __init__(self, covariation=0.99):
        self.covariation = 0.99
        pass

    def fit(self, X, y=None):
        Df_corr = np.abs(np.corrcoef(np.transpose(X)))
        upper_tri = np.triu(Df_corr, k=1)
        to_drop = [
            i for i in range(X.shape[1]) if any(upper_tri[:, i] >= self.covariation)
        ]
        self.to_keep = [i for i in range(X.shape[1]) if i not in to_drop]
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return X_[:, self.to_keep]


# %% function to calculate barycenter
def get_baricentro(element, el1, el2):
    if el2 != -1:
        atoms_coord = [el.coord for el in element.child_list[el1:el2]]
    else:
        atoms_coord = [el.coord for el in element.child_list]

    return np.mean(atoms_coord, axis=0)


# %% function to get each atom from a molecule
def get_atoms_coord(element, el1, el2):
    if el2 != -1:
        atoms_coord = [el.coord for el in element.child_list[el1:el2]]
    else:
        atoms_coord = [el.coord for el in element.child_list]
    return atoms_coord

# %%
def get_covariance(element, el1, el2):
    atoms_coord_tot = [el.coord for el in element.child_list]  # ring's atoms coordinate
    baricenter_tot = np.mean(atoms_coord_tot, axis=0)  # isoalloxazine ring barycenter

    weight_total = 0
    Sxx = 0
    Syy = 0
    Szz = 0
    Sxy = 0
    Sxz = 0
    Syz = 0

    for el in element.child_list[el1:el2]:
        if el.id.startswith("N"):
            weight_total += 14.01
            weight = 14.01
        elif el.id.startswith("C"):
            weight_total += 12.01
            weight = 12.01
        else:
            weight_total += 16
            weight = 16

        Sxx += weight * (el.coord[0] - baricenter_tot[0]) ** 2
        Syy += weight * (el.coord[1] - baricenter_tot[1]) ** 2
        Szz += weight * (el.coord[2] - baricenter_tot[2]) ** 2
        Sxy += (
            weight
            * (el.coord[0] - baricenter_tot[0])
            * (el.coord[1] - baricenter_tot[1])
        )
        Sxz += (
            weight
            * (el.coord[0] - baricenter_tot[0])
            * (el.coord[2] - baricenter_tot[2])
        )
        Syz += (
            weight
            * (el.coord[1] - baricenter_tot[1])
            * (el.coord[2] - baricenter_tot[2])
        )

    S = [[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]]

    S = np.array(S) / weight_total
    eigS, matrix = np.linalg.eig(S)

    return eigS


# %% inizialize dict for amino acids count
def inizializza_dict_amm(nomi_amm):
    dict_cont = dict()
    for el in nomi_amm:
        dict_cont["Bar." + el] = 0
        dict_cont["Protein." + el] = 0
        dict_cont["Ring." + el] = 0
    return dict_cont


# %%
def specific_feature(dict_cont, prefix="", mean=True, total=None):
    """
    Function for specific features calculation:

    Inputs:
        dict_cont: dict to save features
        prefix : prefix prefix to be added to the feature name
        mean: if mean is true, the features will be calculated in function of...
    """

    apolari = ["ALA", "PHE", "GLY", "ILE", "LEU", "MET", "PRO", "TRP", "VAL"]
    polari = ["ASP", "GLU", "SER", "THR", "CYS", "TYR"]
    aromatici = ["TYR", "TRP", "PHE"]

    if prefix != "":
        apolari = [prefix + el for el in apolari]
        polari = [prefix + el for el in polari]
        aromatici = [prefix + el for el in aromatici]

    # other specific features
    dict_cont[prefix + "NumAMM"] = total
    dict_cont[prefix + "RESNEG"] = dict_cont[prefix + "GLU"] + dict_cont[prefix + "ASP"]
    dict_cont[prefix + "RESPOS"] = dict_cont[prefix + "ARG"] + dict_cont[prefix + "LYS"]
    dict_cont[prefix + "FormalCharge"] = (
        dict_cont[prefix + "RESPOS"] - dict_cont[prefix + "RESNEG"]
    )

    ###
    dict_cont[prefix + "ResApolari"] = np.sum([dict_cont[el] for el in apolari])
    dict_cont[prefix + "ResPolari"] = np.sum([dict_cont[el] for el in polari])
    dict_cont[prefix + "ResAromatici"] = np.sum([dict_cont[el] for el in aromatici])

    return dict_cont


# %%
def feature_conteggio(
    dict_cont,
    chain,
    Cof_coord_el,
    Cof_coords_el,
    N5_el,
    min_distBarycenter,
    min_distRing,
    dict_residues,
    nomi_amm,
):
    """
    Amino acid count function:
        1. respect the entire aa sequence
        2. respect r1 sphere (barycenter)
        3. respect r2 sphere
    Inputs:
        dict_cont: dict to save the features
        chain: chain considered
        Cof_coord_el: barycenter coordinate
        Cof_coords_el: coordinate for each atom of the isoalloxazine ring
        N5_el: N5 coordinate
        min_distBarycenter: min distance respect to the barycenter
        min_distRing: min distance respect to any atom of the isoalloxazine ring
        dict_residues: dict for residue information (id and name)
        nomi_amm: amino acid list
    """
    min_dist_amm = 1000000  # default value
    chain_amms = [el for el in chain.get_residues() if el.resname in nomi_amm]
    lenChain = len(chain_amms) - 1

    # count for Oxigen, Nitrogen and Carbon respect the r2 sphere
    dict_cont["Oxigen_around"] = 0
    dict_cont["Nitrogen_around"] = 0
    dict_cont["Carbon_around"] = 0

    # inizialize variable as 0 or None to avoid code interruption
    k = 0
    min_id_amms = None
    N5_nearest_res = None

    # %for cycle on the amino acids
    for residue in chain_amms:
        name_residue = residue.resname
        atoms = [atom for atom in residue.get_atoms()]

        # count for protein amino acids
        dict_cont["Protein." + name_residue] += 1

        # get atoms present in the r1 sphere
        for atom in atoms:
            if norm2(atom.coord - Cof_coord_el) < min_distBarycenter:
                dict_cont["Bar." + name_residue] += 1
                break  # it's sufficient that just one of the amino acid atoms is in the sphere to add the aa to the dict

        #### get atoms present in the r2 sphere
        stop_cont = 0
        for atom in atoms:
            for el2 in Cof_coords_el:
                if norm2(atom.coord - el2) < min_distRing:
                    dict_cont["Ring." + name_residue] += 1
                    stop_cont = 1
                    break
            if stop_cont:
                break

        for atom in atoms:
            for el2 in Cof_coords_el:
                if norm2(atom.coord - el2) < min_distRing:
                    # check if the one of the sphere2 isoloxazine ring atom is near O,N or C atom
                    if atom.element == "O":
                        dict_cont["Oxigen_around"] += 1
                    if atom.element == "N":
                        dict_cont["Nitrogen_around"] += 1
                    if atom.element == "C":
                        dict_cont["Carbon_around"] += 1
                    break

        # get nearest atom to N5 and get the related residue
        for atom in atoms:
            distanza = norm2(atom.coord - N5_el)
            if distanza < min_dist_amm:
                N5_nearest_res = name_residue
                min_dist_amm = distanza
                k_id = k

        k = k + 1

    if min_dist_amm != 1000000:
        if k_id == 0:
            min_id_amms = [chain_amms[k_id + i].id[1] for i in range(0, 2)]
        elif lenChain == k_id:
            min_id_amms = [chain_amms[k_id + i].id[1] for i in range(-1, 1)]
        else:
            min_id_amms = [chain_amms[k_id + i].id[1] for i in range(-1, 2)]

    if min_dist_amm != 1000000:
        N5_3_nearest_res = [dict_residues[chain.id][el] for el in min_id_amms]
    else:
        N5_3_nearest_res = None

    # print the 3 amino acid nearest N5
    ("Name of 3 amminoacids nearest to N5: ", N5_3_nearest_res)

    return dict_cont, N5_nearest_res, N5_3_nearest_res
