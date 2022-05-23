# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:26:19 2022

@author: anton
"""
from collections import OrderedDict
from Bio.PDB import PDBList,PDBParser
from PyBioMed.PyProtein import CTD
import numpy as np
import pandas as pd
from utils import get_baricentro,get_atoms_coord
from numpy.linalg import norm as norm2


def check_ligand_ring(name_protein, chain, Cof_coords_el):
         ligands=[el for el in chain.get_residues() if el.resname not in amm_names and el not in res_list_acc]
         list_ligands=[]
         for ligand in ligands:
             name_ligand=ligand.resname
             if name_ligand != "HOH" and name_ligand != "FAD" and len(name_ligand) != 2:
                 atoms=[atom for atom in ligand.get_atoms()]
                 for atom in atoms:
                     for coord in Cof_coords_el:
                        if norm2(atom.coord-coord)<min_dist_ring:
                            list_ligands.append(name_ligand)
                            dict_pdb_ligand_ring_atoms[name_protein]=list_ligands
                            return dict_pdb_ligand_ring_atoms
         return dict_pdb_ligand_ring_atoms

        
def check_ligand_bar(name_protein, chain, Cof_coords_el):
         ligands=[el for el in chain.get_residues() if el.resname not in amm_names and el!="FAD" and el!="FMN"]
         list_ligands=[]
         for ligand in ligands:
             name_ligand=ligand.resname
             if name_ligand != "HOH" and name_ligand != "FAD" and len(name_ligand) != 2:
                 atoms=[atom for atom in ligand.get_atoms()]
                 for atom in atoms:
                        if norm2(atom.coord-Cof_coord_el)<min_dist_bar:
                            list_ligands.append(name_ligand)
                            dict_pdb_ligand_ring_atoms[name_protein]=list_ligands
                            return dict_pdb_ligand_bar
         return dict_pdb_ligand_bar
            

amm_names=["ALA","ARG","ASN","ASP","CYS",
          "GLN","GLU","GLY","HIS","ILE",
          "LEU","LYS","MET","PHE","PRO",
          "SER","THR","TRP","TYR","VAL"]  #nomi amminoacidi considerati

res_list_acc=["FAD","FMN","NAD","HOH"]
path_dir=""
min_dist_ring=4
min_dist_bar=12
dict_pdb_ligand_ring_atoms=dict()
dict_pdb_ligand_bar=dict()
#%% leggo il file dove sono presente le proteine da considerare
dataset=pd.read_excel(path_dir+"data/dataset.xlsx",usecols=(0,3,4))
features_DS=pd.read_excel(path_dir+"data/DS_Visualizer_Features.xlsx").drop("Unnamed: 0", axis=1)

proteins_PDB=list(OrderedDict.fromkeys(dataset["PDB ID"])) #list of PDB ID used

#%% accesso al database
pdbl = PDBList() 


for name_protein in proteins_PDB:
            #download pdb file  
            pdbl.retrieve_pdb_file(name_protein, pdir = '.', file_format = 'pdb')
            parser = PDBParser(PERMISSIVE = True, QUIET = True) 
            structure = parser.get_structure(path_dir+name_protein,path_dir+"pdb"+name_protein+".ent") 
            
            Cof_coord=dict()      #coordinate del baricentro dell'anello del cofattore
            Cof_coords=dict()     #coordinate di tutti gli atomi dell'anello del cofattore 
            
            for model in structure:
                #header
                header=structure.header        
                chains=model.get_chains()
                
                #scan on chains
                for chain in chains:
                    print(chain)
                               
                    residue_names=[residue.resname for residue in chain.get_residues()] #check on FMN and FAD
             
                    if "FAD" not in residue_names and "FMN" not in residue_names:
                        print("non c'è ne un FAD ne un FMN!")
                        continue  
                                    #scan on residues
                    for residue in chain.get_residues():
                        
                        if residue.resname=="FMN" or residue.resname=="FAD":
                            continue
                        if residue.resname=="FMN":
                            FAD=0
                            #FMN cofactor
                            ind1=0
                            ind2=18
                        else: 
                            FAD=1
                            # FAD cofactor
                            ind1=23
                            ind2=40
                            
                        Cof_coords_el=get_atoms_coord(residue,ind1,ind2)
                        Cof_coord_el=get_baricentro(residue,ind1,ind2)
                        
                        dict_pdb_ligand_ring_atoms=check_ligand_ring(name_protein, chain, Cof_coords_el)
                        #dict_pdb_ligand_bar=check_ligand_bar(name_protein, chain, Cof_coord_el)
                        
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
            