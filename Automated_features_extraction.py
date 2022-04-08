# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:56:30 2022

@author: anton
"""

#%% librerie Python
from collections import OrderedDict
from Bio.PDB import PDBList,PDBParser
from PyBioMed.PyProtein import CTD
import numpy as np
import pandas as pd
from utils import get_baricentro,get_atoms_coord,get_covariance,inizializza_dict_amm,feature_conteggio,specific_feature
#%% parametri di lancio

list_NNB=list(np.arange(11,17))          #distanza rispetto al baricentro dell'anello isocoso
list_N5=list(np.arange(3,7))            #distanza rispetto ad N5 dell'anello isocoso
  
amm_names=["ALA","ARG","ASN","ASP","CYS",
          "GLN","GLU","GLY","HIS","ILE",
          "LEU","LYS","MET","PHE","PRO",
          "SER","THR","TRP","TYR","VAL"]  #nomi amminoacidi considerati
 # You can use a dict to convert three letter code to one letter code
d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

path_dir=""
#%% leggo il file dove sono presente le proteine da considerare
dataset=pd.read_excel(path_dir+"data/final_dataset_Em_flavoproteins.xlsx",usecols=(0,3,4))
features_DS=pd.read_excel(path_dir+"data/DS_Visualizer_Features.xlsx")

proteins_PDB=list(OrderedDict.fromkeys(dataset["PDB ID"])) #list of PDB ID used

table_amm=pd.read_csv(path_dir+"data/tabellaAmm.txt",
                      sep="\t",index_col=1)#,header=None)#.reset_index()
table_amm.index=[el.upper() for el in table_amm.index]
table_amm=table_amm.iloc[:,1:]

#%%ciclo for per considerare i diversi raggi rispetto a baricentro ed N5
for NNB in list_NNB: 
    for N5 in list_N5:
#%% inizializzo il dataframe ed i nomi delle colonne: nome proteina+nome catena
        df_total=pd.DataFrame()  
        nomi=list()
#%% accesso al database
        pdbl = PDBList() 

#%%ciclo for sulle varie proteine

        for name_protein in proteins_PDB:
            
            #download pdb file  
            pdbl.retrieve_pdb_file(name_protein, pdir = '.', file_format = 'pdb')
            parser = PDBParser(PERMISSIVE = True, QUIET = True) 
            structure = parser.get_structure(path_dir+name_protein,path_dir+"pdb"+name_protein+".ent") 
            
            # generation dict 
            dict_residues=dict()  #dizionario dei residui
            Cof_coord=dict()      #coordinate del baricentro dell'anello del cofattore
            Cof_coords=dict()     #coordinate di tutti gli atomi dell'anello del cofattore   
            N1_coord=dict()       #coordinate di N1 del cofattore
            N3_coord=dict()       #coordinate di N3 del cofattore
            N5_coord=dict()       #coordinate di N5 del cofattore
            
            #% start for cicle      
        
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
                    else:
                        nomi.append(name_protein+"chain_"+chain.id)

                    dict_residues[chain.id]=dict()
                    
                    #scan on residues
                    for residue in chain.get_residues():
                        if residue.resname in amm_names:  
                            #dizionario con chiavi gli id e valori i nomi degli amminoacidi
                            dict_residues[chain.id][residue.id[1]]=residue.resname  
                        elif residue.resname=="FMN" or residue.resname=="FAD":
                            #save info about the cofactor 
                            for el in residue.get_atoms():
                                if el.id=="N1":
                                    N1_el=el.coord
                                elif el.id=="N3":
                                    N3_el=el.coord
                                elif el.id=="N5":
                                    N5_el=el.coord
                                           
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
            
                            #calculate barycenter
                            Cof_coord_el=get_baricentro(residue,ind1,ind2)
                            #calculate ring's atoms
                            Cof_coords_el=get_atoms_coord(residue,ind1,ind2) 
                            #calcolo gli autovalori del whim
                            eigS=get_covariance(residue,ind1,ind2)
                            
        
                    #calcolo le feature sul conteggio
                    dict_cont=inizializza_dict_amm(amm_names)
        
                    dict_cont,N5_nearest_res,N5_3_nearest_res=feature_conteggio(dict_cont,
                              chain,
                              Cof_coord_el,
                              Cof_coords_el,
                              N5_el,
                              NNB,
                              N5,
                              dict_residues,
                              amm_names)
                  
                    #calcolo gli amminoacidi totali
                    total=sum([dict_cont[nome] for nome in amm_names])
                    total_protein=sum([dict_cont["Protein."+nome] for nome in amm_names])
                    total_NNB=sum([dict_cont["NNB."+nome] for nome in amm_names])
        
                    #queste righe sono solo per non dividere per zero dopo
                    if total==0:
                        total=1   
                    if total_protein==0:
                        total_protein=1 
                    if total_NNB==0:
                        total_NNB=1 
                    
                    
                    #calcolo i descrittori Pone per l'intorno dell'anello (rispetto al baricentro)
                    for col in table_amm.columns: #28+28
                        values=table_amm[col]
                        val_feature=np.sum([values[nome]*dict_cont[nome] for nome in amm_names])
                        dict_cont[col]=val_feature
                        
            
                    #calcolo i descrittori per tutta la proteina
                    for col in table_amm.columns: 
                        values=table_amm[col]
                        val_feature=sum([values[nome]*dict_cont["Protein."+nome] for nome in amm_names])
                        dict_cont["Protein."+col]=val_feature
                               
                    
                    #calcolo i descrittori Pone per l'intorno dell'anello ( rispetto a un atomo qualunque)
                    for col in table_amm.columns: 
                        values=table_amm[col]
                        val_feature=np.sum([values[nome]*dict_cont["NNB."+nome] for nome in amm_names])
                        dict_cont["NNB."+col]=val_feature
                                   
                        
                    #calcolo descrittori dell'amminoacido davanti a N5
                    if N5_nearest_res:
                        for col in table_amm.columns: #28
                            value=table_amm.loc[N5_nearest_res,col]
                            dict_cont["N5."+col]=value
                    
                    #calcolo descrittori dei 3 amminoacidi più vicini a N5
                    if N5_3_nearest_res:
                        for col in table_amm.columns: #28
                            value=table_amm.loc[N5_3_nearest_res,col].sum()
                            dict_cont["N3_amm."+col]=value        
                    
                    #altre feature  #alifatici, aromatici, etc...
                    dict_cont=specific_feature(dict_cont,prefisso="",mean=True,total=total)
                    dict_cont=specific_feature(dict_cont,prefisso="Protein.",mean=True,total=total_protein)
                    dict_cont=specific_feature(dict_cont,prefisso="NNB.",mean=True,total=total_NNB)    
                    
                    dict_cont["PDB ID"]=name_protein
                    
                    #aggiungo qualche info di header
                    dict_cont["organism"]=header["source"]["1"]["organism_scientific"]
                    
                    #features da PyBioMed
                    lista_fasta=""
                    for residue_name in residue_names:
                        if residue_name in d3to1.keys():
                            lista_fasta=lista_fasta+d3to1[residue_name]
                    protein_descriptor = CTD.CalculateC(lista_fasta)
                    
                    #esegue il merge di due dizionari che in questo caso sono quello relativo alle features da codice e le features di PyBioMed
                    dict_cont={**dict_cont,**protein_descriptor}
        
        
        
                    #%%           
                    ###fine calcolo features!!!
                    df=pd.DataFrame.from_dict(dict_cont, orient ='index')
                    
                    #aggiorno il df totale!!!
                    df_total=pd.concat([df_total,df],axis=1)
        
            
            
    #%% fine ciclo for su tutte le proteine considerate
        df_total=df_total.fillna(0)
        #aggiorno i nomi delle colonne
        df_total.columns=nomi
        df_total=df_total.transpose()
        
        cols = df_total.columns.tolist()
        cols = cols[-7:-5] + cols[-5:] + cols[:-7]
        df_total = df_total[cols]
        print(NNB,N5)
        
        df_total=df_total.groupby("PDB ID").agg(lambda x: np.round(np.mean(x),2))
        
        dataset2=dataset.merge(features_DS, on="PDB ID")
        dataset2=dataset2.join(df_total, on="PDB ID")        
        dataset2.to_excel(path_dir+"dataset_features/database_protein_"+str(NNB)+"_"+str(N5)+".xlsx")    
        

