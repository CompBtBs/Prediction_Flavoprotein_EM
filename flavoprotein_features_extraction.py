# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:51:56 2022

@author: bruno
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 00:12:22 2021

@author: Viego

Codice per generare specifiche features per flavoproteine

"""

#%% librerie Python
from collections import OrderedDict
from Bio.PDB import PDBList,PDBParser
from PyBioMed.PyProtein import CTD
import numpy as np
import pandas as pd
from utils import get_baricentro,get_atoms_coord,get_covariance,inizializza_dict_amm,feature_conteggio,specific_feature
#%% parametri di lancio

list_min_distBaricenter=list(np.arange(8,17))     #distanza rispetto al baricentro dell'anello isocoso
list_min_distRing=list(np.arange(3,7))            #distanza rispetto ad N5 dell'anello isocoso
  
nomi_amm=["ALA","ARG","ASN","ASP","CYS",
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
#path_dir1="G:/Altri computer/Computer_Laboratorio/"
path_dir1="C:/Users/AM866527/Desktop/"
dataset=pd.read_excel(path_dir1+"AntonioM/Features_extraction/PDB_LIST_PH.xlsx",
                      usecols=(1,2))
dataset_pH=pd.read_excel(path_dir1+"AntonioM/Features_extraction/PDB_LIST_PH.xlsx",
                      usecols=(1,3))
dataset_6_variables=pd.read_excel(path_dir1+"AntonioM/Dataset/dataset_completo.xlsx", 
                      usecols=(1,4,5,6,7,8))
dataset_MAESTRO=pd.read_excel(path_dir1+"AntonioM/Features_extraction/Features_maestro_flavoproteine.xlsx")

dataset_6_variables=dataset_6_variables.groupby("PDB ID").agg(lambda x: np.round(np.mean(x)))
dataset_pH=dataset_pH.fillna(float(np.mean(dataset_pH)))
dataset_MAESTRO = dataset_MAESTRO.set_index("Title")
dataset = dataset.set_index("PDB ID")
dataset_pH = dataset_pH.set_index("PDB ID")

nomi_proteins=list(dataset.index)#export_PDB_list(dataset)
nomi_proteins=list(OrderedDict.fromkeys(nomi_proteins)) #rimuove duplicati dalla lista
#nomi_proteins=["1AG9","1AHN"] #quali proteine considero
#%% tabella proprietà amminoacidi
tabella=pd.read_csv(path_dir1+"AntonioM/Features_extraction/tabellaAmm.txt",sep="\t",index_col=1)#,header=None)#.reset_index()
tabella.index=[el.upper() for el in tabella.index]
tabella=tabella.iloc[:,1:]

#%%ciclo for per considerare i diversi raggi rispetto a baricentro ed N5
for i in list_min_distBaricenter: 
    min_distBaricenter=i
    for n in list_min_distRing:
        min_distRing=n
        if i>n: 
#%% inizializzo il dataframe ed i nomi delle colonne: nome proteina+nome catena
            df_total=pd.DataFrame()  
            nomi=list()
#%% accesso al database
            pdbl = PDBList() 

#%%ciclo for sulle varie proteine

            for name_protein in nomi_proteins: #[0:10]#["1GER"]:#nomi_proteins[0:10]:
                
                #queste 3 righe servono per interrogare il codice e tirare giù il file e memorizzarlo
                pdbl.retrieve_pdb_file(name_protein, pdir = '.', file_format = 'pdb')
                parser = PDBParser(PERMISSIVE = True, QUIET = True) 
                structure = parser.get_structure(path_dir+name_protein,path_dir+"pdb"+name_protein+".ent") 
                
                # inizializzo cose che mi servono
                dict_residues=dict()  #dizionario dei residui
                Cof_coord=dict()      #coordinate del baricentro dell'anello del cofattore
                Cof_coords=dict()     #coordinate di tutti gli atomi dell'anello del cofattore   
                N1_coord=dict()       #coordinate di N1 del cofattore
                N3_coord=dict()       #coordinate di N3 del cofattore
                N5_coord=dict()       #coordinate di N5 del cofattore
                
                #% qui faccio un ciclo for        
            
                for model in structure:
                    #header
                    header=structure.header        
                    chains=model.get_chains()
                    
                    #scorro per catene
                    for chain in chains:
                        print(chain)
            
                        #tutti gli atomi 
                        #atoms=[atom for atom in chain.get_atoms() ]
                       # print(list(set([atom.id for atom in chain.get_atoms()])))
                        #controllo se c'èfad o fmn
                        nomi_residui=[residue.resname for residue in chain.get_residues()]
            
                            
                        if "FAD" not in nomi_residui and "FMN" not in nomi_residui:
                            print("non c'è ne un FAD ne un FMN!")
                            continue
                        else:
                            nomi.append(name_protein+"chain_"+chain.id)
            
            
                        dict_residues[chain.id]=dict()
                        #scorro tutti i residui
                        for residue in chain.get_residues():
                            if residue.resname in nomi_amm:  
                                #dizionario con chiavi gli id e valori i nomi degli amminoacidi
                                dict_residues[chain.id][residue.id[1]]=residue.resname  
                            elif residue.resname=="FMN" or residue.resname=="FAD":
                                #salvo le info sul cofattore (si ipotizza che ci sia un solo cofattore per catena)
                                for el in residue.get_atoms():
                                    if el.id=="N1":
                                        N1_el=el.coord
                                    elif el.id=="N3":
                                        N3_el=el.coord
                                    elif el.id=="N5":
                                        N5_el=el.coord
                                               
                                if residue.resname=="FMN":
                                    FAD=0
                                    #cofattore FMN
                                    ind1=0
                                    ind2=18
                                else: 
                                    FAD=1
                                    #cofattore FAD
                                    ind1=23
                                    ind2=40
                
                                #calcolo il baricentro
                                Cof_coord_el=get_baricentro(residue,ind1,ind2)
                                #calcolo gli atomi dell'anello
                                Cof_coords_el=get_atoms_coord(residue,ind1,ind2) 
                                #calcolo gli autovalori del whim
                                eigS=get_covariance(residue,ind1,ind2)
                                
            
                        #calcolo le feature sul conteggio
                        dict_cont=inizializza_dict_amm(nomi_amm)
            
                        dict_cont,N5_nearest_res,N5_3_nearest_res=feature_conteggio(dict_cont,
                                  chain,
                                  Cof_coord_el,
                                  Cof_coords_el,
                                  N5_el,
                                  min_distBaricenter,
                                  min_distRing,
                                  dict_residues,
                                  nomi_amm)
                      
                        #calcolo gli amminoacidi totali
                        total=sum([dict_cont[nome] for nome in nomi_amm])
                        total_protein=sum([dict_cont["Protein."+nome] for nome in nomi_amm])
                        total_NNB=sum([dict_cont["NNB."+nome] for nome in nomi_amm])
            
                        #queste righe sono solo per non dividere per zero dopo
                        if total==0:
                            total=1   
                        if total_protein==0:
                            total_protein=1 
                        if total_NNB==0:
                            total_NNB=1 
                        
                        #calcolo le percentuali
                        #for nome in nomi_amm: #20+20
                            #dict_cont[nome+"%"]=dict_cont[nome]/total*100
                            #dict_cont["Protein."+nome+"%"]=dict_cont["Protein."+nome]/total_protein*100
                            #dict_cont["NNB."+nome+"%"]=dict_cont["Protein."+nome]/total_NNB*100
                        
                        #calcolo i descrittori Pone per l'intorno dell'anello (rispetto al baricentro)
                        for col in tabella.columns: #28+28
                            values=tabella[col]
                            val_feature=np.sum([values[nome]*dict_cont[nome] for nome in nomi_amm])
                            dict_cont[col]=val_feature
                            #dict_cont[col+"_mean"]=val_feature/total
                
                        #calcolo i descrittori per tutta la proteina
                        for col in tabella.columns: 
                            values=tabella[col]
                            val_feature=sum([values[nome]*dict_cont["Protein."+nome] for nome in nomi_amm])
                            dict_cont["Protein."+col]=val_feature
                            #dict_cont["Protein."+col+"_mean"]=val_feature/total_protein        
                        
                        #calcolo i descrittori Pone per l'intorno dell'anello ( rispetto a un atomo qualunque)
                        for col in tabella.columns: 
                            values=tabella[col]
                            val_feature=np.sum([values[nome]*dict_cont["NNB."+nome] for nome in nomi_amm])
                            dict_cont["NNB."+col]=val_feature
                            #dict_cont["NNB."+col+"_mean"]=val_feature/total_NNB           
                            
                        #calcolo descrittori dell'amminoacido davanti a N5
                        if N5_nearest_res:
                            for col in tabella.columns: #28
                                value=tabella.loc[N5_nearest_res,col]
                                dict_cont["N5."+col]=value
                        
                        #calcolo descrittori dei 3 amminoacidi più vicini a N5
                        if N5_3_nearest_res:
                            for col in tabella.columns: #28
                                value=tabella.loc[N5_3_nearest_res,col].sum()
                                dict_cont["N3_amm."+col]=value        
                        
                        #altre feature  #alifatici, aromatici, etc...
                        dict_cont=specific_feature(dict_cont,prefisso="",mean=True,total=total)
                        dict_cont=specific_feature(dict_cont,prefisso="Protein.",mean=True,total=total_protein)
                        dict_cont=specific_feature(dict_cont,prefisso="NNB.",mean=True,total=total_NNB)
                
                        #calcolo le feature sul cofattore
                        dict_cont["FAD"]=FAD             #c'è il FAD?
                        dict_cont["WhimX_cof"]=eigS[0]   
                        dict_cont["WhimY_cof"]=eigS[1]
                        dict_cont["WhimZ_cof"]=eigS[2]        
                        
                        dict_cont["Protein_name"]=name_protein
                        
                        #aggiungo qualche info di header
                        #dict_cont["organism"]=header["source"]["1"]["organism_scientific"]
                        
                        #features da PyBioMed
                        lista_fasta=""
                        for residue_name in nomi_residui:
                            if residue_name in d3to1.keys():
                                lista_fasta=lista_fasta+d3to1[residue_name]
                        protein_descriptor = CTD.CalculateC(lista_fasta)
                        
                        #esegue il merge di due dizionari che in questo caso sono quello relativo alle features da codice e le features di PyBioMed
                        #dict_cont=dict_cont.update(AAC)
                        #dict_cont=dict_cont.update(protein_descriptor)
                        dict_cont={**dict_cont,**protein_descriptor}
                        
                        #aggiunge features MAESTRO
                        dict_cont["Exper_Temperature"]=float(dataset_MAESTRO["PDB EXPDTA TEMPERATURE"].loc[name_protein])
                        dict_cont["Exper_pH"]=float(dataset_MAESTRO["PDB EXPDTA PH"].loc[name_protein])
                        dict_cont["Molecular charge"]=float(dataset_MAESTRO["Molecular charge"].loc[name_protein])
                        dict_cont["Spin multiplicity"]=float(dataset_MAESTRO["Spin multiplicity"].loc[name_protein])
                        dict_cont["AlogP"]=float(dataset_MAESTRO["AlogP"].loc[name_protein])
                        dict_cont["Polar SA"]=float(dataset_MAESTRO["Polar SA"].loc[name_protein])
                        dict_cont["Polarizability"]=float(dataset_MAESTRO["Polarizability"].loc[name_protein])
                        
                        #aggiunge il potenziale medio  
                        dict_cont["EM"]=float(dataset.loc[name_protein].values[0])
                        
                        #aggiunge il pH
                        dict_cont["pH"]=float(dataset_pH.loc[name_protein].values[0])
                        
                        dict_cont["N1_N3"] = float(dataset_6_variables["Pymol_N1-N3"].loc[name_protein])
                        dict_cont["Oxygen_H_bond"] = float(dataset_6_variables["Pymol_O"].loc[name_protein])
                        dict_cont["Pi-Pi_Stacking"] = float(dataset_6_variables["Pymol_Pi-Pistacking"].loc[name_protein])
                        dict_cont["Stacking_Alifatico"] = float(dataset_6_variables["Pymol_Stackingalifatico"].loc[name_protein])
                        dict_cont["Pi_cation"] = float(dataset_6_variables["Pymol_Pi Cation"].loc[name_protein])
                        
            
            
            
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
        print(min_distBaricenter,min_distRing)
        #%% colonne da tenere
        #columns_to_keep=[el for el in df_total.columns if "%" not in el]
        #columns_to_keep2=[el for el in df_total.columns if "%" not in el and el!='Protein_name']
    
    
        #salvo il database nuovo
        df_total.to_excel(path_dir1+"AntonioM/Features_extraction/Dataset_ottenuti_con_codice_python_dropmeanandperc/database_dropped_chains_"+str(i)+"_"+str(n)+".xlsx")
        #df_total.loc[:,columns_to_keep].to_excel(path_dir1+"AntonioM/Features_extraction/Dataset_ottenuti_con_codice_python/database_chains_droppate_"+str(i)+"_"+str(n)+".xlsx")
        
        #raggruppo in funzione della proteina tutte le catene
        df_total=df_total.groupby("Protein_name").agg(lambda x: np.round(np.mean(x)))
        
        #salvo il database nuovo
        df_total.to_excel(path_dir1+"AntonioM/Features_extraction/Dataset_ottenuti_con_codice_python_dropmeanandperc/database_dropped_proteins_"+str(i)+"_"+str(n)+".xlsx")
        #df_total.loc[:,columns_to_keep2].to_excel(path_dir1+"AntonioM/Features_extraction/Dataset_ottenuti_con_codice_python/database_proteins_droppate_"+str(i)+"_"+str(n)+".xlsx")
