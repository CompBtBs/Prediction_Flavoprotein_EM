# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:23:22 2021

@author: Tyrion

Funzioni utilizzate nel main

"""


from sklearn.pipeline import Pipeline, TransformerMixin
import numpy as np
from numpy.linalg import norm as norm2
from sklearn.ensemble import IsolationForest

def feature_selected():
    return [
     "EM",
     'pH',
  'Oxygen_H_bond',
  'Pi-Pi_Stacking',
  'Stacking_Alifatico',
  'Pi_cation',
  'NNB.ALA',
  'ASP',
  'CYS',
  'NNB.GLU',
  'HIS',
  'NNB.HIS',
  'Nitrogen_around',
  'NNB.nNats in side chain',
  'N5.Isoelectric point',
  'N5.log(Solub) x Flex',
  'N3_amm.Hydrophobicity x Flex.',
  'RESNEG',
  '_SolventAccessibilityC2',
  '_HydrophobicityC1',
  '_HydrophobicityC3'        
        ]

def RemoveOutliar(X,y):
    clf = IsolationForest(random_state=0).fit(X)
    X_=X.copy()
    y_=y.copy()
    X_=X_[clf.predict(X)==1,:] 
    y_=y_[clf.predict(X)==1]
    return X_,y_

class RHCF():
    def __init__(self,covariation=0.99):
        self.covariation=0.99
        pass


    def fit(self, X,y=None):
        Df_corr=np.abs(np.corrcoef(np.transpose(X)))
        upper_tri = np.triu(Df_corr,k=1)
        to_drop= [ i for i in range(X.shape[1]) if any(upper_tri[:,i] >= self.covariation)]
        self.to_keep=[i for i in range(X.shape[1]) if i not in to_drop]
        return self

    def transform(self, X,y=None):
        X_=X.copy()
        return X_[:,self.to_keep]


#%% funzione per prelevare il baricentro di una struttura
def get_baricentro(element,el1,el2):
    if el2!=-1:
        atoms_coord=[el.coord for el in element.child_list[el1:el2]]
    else:
        atoms_coord=[el.coord for el in element.child_list]

    return np.mean(atoms_coord,axis=0)

#%% funzione per prelevare tutti gli atomi di una molecola
def get_atoms_coord(element,el1,el2):
    if el2!=-1:
        atoms_coord=[el.coord for el in element.child_list[el1:el2]]
    else:
        atoms_coord=[el.coord for el in element.child_list]
    return atoms_coord

#%%
def get_covariance(element,el1,el2):

    atoms_coord_tot=[el.coord for el in element.child_list]  #coordinate dei vari atomi
    baricenter_tot=np.mean(atoms_coord_tot,axis=0)           #baricentro del cofattore
    
    weight_total=0
    Sxx=0
    Syy=0
    Szz=0
    Sxy=0
    Sxz=0
    Syz=0
    
    for el in  element.child_list[el1:el2]:
        if el.id.startswith("N"):
            weight_total+=14.01
            weight=14.01
        elif el.id.startswith("C"):
            weight_total+=12.01
            weight=12.01
        else:
            weight_total+=16
            weight=16
            
        Sxx+=weight*(el.coord[0]-baricenter_tot[0])**2
        Syy+=weight*(el.coord[1]-baricenter_tot[1])**2
        Szz+=weight*(el.coord[2]-baricenter_tot[2])**2
        Sxy+=weight*(el.coord[0]-baricenter_tot[0])*(el.coord[1]-baricenter_tot[1])
        Sxz+=weight*(el.coord[0]-baricenter_tot[0])*(el.coord[2]-baricenter_tot[2])
        Syz+=weight*(el.coord[1]-baricenter_tot[1])*(el.coord[2]-baricenter_tot[2])
        

    S=[[Sxx, Sxy, Sxz],
       [Sxy, Syy,Syz],
       [Sxz, Syz, Szz]]
    
    S=np.array(S)/weight_total
    eigS,matrix=np.linalg.eig(S)

    return eigS
#%% inizializzo il dizionario dove salvo i conteggi degli amminoacidi
def inizializza_dict_amm(nomi_amm):
    dict_cont=dict()
    for el in nomi_amm:
        dict_cont[el]=0
        dict_cont["Protein."+el]=0
        dict_cont["NNB."+el]=0
    return dict_cont

#%%
def specific_feature(dict_cont,prefisso="",mean=True,total=None):
    """
    Funzione per il calcolo di specifiche feature:

    Gli input sono:
        dict_cont: il dizionario dove salvo le feature
        prefisso: prefisso da aggiungere al nome della feature (molto comodo)
        mean: se mean è vero calcolo anche le feature in funzione di...
    """

    apolari=["ALA","PHE","GLY","ILE","LEU","MET","PRO","TRP","VAL"]
    polari=["ASP","GLU","SER","THR","CYS","TYR"]
    aromatici=["TYR","TRP","PHE"]
    
    if prefisso!="":
        apolari=[prefisso+el for el in apolari]
        polari=[prefisso+el for el in polari]
        aromatici=[prefisso+el for el in aromatici]
        
    #altre feature
    dict_cont[prefisso+"NumAMM"]=total
    dict_cont[prefisso+"RESNEG"]=dict_cont[prefisso+"GLU"]+dict_cont[prefisso+"ASP"]
    dict_cont[prefisso+"RESPOS"]=dict_cont[prefisso+"ARG"]+dict_cont[prefisso+"LYS"]
    dict_cont[prefisso+"FormalCharge"]=dict_cont[prefisso+"RESPOS"]-dict_cont[prefisso+"RESNEG"]

    ###
    dict_cont[prefisso+"ResApolari"]=np.sum([dict_cont[el] for el in apolari])
    dict_cont[prefisso+"ResPolari"]=np.sum([dict_cont[el] for el in polari])
    dict_cont[prefisso+"ResAromatici"]=np.sum([dict_cont[el] for el in aromatici])
    
    
    return dict_cont
#%% definisco alcune funzione utili
def feature_conteggio(dict_cont,
                      chain,
                      Cof_coord_el,
                      Cof_coords_el,
                      N5_el,
                      min_distBaricenter,
                      min_distRing,
                      dict_residues,
                      nomi_amm):
    """
    Funzione che conta gli amminoacidi:
        1. in tutta la proteina
        2. in un intorno dell'anello isocoso (rispetto al suo baricentro'
        3. in un intorno dell'anello isocoso (rispetto a qualunque suo atomo'
    Gli input sono:
        dict_cont: il dizionario dove salvo le feature
        chain: la catena che considero
        Cof_coord_el: coordinate del baricentro dell'anello isocoso
        Cof_coords_el: coordinate degli atomi dell'anello isocoso
        N5_el: coordinate di N5 dell'anello isocoso
        min_distBaricenter: distanza minima dal baricentro
        min_distRing: distanza minima da un qualunque atomo dell'anello isocoso
        dict_residues: dizionario dei residui (identificativo e nome)
        nomi_amm: lista degli amminoacidi
    """
    min_dist_amm=1000000 #valore di defau
    chain_amms=[el for el in chain.get_residues() if el.resname in nomi_amm]
    lenChain=len(chain_amms)-1
    
    #imposto le voci dei contatori di Oxigen, Nitrogen e Carbon
    dict_cont["Oxigen_around"]=0
    dict_cont["Nitrogen_around"]=0
    dict_cont["Carbon_around"]=0
    
    #inizializzo a zero o None in modo che se certi valori non vengono calcoli, cmq il codice gira
    k=0
    min_id_amms=None
    N5_nearest_res=None
    
    #%ciclo for su tutti gli amminoacidi
    for residue in chain_amms:
        name_residue=residue.resname
        atoms=[atom for atom in residue.get_atoms()]

        #faccio il conteggio dei vari amminoacidi della proteina
        dict_cont["Protein."+name_residue]+=1

        #se l'atomo di un amminoacido è vicino al baricentro allora lo conteggio
        for atom in atoms:
            if norm2(atom.coord-Cof_coord_el)<min_distBaricenter:
                dict_cont[name_residue]+=1 #basta solo che un atomo dell'amminoacido sia vicino!!!!
                break

        #### verifico se un amminoacidi ha almeno un atomo vicino a un qualunque atomo dell'anello
        stop_cont=0
        for atom in atoms:
            for el2 in Cof_coords_el:
                if norm2(atom.coord-el2)<min_distRing:
                    dict_cont["NNB."+name_residue]+=1
                    stop_cont=1
                    break                    
            if stop_cont:
                break


        for atom in atoms:
            for el2 in Cof_coords_el:
                if norm2(atom.coord-el2)<min_distRing:
                    #controllo se l'atomo è vicino a qualcosa e di che tipo di atomo si tratta 
                    if atom.element=="O":
                        dict_cont["Oxigen_around"]+=1                
                    if atom.element=="N":
                        dict_cont["Nitrogen_around"]+=1
                    if atom.element=="C":
                        dict_cont["Carbon_around"]+=1
                    break
                        
        #trovo l'amminoacido più vicino a N5
        for atom in atoms:
            distanza=norm2(atom.coord-N5_el)
            if distanza<min_dist_amm:
                N5_nearest_res=name_residue
                min_dist_amm=distanza
                k_id=k
                
        k=k+1

    if min_dist_amm!=1000000:
        print(min_dist_amm)
        if k_id==0:
            min_id_amms=[chain_amms[k_id+i].id[1] for i in range(0,2)]
        elif lenChain==k_id:
            min_id_amms=[chain_amms[k_id+i].id[1] for i in range(-1,1)]
        else:
            min_id_amms=[chain_amms[k_id+i].id[1] for i in range(-1,2)]
        

    if min_dist_amm!=1000000:
        N5_3_nearest_res=[dict_residues[chain.id][el] for el in min_id_amms]  
    else:
        N5_3_nearest_res=None
         
    #stampo quali sono gli amminoacidi davanti a N5
    print("Name of 3 amminoacids nearest to N5: ",N5_3_nearest_res)

    return dict_cont,N5_nearest_res,N5_3_nearest_res
