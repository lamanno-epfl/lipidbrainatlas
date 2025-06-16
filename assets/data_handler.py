# Imports
import os
import re

from collections import defaultdict
import anndata
import numpy as np
import pandas as pd

class DataHandler:
    
    def __config__(self, df):
        print("Configuring Data Processor...\n")

        self.df = df.copy()
        print(f"Data Shape: {self.df.shape}\n")

    def __init__(
        self,
        df,
        # reactions,
        # lipizones,
        initial_format="log",
        final_format="norm_exp",
    ):
    
        self.__config__(df)#, reactions, lipizones)

        self._extract_lipid_families()
        
        self.lipid_columns = []
        self.other_columns = []
        self.__fill_lipid_meta_cols()
        
        self._transform_data(initial_format, final_format)
        self.to_anndata(final_format)
        
    def _extract_lipid_families(self):
    
        self.lipid_families = set()
        pattern = r"^([A-Z][a-zA-Z]*)\s"

        for col in self.df.columns:
            match = re.match(pattern, col)
            if match:
                self.lipid_families.add(match.group(1))

        print(f"Lipid Families: {self.lipid_families}\n")

    def __fill_lipid_meta_cols(self):
    
        self.lipid_columns = [
            col for col in self.df.columns if any([l_fam in col for l_fam in self.lipid_families])
        ]
        print(f"{len(self.lipid_columns)} Lipid Expressions Columns")

        self.other_columns = [
            col
            for col in self.df.columns
            if all([l_fam not in col for l_fam in self.lipid_families])
        ]
        print(f"{len(self.other_columns)} Other Columns (Metadata)\n")

    def _transform_data(self, initial_format, final_format):
    
        if initial_format == "log" and final_format == "exp":
            self.df[self.lipid_columns] = np.exp(self.df[self.lipid_columns])

        elif initial_format == "exp" and final_format == "log":
            self.df[self.lipid_columns] = np.log(self.df[self.lipid_columns])
            
        elif initial_format == "log" and final_format == "norm_exp":
            for col in self.lipid_columns:
                self.df[col] = np.exp(self.df[col])
                self.df[col] = (self.df[col] - self.df[col].min()) / (
                    self.df[col].max() - self.df[col].min()
                )

        elif initial_format == "exp" and final_format == "norm_exp":
            for col in self.lipid_columns:
                self.df[col] = (self.df[col] - self.df[col].min()) / (
                    self.df[col].max() - self.df[col].min()
                )

        elif initial_format == final_format:
            return
        else:
            raise ValueError("initial_format and final_format must be 'log', 'exp' or 'norm_exp'")

        print(f"Data transformed from {initial_format} to {final_format}")

    def to_anndata(self, final_format="norm_exp"):
    
        assert final_format in [
            "log",
            "exp",
            "norm_exp",
        ], "final_format must be 'log', 'exp' or 'norm_exp'"

        cols = self.lipid_columns.copy()
        X_data = self.df[cols].astype("float32")

        # Select the columns for .obs attribute
        obs_data = self.df.drop(columns=cols)

        # if 'structure_id_path' in obs_data.columns and 'structure_set_ids' in obs_data.columns and 'rgb_triplet' in obs_data.columns:
        #     obs_data = obs_data.drop(columns=['structure_id_path', 'structure_set_ids', 'rgb_triplet'])

        # Create an AnnData object
        self.adata = anndata.AnnData(X=X_data, obs=obs_data)
    
    def create_reaction_matrix(self):
        if self.reactions is None:
            return
    
        # matrix from linex, not symmetric, 348 reactions in total
        
        # Excluded reactions for now:
        # LPC 20:1->PC 40:2,211,LPC 20:1,PC 40:2,(2x),
        # LPC 16:0->LPC 18:0,15,LPC 16:0,LPC 18:0,esiste sul fatty acid,
        # LPC 18:1->LPC 20:1,132,LPC 18:1,LPC 20:1,esiste sul fatty acid,ELOVL6?

        self.reactions = pd.read_csv(self.reactions, index_col=0) if isinstance(self.reactions, str) else self.reactions
        unique_reagents = self.reactions['reagent'].unique()
        unique_products = self.reactions['product'].unique()
        unique_species = sorted(set(unique_reagents).union(set(unique_products)))
        self.adata = self.adata[:, unique_species] ############################################################

        self.species_to_index = {spec: idx for idx, spec in enumerate(self.adata.var_names)}
        self.species_to_index_prova = {spec : self.adata.var_names.get_loc(spec) for spec in self.adata.var_names}

        assert self.species_to_index == self.species_to_index_prova

        self.adata.varm['metabolicmodule'] = np.zeros((self.adata.n_vars, self.adata.n_vars), dtype=int)
        for _, row in self.reactions.iterrows():
            self.adata.varm['metabolicmodule'][self.adata.var_names.get_loc(row['reagent']), self.adata.var_names.get_loc(row['product'])] = 1

    def create_lipizone_centroids(self):
    
        centroids = pd.DataFrame(self.adata.X, 
                                columns=self.adata.var_names, 
                                index=self.adata.obs_names).groupby(
                                    self.adata.obs['lipizone_names']
                                    ).mean()
        
        self.adata.uns['lipizone_names'] = centroids.index
        self.adata.varm['lipizone_centroids'] = np.array(centroids.T)

    def create_stoichiometric_matrix(self):
        if self.reactions is None:
            return
        
        reactions_idx = np.argwhere(self.adata.varm['metabolicmodule'] == 1)
        self.adata.varm['stoichiometric_matrix'] = np.zeros((len(self.adata.var_names), len(reactions_idx)), dtype=float)
        reaction_list = []
        
        # Fill the stoichiometric matrix
        for i, (reagent_idx, product_idx) in enumerate(reactions_idx):
            reagent = self.adata.var_names[reagent_idx]
            product = self.adata.var_names[product_idx]
            
            reaction_name = f"{reagent}->{product}" # k_{reagent}->{product}
            reaction_list.append(reaction_name)
            
            self.adata.varm['stoichiometric_matrix'][reagent_idx, i] = -1  # Consume reagent
            self.adata.varm['stoichiometric_matrix'][product_idx, i] = 1   # Produce product

        self.adata.uns['reactions'] = reaction_list

    def get_lipizone_names(self):
        return self.adata.uns["lipizone_names"]
    
    def get_reactions(self):
        return self.adata.uns["reactions"]

    def get_lipizone_centroids(self):
        return pd.DataFrame(
            self.adata.varm["lipizone_centroids"],
            columns=self.adata.uns["lipizone_names"],
            index=self.adata.var_names,
        )
    
    def get_stoichiometric_matrix(self):
        return pd.DataFrame(
            self.adata.varm["stoichiometric_matrix"],
            columns=self.adata.uns["reactions"],
            index=self.adata.var_names,
        )
    
    def get_metabolicmodule(self):
        return pd.DataFrame(
            self.adata.varm["metabolicmodule"],
            columns=self.adata.var_names,
            index=self.adata.var_names,
        )

    def get_data(self):
        return pd.DataFrame(
            self.adata.X,
            columns=self.adata.var_names,
            index=self.adata.obs_names,
        )