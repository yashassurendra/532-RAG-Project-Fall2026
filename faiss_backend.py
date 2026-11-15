import faiss
import numpy as np
import pandas as pd

class FaissBackend:
    def __init__(self, passages_path, metadata_path):
        print("Loading vectors and metadata into RAM...")
        self.vectors = np.load(passages_path)
        self.metadata = pd.read_parquet(metadata_path)
        
        # Build the FAISS in-memory index
        # Using IndexFlatIP (Inner Product) because the vectors are already L2 normalized
        dimension = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.vectors)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_vector, filter_dict=None, k=10):
        """
        Executes a similarity search, applying metadata filters if provided.
        """
        # Ensure the query vector is a 2D array for FAISS
        query_vector = query_vector.reshape(1, -1)

        # 1. No Filters: Run a pure vector search
        if not filter_dict:
            distances, indices = self.index.search(query_vector, k)
            return indices[0]

        # 2. Filtered Search: Application-managed metadata filtering
        valid_rows = self.metadata.copy()
        for key, value in filter_dict.items():
            valid_rows = valid_rows[valid_rows[key] == value]
        
        valid_indices = valid_rows["row_idx"].values
        
        # For this prototype, we build a temporary index of just the valid vectors.
        # This simulates the overhead of application-side filtering evaluated in the benchmark.
        filtered_vectors = self.vectors[valid_indices]
        temp_index = faiss.IndexFlatIP(self.vectors.shape[1])
        temp_index.add(filtered_vectors)
        
        distances, temp_indices = temp_index.search(query_vector, k)
        
        # Map the results back to their original row IDs
        if len(temp_indices[0]) > 0 and temp_indices[0][0] != -1:
            original_indices = valid_indices[temp_indices[0]]
            return original_indices
        
        return np.array([])