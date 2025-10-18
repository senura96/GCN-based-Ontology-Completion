import openai
import numpy as np
import re
import time
import json
import os
from typing import Dict, List, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class OpenAIEmbeddingGenerator:
    """
    OpenAI-based concept embedding generator for ontology completion.
    Replaces Word2Vec with OpenAI's text-embedding models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", 
                 batch_size: int = 100, cache_file: str = "embeddings_cache.json"):
        """
        Initialize OpenAI embedding generator.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI embedding model to use
            batch_size: Number of concepts to process in each batch
            cache_file: File to cache embeddings for reuse
        """
        self.model = model
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.embedding_dim = self._get_embedding_dimension()
        
        # Initialize OpenAI client
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Load existing cache
        self.embeddings_cache = self._load_cache()
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model."""
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return model_dims.get(self.model, 1536)
    
    def _load_cache(self) -> Dict:
        """Load embeddings cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save embeddings cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.embeddings_cache, f)
    
    def sep_by_uppercase(self, text: str) -> str:
        """
        Convert camelCase/PascalCase to space-separated words.
        
        Args:
            text: Input text (e.g., "CargoShip")
            
        Returns:
            Space-separated text (e.g., "Cargo Ship")
        """
        pattern = "[A-Z]"
        new_str = re.sub(pattern, lambda x: " " + x.group(0), text)
        if text[0].islower():
            return new_str
        else:
            return new_str[1:]
    
    def extract_concept_name(self, uri: str) -> str:
        """
        Extract concept name from URI.
        
        Args:
            uri: Full URI (e.g., "http://example.com/Transportation.owl#CargoShip")
            
        Returns:
            Clean concept name (e.g., "Cargo Ship")
        """
        # Split by '#' and take the last part
        parts = uri.strip().split('#')
        if len(parts) == 1:
            # No '#', split by '/' and take last part
            concept = parts[0].split('/')[-1]
        else:
            concept = parts[1]
        
        # Remove trailing '>' if present
        concept = concept.rstrip('>')
        
        # Convert to readable format
        concept = self.sep_by_uppercase(concept)
        
        return concept.strip()
    
    def create_embedding_prompt(self, concept: str) -> str:
        """
        Create a prompt for embedding generation that captures semantic meaning.
        
        Args:
            concept: Concept name (e.g., "Cargo Ship")
            
        Returns:
            Enhanced prompt for better embeddings
        """
        # Create a more descriptive prompt for better embeddings
        prompt = f"Transportation concept: {concept}. This is a type of vehicle, infrastructure, or transportation-related entity."
        return prompt
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text using OpenAI API.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        # Check cache first
        if text in self.embeddings_cache:
            return np.array(self.embeddings_cache[text])
        
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response['data'][0]['embedding'])
            
            # Cache the result
            self.embeddings_cache[text] = embedding.tolist()
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (None for failed ones)
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model
                )
                
                for j, data in enumerate(response['data']):
                    embedding = np.array(data['embedding'])
                    embeddings.append(embedding)
                    
                    # Cache the result
                    self.embeddings_cache[batch[j]] = embedding.tolist()
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add None for failed batch
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def word_embedding(self, nodes_dict: Dict[str, int]) -> np.ndarray:
        """
        Generate embeddings for all nodes using OpenAI.
        
        Args:
            nodes_dict: Dictionary mapping node URIs to indices
            
        Returns:
            Numpy array of embeddings (n_nodes, embedding_dim)
        """
        print(f"Generating embeddings for {len(nodes_dict)} concepts using OpenAI {self.model}")
        
        # Extract and prepare concept names
        concepts = []
        concept_to_index = {}
        
        for node_uri, index in nodes_dict.items():
            concept_name = self.extract_concept_name(node_uri)
            enhanced_prompt = self.create_embedding_prompt(concept_name)
            
            concepts.append(enhanced_prompt)
            concept_to_index[enhanced_prompt] = index
        
        # Get embeddings
        embeddings = self.get_embeddings_batch(concepts)
        
        # Create result matrix
        result_embeddings = np.zeros((len(nodes_dict), self.embedding_dim))
        
        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                result_embeddings[concept_to_index[concepts[i]]] = embedding
            else:
                print(f"Warning: Failed to get embedding for concept {i}")
        
        # Save cache
        self._save_cache()
        
        print(f"Generated embeddings with shape: {result_embeddings.shape}")
        return result_embeddings


def word_embedding_openai(nodes_dict: Dict[str, int], 
                         api_key: Optional[str] = None,
                         model: str = "text-embedding-3-small",
                         cache_file: str = "openai_embeddings_cache.json") -> np.ndarray:
    """
    Convenience function to generate OpenAI embeddings (drop-in replacement for word_embedding).
    
    Args:
        nodes_dict: Dictionary mapping node URIs to indices
        api_key: OpenAI API key (optional)
        model: OpenAI model to use
        cache_file: Cache file for embeddings
        
    Returns:
        Numpy array of embeddings
    """
    generator = OpenAIEmbeddingGenerator(api_key=api_key, model=model, cache_file=cache_file)
    return generator.word_embedding(nodes_dict)


# Example usage and testing
if __name__ == "__main__":
    # Example nodes dictionary
    sample_nodes = {
        "http://example.com/Transportation.owl#CargoShip": 0,
        "http://example.com/Transportation.owl#PassengerShip": 1,
        "http://example.com/Transportation.owl#RailwayVehicle": 2,
        "http://example.com/Transportation.owl#Airport": 3,
        "http://example.com/Transportation.owl#Bridge": 4
    }
    
    print("OpenAI Embedding Generator Demo")
    print("=" * 50)
    
    # Test concept extraction
    generator = OpenAIEmbeddingGenerator()
    
    for uri in sample_nodes.keys():
        concept = generator.extract_concept_name(uri)
        prompt = generator.create_embedding_prompt(concept)
        print(f"URI: {uri}")
        print(f"Concept: {concept}")
        print(f"Prompt: {prompt}")
        print("-" * 30)
    
    # Note: Actual embedding generation requires valid OpenAI API key
    print("\nTo use with actual embeddings:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Call: embeddings = word_embedding_openai(sample_nodes)")
