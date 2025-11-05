# Importing Required Libraries

# Essential imports for JSON reading
import json
import os
import sys
from typing import List, Dict, Any


#Preprocess CONTOR dataset
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.decomposition import PCA
import os
import csv

print("✅ Essential imports loaded successfully!")

from typing import List, Dict, Any, Optional
import logging

# Importing Required Libraries

# Fix numpy compatibility issue with gensim
import warnings
warnings.filterwarnings('ignore', message='.*numpy.dtype size changed.*')
warnings.filterwarnings('ignore', message='.*binary incompatibility.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
import os
import csv
import re
from dotenv import load_dotenv
import json 

# Fix VoyageAI API Key Loading
import voyageai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('/Users/senurafernando/Documents/GitHub/GCN-based-Ontology-Completion-Fork/code/conf/project_details.env')

# Get API key from environment
voyage_api_key = os.getenv('VOYAGE_API_KEY')
print(f"API Key loaded: {voyage_api_key[:10]}..." if voyage_api_key else "No API key found!")




# Try to import with error handling
try:
    from contor_utils_5 import _read_dictionary
    print("✅ Utils import successful!")
except ImportError as e:
    print(f"❌ Utils import error: {e}")




# Handle gensim import separately to avoid numpy compatibility issues
try:
    # Try to suppress the specific numpy compatibility warning
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
    print("✅ Gensim import successful!")
except Exception as e:
    print(f"❌ Gensim import error: {e}")
    print("Word embeddings will use fallback mode (zero embeddings).")
    GENSIM_AVAILABLE = False


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def import_contor_dataset(path: str, file_name: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced function to import the CONTOR dataset from a given path and file name.
    
    Args:
        path: The path to the dataset directory
        file_name: Name of the JSON file (e.g., 'train.json', 'dev.json', 'test.json')
        verbose: Whether to print detailed information
    
    Returns:
        List of dictionaries containing the dataset
    """
    file_path = os.path.join(path, file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        error_msg = f"File not readable: {file_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg)
    
    if verbose:
        print(f"📖 Reading CONTOR dataset from: {file_path}")
        print(f"📊 File size: {os.path.getsize(file_path)} bytes")
    
    data = []
    line_count = 0
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                line_count += 1
                
                if not line:  # Skip empty lines
                    continue
                
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    error_count += 1
                    error_msg = f"Invalid JSON on line {line_num}: {e}"
                    logger.warning(error_msg)
                    if verbose:
                        print(f"⚠️  {error_msg}")
                        print(f"   Problematic line: {line[:100]}...")
                    continue
                except Exception as e:
                    error_count += 1
                    error_msg = f"Unexpected error on line {line_num}: {e}"
                    logger.error(error_msg)
                    if verbose:
                        print(f"❌ {error_msg}")
                    continue
        
        if verbose:
            print(f"✅ Successfully loaded {len(data)} samples from {file_name}")
            print(f"📈 Total lines processed: {line_count}")
            if error_count > 0:
                print(f"⚠️  Errors encountered: {error_count}")
        
        return data
        
    except UnicodeDecodeError as e:
        error_msg = f"Encoding error reading file {file_path}: {e}"
        logger.error(error_msg)
        if verbose:
            print(f"❌ {error_msg}")
        return []
    except Exception as e:
        error_msg = f"Unexpected error reading file {file_path}: {e}"
        logger.error(error_msg)
        if verbose:
            print(f"❌ {error_msg}")
        return []

def load_contor_transport_dataset(base_path: str = "dataset/contor_datasets/transport/", verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Enhanced function to load the complete Transport dataset (train, dev, test)
    
    Args:
        base_path: Base path to the transport dataset
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary containing train, dev, and test data
    """
    if verbose:
        print("🚀 Loading CONTOR Transport Dataset")
        print("=" * 50)
    
    dataset = {}
    total_samples = 0
    
    # Define the splits to load
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        if verbose:
            print(f"\n📂 Loading {split} data...")
        
        try:
            dataset[split] = import_contor_dataset(base_path, f'{split}.json', verbose=verbose)
            total_samples += len(dataset[split])
            
            if verbose:
                print(f"✅ {split.capitalize()} data: {len(dataset[split])} samples")
                
        except FileNotFoundError as e:
            if verbose:
                print(f"❌ Error loading {split} data: {e}")
            dataset[split] = []
        except Exception as e:
            if verbose:
                print(f"❌ Unexpected error loading {split} data: {e}")
            dataset[split] = []
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"📊 Dataset Summary:")
        print(f"   Total samples: {total_samples}")
        for split in splits:
            print(f"   {split.capitalize()}: {len(dataset[split])} samples")
    
    return dataset

# Updated analyze_dataset_structure function that returns concepts and labels
def analyze_dataset_structure(dataset: Dict[str, List[Dict[str, Any]]], verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Analyze the structure of the loaded dataset and return concepts and labels

    Args:
        dataset: The loaded dataset dictionary
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary containing sub-concepts, super-concepts, and labels for each split
    """
    if verbose:
        print("\n🔍 Dataset Structure Analysis")
        print("=" * 50)

    results = {}

    for split_name, split_data in dataset.items():
        if not split_data:
            if verbose:
                print(f"\n📂 {split_name.upper()}: No data")
            results[split_name] = {
                'sub_concepts': set(),
                'super_concepts': set(),
                'labels': set(),
                'count': 0
            }
            continue
            
        if verbose:
            print(f"\n📂 {split_name.upper()}: {len(split_data)} samples")
        
        # Analyze first sample
        sample = split_data[0]
        if verbose:
            print(f"   Sample structure:")
            for key, value in sample.items():
                print(f"     {key}: {type(value).__name__} = {value}")
        
        # Extract concepts and labels
        sub_concepts = set(item.get('v_sub_concept', '') for item in split_data)
        super_concepts = set(item.get('v_super_concept', '') for item in split_data)
        labels = set(item.get('label', '') for item in split_data)
        
        # Store results
        results[split_name] = {
            'sub_concepts': sub_concepts,
            'super_concepts': super_concepts,
            'labels': labels,
            'count': len(split_data)
        }
        
        if verbose:
            print(f"   Unique sub-concepts: {len(sub_concepts)}")
            print(f"   Unique super-concepts: {len(super_concepts)}")
            print(f"   Unique labels: {labels}")

    return results








base_path = "/Users/senurafernando/Documents/GitHub/GCN-based-Ontology-Completion-Fork/dataset/contor_datasets/transport"


import_contor_dataset("/Users/senurafernando/Documents/GitHub/GCN-based-Ontology-Completion-Fork/dataset/contor_datasets/transport" , "train.json")



load_contor_transport_dataset(base_path = "/Users/senurafernando/Documents/GitHub/GCN-based-Ontology-Completion-Fork/dataset/contor_datasets/transport")




"""
Preprocess CONTOR dataset with all functionalities from load_whole_data.

"""



def preprocess_contor_data_new(json_file_path, 
                          ftype='voyage', 
                          dim=1024,
                          voyage_embeddings=None,
                          voyage_client=None,
                          voyage_model='voyage-3-large',
                          voyage_input_type='document',
                          bidirectional=False,
                          cache_dir=None,
                          return_dict=True,
                          verbose=True):
    """
    Enhanced preprocess CONTOR train.json with Voyage embeddings as default.
    All functionalities from load_whole_data, optimized for Voyage AI embeddings.
    
    Args:
        json_file_path: Path to CONTOR JSON file (one JSON object per line)
        ftype: Feature type - 'voyage' (default), 'analogy', 'embedding', or 'none'
        dim: Dimension for embeddings (default: 1024 for Voyage-3-large)
        voyage_embeddings: Pre-computed Voyage embeddings dict {concept: embedding_array}
                          If None and ftype='voyage', will generate using voyage_client
        voyage_client: VoyageAI client instance (if None, will try to create from env)
        voyage_model: Voyage model name (default: 'voyage-3-large')
        voyage_input_type: Input type for Voyage ('document' or 'query', default: 'document')
        bidirectional: If True, add reverse edges (bidirectional)
        cache_dir: Directory to cache nodes_dict, label_dict, and features files
        return_dict: If True, return dictionary; if False, return tuple like load_whole_data
        verbose: Whether to print progress information
        
    Returns:
        If return_dict=True:
            Dictionary with all components
        If return_dict=False:
            Tuple: (num_node, edge_list, edge_src, edge_dst, edge_type, edge_norm, 
                   num_rel, node_id_con, labels, node_features)
    """
    
    # 1. Load CONTOR JSON data
    if verbose:
        print(f"📖 Loading CONTOR data from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if verbose:
        print(f"✅ Loaded {len(data)} samples")
    
    # Determine cache file paths
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        node_dict_file = os.path.join(cache_dir, 'all_nodes.dict')
        label_dict_file = os.path.join(cache_dir, 'all_unary_templates.dict')
    else:
        node_dict_file = None
        label_dict_file = None
    
    # 2. Create nodes dictionary with caching
    nodes_dict = None
    if node_dict_file and os.path.exists(node_dict_file):
        if verbose:
            print(f"📦 Loading nodes_dict from cache: {node_dict_file}")
        try:
            nodes_dict = _read_dictionary(node_dict_file)
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Could not load cache: {e}")
            nodes_dict = None
    
    if nodes_dict is None:
        if verbose:
            print("📝 Creating nodes_dict from data...")
        nodes_dict = {}
        all_concepts = set()
        
        for item in data:
            sub_concept = item['v_sub_concept']
            super_concept = item['v_super_concept']
            all_concepts.add(sub_concept)
            all_concepts.add(super_concept)
        
        # Assign node IDs
        for i, concept in enumerate(sorted(all_concepts)):
            nodes_dict[concept] = i
        
        # Save to cache
        if node_dict_file:
            if verbose:
                print(f"💾 Saving nodes_dict to cache: {node_dict_file}")
            node_id_str = ''
            for nod_id, nod in enumerate(sorted(all_concepts)):
                node_id_str += str(nod_id) + '\t' + nod + '\n'
            with open(node_dict_file, 'w', encoding='utf-8') as f:
                f.write(node_id_str)
    
    num_nodes = len(nodes_dict)
    if verbose:
        print(f"📊 Number of nodes: {num_nodes}")
    
    # 3. Create label dictionary with caching
    label_dict = None
    if label_dict_file and os.path.exists(label_dict_file):
        if verbose:
            print(f"📦 Loading label_dict from cache: {label_dict_file}")
        try:
            label_dict = _read_dictionary(label_dict_file)
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Could not load cache: {e}")
            label_dict = None
    
    if label_dict is None:
        if verbose:
            print("📝 Creating label_dict from data...")
        label_dict = {}
        label_id = 0
        
        for item in data:
            rule = item['rule']
            if rule not in label_dict:
                label_dict[rule] = label_id
                label_id += 1
        
        # Save to cache
        if label_dict_file:
            if verbose:
                print(f"💾 Saving label_dict to cache: {label_dict_file}")
            label_id_str = ''
            for lab_id, lab in enumerate(sorted(label_dict.keys(), key=lambda x: label_dict[x])):
                label_id_str += str(lab_id) + '\t' + lab + '\n'
            with open(label_dict_file, 'w', encoding='utf-8') as f:
                f.write(label_id_str)
    
    num_labels = len(label_dict)
    if verbose:
        print(f"📊 Number of labels: {num_labels}")
    
    # 4. Create label matrix
    if verbose:
        print("📝 Creating label matrix...")
    labels = sp.lil_matrix((num_nodes, num_labels))
    
    for item in data:
        sub_concept = item['v_sub_concept']
        rule = item['rule']
        label = item['label']
        
        if sub_concept in nodes_dict and rule in label_dict:
            node_id = nodes_dict[sub_concept]
            label_id = label_dict[rule]
            labels[node_id, label_id] = label
    
    labels = labels.tocsr()
    
    # 5. Create edge list (concept relationships)
    if verbose:
        print("📝 Creating edge list...")
    edge_list = []
    relation_dict = {}
    rid = 1  # 0 for self-relation
    node_id_con = set()  # nodes connected with others
    
    for item in data:
        sub_concept = item['v_sub_concept']
        super_concept = item['v_super_concept']
        label = item['label']
        
        if sub_concept in nodes_dict and super_concept in nodes_dict:
            # Create relationship edge
            relation_type = f"IS_A_{label}"  # Different relation types for valid/invalid
            if relation_type not in relation_dict:
                relation_dict[relation_type] = rid
                rid += 1
            
            src = nodes_dict[sub_concept]
            dst = nodes_dict[super_concept]
            node_id_con.add(src)
            node_id_con.add(dst)
            
            edge_list.append((src, dst, relation_dict[relation_type]))
            
            # Add bidirectional edge if requested
            if bidirectional:
                edge_list.append((dst, src, relation_dict[relation_type]))
    
    # 6. Add self-connections (only for connected nodes, like load_whole_data)
    for node_id in node_id_con:
        edge_list.append((node_id, node_id, 0))  # Self-relation
    
    # 7. Sort edges (like load_whole_data)
    edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
    edge_list = np.array(edge_list, dtype=np.int32)
    
    num_rel = len(relation_dict) + 1  # +1 for self-relation
    
    # 8. Compute edge normalization (like load_whole_data)
    if verbose:
        print("📝 Computing edge normalization...")
    edge_src, edge_dst, edge_type = edge_list.transpose()
    _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, 
                                        return_inverse=True, return_counts=True)
    degrees = count[inverse_index]  # c_{i,r} for each relation type
    edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)
    
    # 9. Generate node features (Voyage embeddings as default)
    node_features = None
    if ftype != 'none':
        if cache_dir:
            if ftype == 'voyage':
                feature_file = os.path.join(cache_dir, f'all_voyage_features_{dim}.csv')
            elif ftype == 'analogy':
                feature_file = os.path.join(cache_dir, f'all_an_features_{dim}.csv')
            elif ftype == 'embedding':
                feature_file = os.path.join(cache_dir, 'all_em_features.csv')
            else:
                feature_file = None
        else:
            feature_file = None
        
        if feature_file and os.path.exists(feature_file):
            if verbose:
                print(f"📦 Loading node features from cache: {feature_file}")
            try:
                node_features = pd.read_csv(feature_file, sep=',', encoding='utf-8', header=None)
                node_features = node_features.values
                if verbose:
                    print(f"✅ Loaded features with shape: {node_features.shape}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Warning: Could not load cached features: {e}")
                node_features = None
        
        if node_features is None:
            if verbose:
                print(f"📝 Generating node features (type: {ftype})...")
            
            if ftype == 'voyage':
                # Voyage embeddings (primary/default method)
                if voyage_embeddings:
                    # Use provided Voyage embeddings
                    if verbose:
                        print(f"📦 Using provided Voyage embeddings...")
                    node_features = np.zeros((num_nodes, dim), dtype=np.float32)
                    missing_count = 0
                    for concept, node_id in nodes_dict.items():
                        if concept in voyage_embeddings:
                            emb = voyage_embeddings[concept]
                            # Handle both array and list formats
                            if isinstance(emb, (list, tuple)):
                                emb = np.array(emb)
                            # Truncate or pad to desired dimension
                            if len(emb) >= dim:
                                node_features[node_id] = emb[:dim]
                            else:
                                # Pad with zeros if embedding is shorter
                                node_features[node_id, :len(emb)] = emb
                        else:
                            missing_count += 1
                            if verbose and missing_count <= 5:
                                print(f"⚠️  Warning: No Voyage embedding for concept '{concept}'")
                    if verbose and missing_count > 5:
                        print(f"⚠️  Warning: {missing_count} concepts missing Voyage embeddings")
                
                elif voyage_client:
                    # Generate Voyage embeddings on the fly
                    if verbose:
                        print(f"🚀 Generating Voyage embeddings using {voyage_model}...")
                    try:
                        import voyageai
                        concepts_list = sorted(list(nodes_dict.keys()))
                        texts_to_embed = [f"Transportation concept: {concept}" for concept in concepts_list]
                        
                        # Batch processing
                        batch_size = 128
                        all_embeddings_list = []
                        
                        for i in range(0, len(texts_to_embed), batch_size):
                            batch_texts = texts_to_embed[i:i + batch_size]
                            if verbose:
                                print(f"   Processing batch {i//batch_size + 1}/{(len(texts_to_embed)-1)//batch_size + 1}...")
                            
                            result = voyage_client.embed(
                                batch_texts,
                                model=voyage_model,
                                input_type=voyage_input_type,
                                truncation=True
                            )
                            all_embeddings_list.extend(result.embeddings)
                        
                        # Create feature matrix
                        node_features = np.zeros((num_nodes, dim), dtype=np.float32)
                        for idx, concept in enumerate(concepts_list):
                            emb = np.array(all_embeddings_list[idx])
                            if len(emb) >= dim:
                                node_features[nodes_dict[concept]] = emb[:dim]
                            else:
                                node_features[nodes_dict[concept], :len(emb)] = emb
                        
                        if verbose:
                            print(f"✅ Generated Voyage embeddings with shape: {node_features.shape}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"❌ Error generating Voyage embeddings: {e}")
                            print("⚠️  Falling back to zero embeddings")
                        node_features = np.zeros((num_nodes, dim), dtype=np.float32)
                
                else:
                    # Try to get from environment or create client
                    try:
                        import voyageai
                        voyage_api_key = os.getenv('VOYAGE_API_KEY')
                        if voyage_api_key:
                            if verbose:
                                print(f"🔑 Creating Voyage client from API key...")
                            voyage_client = voyageai.Client(api_key=voyage_api_key)
                            # Recursive call with client
                            # Extract relevant parameters and call again
                            return preprocess_contor_data_new(
                                json_file_path, ftype='voyage', dim=dim,
                                voyage_client=voyage_client, voyage_model=voyage_model,
                                voyage_input_type=voyage_input_type, bidirectional=bidirectional,
                                cache_dir=cache_dir, return_dict=return_dict, verbose=verbose
                            )
                        else:
                            if verbose:
                                print("⚠️  Warning: No Voyage API key found and no embeddings provided")
                            node_features = np.zeros((num_nodes, dim), dtype=np.float32)
                    except ImportError:
                        if verbose:
                            print("⚠️  Warning: voyageai package not available")
                        node_features = np.zeros((num_nodes, dim), dtype=np.float32)
            
            elif ftype == 'analogy':
                # Use PCA on labels
                labels_dense = labels.todense()
                pca = PCA(n_components=dim)
                node_features = pca.fit_transform(labels_dense)
                if verbose:
                    print(f"✅ Generated analogy features with shape: {node_features.shape}")
            
            elif ftype == 'embedding':
                # Fallback to Word2Vec (if needed)
                if verbose:
                    print("⚠️  Note: Using Word2Vec embeddings (consider using Voyage instead)")
                try:
                    from word_embedding_2 import word_embedding
                    embedding_file = 'dataset/GoogleNews-vectors-negative300.bin.gz'
                    node_features = word_embedding(embedding_file, nodes_dict)
                except ImportError:
                    if verbose:
                        print("⚠️  Warning: word_embedding function not available, using zero embeddings")
                    node_features = np.zeros((num_nodes, dim), dtype=np.float32)
            
            # Save features to cache
            if feature_file and node_features is not None:
                if verbose:
                    print(f"💾 Saving node features to cache: {feature_file}")
                try:
                    with open(feature_file, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        # Write without header row for consistency
                        writer.writerows(node_features)
                    if verbose:
                        print(f"✅ Saved features to {feature_file}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not save features to cache: {e}")
    
    # Convert labels to dense format (like load_whole_data)
    labels = labels.todense()
    
    if verbose:
        print("✅ Preprocessing complete!")
        print(f"   Nodes: {num_nodes}, Edges: {len(edge_list)}, Relations: {num_rel}")
        if node_features is not None:
            print(f"   Features shape: {node_features.shape}")
    
    # Return format
    if return_dict:
        return {
            'nodes_dict': nodes_dict,
            'label_dict': label_dict,
            'train_labels': labels,
            'labels': labels,
            'edge_list': edge_list,
            'relation_dict': relation_dict,
            'edge_src': edge_src,
            'edge_dst': edge_dst,
            'edge_type': edge_type,
            'edge_norm': edge_norm,
            'num_node': num_nodes,
            'num_rel': num_rel,
            'node_id_con': node_id_con,
            'node_features': node_features
        }
    else:
        # Return tuple format like load_whole_data
        return (num_nodes, edge_list, edge_src, edge_dst, edge_type, edge_norm, 
                num_rel, node_id_con, labels, node_features)