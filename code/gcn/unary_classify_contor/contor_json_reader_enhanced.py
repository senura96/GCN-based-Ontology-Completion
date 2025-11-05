#!/usr/bin/env python3
"""
Enhanced CONTOR Dataset JSON Reading Functions
Robust JSONL file reading with comprehensive error handling
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging

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

def analyze_dataset_structure(dataset: Dict[str, List[Dict[str, Any]]], verbose: bool = True) -> None:
    """
    Analyze the structure of the loaded dataset
    
    Args:
        dataset: The loaded dataset dictionary
        verbose: Whether to print detailed information
    """
    if verbose:
        print("\n🔍 Dataset Structure Analysis")
        print("=" * 50)
    
    for split_name, split_data in dataset.items():
        if not split_data:
            if verbose:
                print(f"\n📂 {split_name.upper()}: No data")
            continue
            
        if verbose:
            print(f"\n📂 {split_name.upper()}: {len(split_data)} samples")
        
        # Analyze first sample
        sample = split_data[0]
        if verbose:
            print(f"   Sample structure:")
            for key, value in sample.items():
                print(f"     {key}: {type(value).__name__} = {value}")
        
        # Count unique concepts
        if split_data:
            sub_concepts = set(item.get('v_sub_concept', '') for item in split_data)
            super_concepts = set(item.get('v_super_concept', '') for item in split_data)
            labels = set(item.get('label', '') for item in split_data)
            
            if verbose:
                print(f"   Unique sub-concepts: {len(sub_concepts)}")
                print(f"   Unique super-concepts: {len(super_concepts)}")
                print(f"   Unique labels: {labels}")

def test_json_reading():
    """
    Test function to verify JSON reading works correctly
    """
    print("🧪 Testing CONTOR JSON Reading Functions")
    print("=" * 60)
    
    try:
        # Test loading the transport dataset
        dataset = load_contor_transport_dataset(verbose=True)
        
        # Analyze the dataset structure
        analyze_dataset_structure(dataset, verbose=True)
        
        print("\n✅ JSON reading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ JSON reading test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    test_json_reading()
