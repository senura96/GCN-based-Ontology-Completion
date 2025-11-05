#!/usr/bin/env python3
"""
Debug JSON Reader - Simple diagnostic tool
"""

import json
import os
import sys

def debug_json_file(file_path: str):
    """
    Debug a specific JSON file to identify issues
    """
    print(f"🔍 Debugging JSON file: {file_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📊 File size: {file_size} bytes")
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        print(f"❌ File is not readable: {file_path}")
        return False
    
    # Try to read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            print(f"📈 Total lines: {len(lines)}")
            
            # Check first few lines
            print(f"\n📝 First 3 lines:")
            for i, line in enumerate(lines[:3]):
                print(f"   Line {i+1}: {line.strip()[:100]}...")
            
            # Try to parse each line
            valid_json_count = 0
            error_lines = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json.loads(line)
                        valid_json_count += 1
                    except json.JSONDecodeError as e:
                        error_lines.append((line_num, str(e)))
                        if len(error_lines) <= 5:  # Show first 5 errors
                            print(f"⚠️  Line {line_num}: {e}")
            
            print(f"\n📊 Results:")
            print(f"   Valid JSON lines: {valid_json_count}")
            print(f"   Error lines: {len(error_lines)}")
            
            if error_lines:
                print(f"   First error on line: {error_lines[0][0]}")
                print(f"   Error message: {error_lines[0][1]}")
            
            return len(error_lines) == 0
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """
    Main debug function
    """
    print("🐛 JSON Reader Debug Tool")
    print("=" * 60)
    
    # Test the transport dataset files
    base_path = "dataset/contor_datasets/transport/"
    files_to_test = ['train.json', 'dev.json', 'test.json']
    
    all_good = True
    
    for file_name in files_to_test:
        file_path = os.path.join(base_path, file_name)
        print(f"\n{'='*60}")
        result = debug_json_file(file_path)
        if not result:
            all_good = False
        print()
    
    if all_good:
        print("✅ All JSON files are valid!")
    else:
        print("❌ Some JSON files have issues!")
    
    return all_good

if __name__ == "__main__":
    main()
