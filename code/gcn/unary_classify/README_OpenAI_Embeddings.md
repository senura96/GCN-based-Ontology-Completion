# OpenAI Embeddings for GCN-based Ontology Completion

## 📁 Files Created

1. **`openai_embedding_2.py`** - Main OpenAI embedding implementation
2. **`openai_integration_guide.py`** - Integration examples and usage guide
3. **`test_openai_embeddings.py`** - Test suite for the embedding functionality
4. **`README_OpenAI_Embeddings.md`** - This documentation

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install OpenAI package
pip install openai

# Set your API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Basic Usage
```python
from openai_embedding_2 import word_embedding_openai

# Your nodes dictionary (from your data processing)
nodes_dict = {
    "http://example.com/Transportation.owl#CargoShip": 0,
    "http://example.com/Transportation.owl#PassengerShip": 1,
    "http://example.com/Transportation.owl#RailwayVehicle": 2,
}

# Generate embeddings (drop-in replacement for Word2Vec)
embeddings = word_embedding_openai(
    nodes_dict=nodes_dict,
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    cache_file="embeddings_cache.json"
)

print(f"Generated embeddings shape: {embeddings.shape}")
# Output: (3, 1536) for text-embedding-3-small
```

## 🔧 Integration with Your Data Processing

When you create your data processing pipeline, replace the Word2Vec section:

### OLD CODE (Word2Vec):
```python
from word_embedding_2 import word_embedding

if ftype == 'embedding':
    embedding_file = 'dataset/GoogleNews-vectors-negative300.bin.gz'
    node_features = word_embedding(embedding_file, nodes_dict)
```

### NEW CODE (OpenAI):
```python
from openai_embedding_2 import word_embedding_openai

if ftype == 'embedding':
    node_features = word_embedding_openai(
        nodes_dict=nodes_dict,
        model="text-embedding-3-small",
        cache_file=f"{train_path}/openai_embeddings_cache.json"
    )
```

## 🎯 Key Features

### ✅ **Drop-in Replacement**
- Same interface as original `word_embedding` function
- No changes needed to your GCN model code
- Just replace the import and function call

### ✅ **Smart Concept Extraction**
- Automatically extracts concept names from URIs
- Converts camelCase to readable format
- Creates enhanced prompts for better embeddings

### ✅ **Caching System**
- Saves embeddings to avoid re-computing
- Significantly reduces API costs
- Automatic cache loading/saving

### ✅ **Batch Processing**
- Processes concepts in batches for efficiency
- Respects OpenAI rate limits
- Configurable batch sizes

### ✅ **Error Handling**
- Graceful fallback for API failures
- Clear error messages
- Robust caching system

## 📊 Model Options

| Model | Dimension | Cost | Quality |
|-------|-----------|------|---------|
| `text-embedding-3-small` | 1536 | Low | Good |
| `text-embedding-3-large` | 3072 | High | Excellent |
| `text-embedding-ada-002` | 1536 | Medium | Good |

## 💰 Cost Optimization

### Use Caching
```python
# Embeddings are automatically cached
# First run: API calls + caching
embeddings = word_embedding_openai(nodes_dict, cache_file="cache.json")

# Subsequent runs: Load from cache (no API calls)
embeddings = word_embedding_openai(nodes_dict, cache_file="cache.json")
```

### Batch Processing
```python
# Process in larger batches for efficiency
generator = OpenAIEmbeddingGenerator(batch_size=100)
embeddings = generator.word_embedding(nodes_dict)
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Test without API calls (mock data)
python test_openai_embeddings.py

# Test with real API calls (requires API key)
python openai_embedding_2.py
```

## 🔄 Migration Path

1. **Phase 1**: Test with sample data (current)
2. **Phase 2**: Integrate with your data processing pipeline
3. **Phase 3**: Replace Word2Vec in your GCN model
4. **Phase 4**: Optimize for production use

## 📝 Example Output

```
=== Concept Extraction ===
URI: http://example.com/Transportation.owl#CargoShip
  → Concept: Cargo Ship
  → Prompt: Transportation concept: Cargo Ship. This is a type of vehicle, infrastructure, or transportation-related entity.

=== Embedding Generation ===
Generated embeddings for 5 concepts using OpenAI text-embedding-3-small
Processing batch 1/1
Generated embeddings with shape: (5, 1536)
```

## 🆘 Troubleshooting

### Common Issues:

1. **API Key Error**: Set `OPENAI_API_KEY` environment variable
2. **Rate Limits**: Reduce batch size or add delays
3. **Cost Concerns**: Use caching and smaller models
4. **Import Errors**: Install with `pip install openai`

### Support:
- Check the integration guide: `openai_integration_guide.py`
- Run tests: `python test_openai_embeddings.py`
- Review examples in the code comments

## 🎉 Ready for Integration!

The OpenAI embedding system is ready to use. When you create your data processing pipeline, simply:

1. Import the function: `from openai_embedding_2 import word_embedding_openai`
2. Replace your Word2Vec call with the OpenAI version
3. Set your API key and run!

The system will handle concept extraction, embedding generation, and caching automatically.
