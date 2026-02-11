# NSCK Natural Language Learning System - User Guide

## Overview

The NSCK Natural Language Learning System enables the cognitive architecture to learn from text files, store knowledge in its semantic and episodic memory, and answer questions based on what it learned. 

**CRITICAL: This system does NOT rely on an LLM for learning or reasoning.** The LLM (if present) is only used as a peripheral translator for natural language input/output. All learning, understanding, and reasoning happens in the NSCK cognitive architecture using:

- **Hypervectors (VSA)** for semantic representation
- **Semantic Memory** for concept graphs and relations
- **Episodic Memory** for experience storage
- **Causal Reasoning** for inference
- **Context Engine** for disambiguation

## Architecture

### Core Components

1. **TextKnowledgeLearner** (`text_knowledge_learner.py`)
   - Parses text files into sentences
   - Extracts concepts and relations using pattern matching
   - Encodes concepts into hypervectors using LinguaCortex (Semantic Folding)
   - Stores knowledge in SemanticMemory (graph + hypervector index)
   - Records experiences in EpisodicMemory
   - Builds causal relationships

2. **LinguaCortex** (Semantic Folding)
   - Converts text to Sparse Distributed Representations (SDRs)
   - Uses 2D topographical semantic maps (128x128 grid)
   - 2% sparsity for efficient computation
   - NO dense embeddings or transformer models

3. **SemanticMemory**
   - Graph-based knowledge representation
   - Concept → Hypervector mappings
   - Spreading activation for retrieval
   - Relation types: is_a, has_property, causes, part_of, similar_to

4. **EpisodicMemory**
   - Experience storage with VSA-based similarity search
   - LSH indexing for fast retrieval
   - Consolidation and compression

5. **Dashboard Integration**
   - Web-based interface at `http://localhost:5051`
   - Text Learning tab for file upload and queries
   - Real-time statistics and metrics
   - Export functionality

## How to Use

### Starting the Dashboard

```bash
cd Node_network/nsck-demo/python
python testing_dashboard.py
```

The dashboard will be available at `http://localhost:5051`

### Learning from Text Files

#### Method 1: File Upload

1. Navigate to the **Text Learning** tab
2. Click "Choose Files" and select one or more .txt files
3. Click "Upload & Learn"
4. The system will:
   - Parse sentences
   - Extract concepts (nouns, named entities)
   - Identify relations between concepts
   - Encode knowledge into hypervectors
   - Store in semantic and episodic memory

#### Method 2: Direct Text Input

1. Navigate to the **Text Learning** tab
2. Paste text into the "Or Paste Text Directly" area
3. Optionally provide a filename
4. Click "Learn from Text"

### Querying Learned Knowledge

1. In the **Text Learning** tab, find the "Query Learned Knowledge" section
2. Type your question (e.g., "What is photosynthesis?")
3. Click "Search"
4. The system will display:
   - **Confidence score** (0-100%)
   - **Relevant concepts** found via semantic search
   - **Learned facts** extracted from text
   - **Associated concepts** via spreading activation
   - **Reasoning trace** showing the cognitive process

### Using the Chat Interface

1. Navigate to the **Chat & Test** tab
2. Type your message
3. The system will automatically:
   - Query learned knowledge from text files
   - Process through the cognitive pipeline
   - Combine responses from multiple sources
   - Display confidence and reasoning trace

## Learning Process Details

### 1. Sentence Processing

```
Text → Sentences → Concepts + Relations
```

- Splits text by sentence boundaries
- Cleans and normalizes text

### 2. Concept Extraction

Extracts concepts using:
- Capitalized words (potential named entities)
- Quoted text
- Important domain terms (longer words)

Example:
```
"Photosynthesis is the process by which plants convert sunlight into energy."

Extracted concepts:
- Photosynthesis
- Process
- Plants
- Sunlight
- Energy
```

### 3. Relation Extraction

Pattern-based extraction:
- "X is a Y" → is_a relation
- "X has Y" → has_property relation
- "X causes Y" → causes relation
- "X results in Y" → results_in relation
- "X leads to Y" → leads_to relation
- "X produces Y" → produces relation
- "X contains Y" → contains relation

Example:
```
"Photosynthesis is the process by which plants convert sunlight into energy."

Extracted relations:
- Process related_to Photosynthesis
- Plants related_to Process
```

### 4. Hypervector Encoding

Each concept is encoded as a 10,240-dimensional hypervector:
- Uses hash-based initialization for determinism
- Bundle operation for combining concepts
- XOR for role-filler binding
- Similarity computed via Hamming distance

### 5. Storage

**Semantic Memory:**
- Concept graph with relations
- Hypervector index for similarity search
- Spreading activation network

**Episodic Memory:**
- Each sentence becomes an episode
- Stores context, concepts, relations
- Indexed by situation hypervector
- Enables recall of similar experiences

## Statistics and Metrics

The dashboard provides real-time metrics:

- **Sessions**: Number of learning sessions completed
- **Concepts**: Total unique concepts learned
- **Facts**: Total facts (relations) stored
- **Episodes**: Experiences recorded in episodic memory

### Learning Session Details

Each session tracks:
- Filename
- Duration
- Sentences processed
- Concepts extracted
- Relations identified
- Facts stored

### Top Concepts

Shows most frequently occurring concepts across all learned text.

## Query System

### Semantic Search

1. Query encoded to hypervector
2. Similarity search in concept space
3. Returns top-k most similar concepts

### Spreading Activation

1. Start from query concepts
2. Activate neighbors in concept graph
3. Propagate activation with decay
4. Returns highly activated concepts

### Fact Retrieval

1. Match query concepts to fact subjects/objects
2. Substring matching for flexibility
3. Return relevant facts with source text

### Confidence Calculation

```python
confidence = (avg_similarity * 0.7) + (fact_score * 0.3)
```

Where:
- avg_similarity: Average of top-3 concept similarities
- fact_score: min(num_facts * 0.1, 0.5)

## API Endpoints

### POST /api/learn/upload

Upload text files or direct text content for learning.

**Request (file upload):**
```
Content-Type: multipart/form-data
files: [file1.txt, file2.txt, ...]
```

**Request (direct text):**
```json
{
  "text": "Text content to learn...",
  "filename": "optional_name.txt"
}
```

**Response:**
```json
{
  "success": true,
  "session": {
    "session_id": "abc123",
    "filename": "test.txt",
    "duration": 0.05,
    "concepts": 25,
    "relations": 18,
    "facts": 18,
    "sentences": 10
  },
  "total_concepts": 100,
  "total_facts": 75
}
```

### GET /api/learn/stats

Get learning statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_sessions": 3,
    "total_concepts": 100,
    "total_facts": 75,
    "total_episodes": 50,
    "concept_frequencies": {...},
    "relation_types": {...},
    "sessions": [...]
  }
}
```

### POST /api/learn/query

Query learned knowledge.

**Request:**
```json
{
  "query": "What is photosynthesis?"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "answer": "...",
    "confidence": 0.85,
    "similar_concepts": [...],
    "activated_concepts": [...],
    "related_facts": [...],
    "recalled_episodes": 5,
    "reasoning_trace": [...]
  }
}
```

### GET /api/learn/export

Export all learned knowledge as formatted text file.

### POST /api/learn/reset

Reset all learned knowledge (destructive operation).

## Example Workflow

### 1. Learn from Science Text

Create `science.txt`:
```
Photosynthesis is the process by which plants convert sunlight into energy.
Plants use chlorophyll to absorb light.
Carbon dioxide and water are converted into glucose and oxygen.
```

Upload via dashboard → System extracts:
- Concepts: Photosynthesis, Process, Plants, Sunlight, Energy, Chlorophyll, Light, Carbon, Dioxide, Water, Glucose, Oxygen
- Relations: Process related_to Photosynthesis, Plants related_to Use, etc.

### 2. Query the System

**Query:** "What is photosynthesis?"

**Response:**
- Confidence: 75%
- Facts: "Process related_to Photosynthesis"
- Source: Original sentence from text
- Associated concepts: Plants, Energy, Light

### 3. Test Understanding

**Query:** "How do plants get energy?"

**Response:**
- Retrieves photosynthesis facts
- Links plants → photosynthesis → energy
- Shows reasoning trace

## Limitations and Future Work

### Current Limitations

1. **Pattern-based relation extraction**: Limited to predefined patterns
2. **Simple concept extraction**: Relies on capitalization and length heuristics
3. **No coreference resolution**: "It" and "they" not linked to referents
4. **No temporal reasoning**: Cannot understand sequences of events
5. **Limited abstraction**: Concepts stored as-is without generalization

### Future Enhancements

1. **Improved NLP**: Better tokenization, POS tagging, dependency parsing
2. **Coreference resolution**: Track entities across sentences
3. **Abstraction learning**: Generalize concepts (e.g., "dog" and "cat" → "animal")
4. **Temporal reasoning**: Understand before/after/during relationships
5. **Multi-document synthesis**: Integrate knowledge from multiple sources
6. **Active learning**: Ask clarifying questions about ambiguous content

## Technical Notes

### Why No LLM for Learning?

The NSCK architecture is designed to be **LLM-independent** for core cognition:

1. **Transparency**: Hypervector operations are interpretable
2. **Efficiency**: VSA operations are computationally cheap
3. **Grounding**: Concepts grounded in symbolic representations
4. **Compositionality**: Clean algebraic operations (XOR, bundle)
5. **Scalability**: Can handle millions of concepts efficiently

The LLM (when present) is **only** for:
- Natural language parsing (optional)
- Text generation (optional)
- User interface convenience

### Hypervector Properties

- **High dimensionality**: 10,240 dimensions
- **Sparsity**: ~2% active bits in SDRs
- **Robustness**: Tolerant to noise and corruption
- **Compositionality**: Clean algebraic operations
- **Similarity**: Hamming distance for comparison

### Memory Capacity

- **Semantic Memory**: Unlimited concepts (memory permitting)
- **Episodic Memory**: 
  - Recent: 1,000 episodes in RAM
  - Long-term: 10,000+ episodes in SQLite
  - Automatic consolidation and pruning

## Troubleshooting

### "No concepts learned"

- Check that text has capitalized words or longer terms
- Try adding explicit entity names
- Use clearer subject-verb-object sentences

### "Low confidence in queries"

- Upload more text on the topic
- Use query terms that match learned concepts
- Check learned concepts in statistics panel

### "No facts found"

- Text needs explicit relational phrases ("is a", "causes", etc.)
- Try more structured text
- Check relation patterns in code

## References

- VSA/HDC: Kanerva, P. (2009). "Hyperdimensional Computing"
- Semantic Folding: Webber, F. (2016). "Semantic Folding Theory"
- NSCK Architecture: See `docs/ARCHITECTURE.md`

---

**Version**: 1.0  
**Last Updated**: 2026-02-11  
**Author**: NSCK Development Team
