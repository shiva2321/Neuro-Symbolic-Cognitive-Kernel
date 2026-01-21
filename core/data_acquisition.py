"""
NCGN Data Acquisition Module - External Dataset Integration

Acquires training data from various sources:
- HuggingFace Datasets (structured knowledge)
- Web Search (ConceptNet, DBpedia, Wikidata)
- Natural language processing to extract relations

Includes noise filtering and format conversion for NCGN training.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import hashlib


class DataSource(Enum):
    """Supported data sources."""
    HUGGINGFACE = "huggingface"
    CONCEPTNET = "conceptnet"
    WIKIDATA = "wikidata"
    WEB_SEARCH = "web_search"
    LOCAL_FILE = "local_file"


@dataclass
class RawFact:
    """A raw fact extracted from external sources."""
    subject: str
    relation: str
    object: str
    confidence: float = 0.5
    source: DataSource = DataSource.LOCAL_FILE
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedFact:
    """A cleaned, normalized fact ready for training."""
    subject: str
    relation: str
    object: str
    confidence: float
    is_property: bool = False  # True if object is a boolean property
    property_value: bool = True


class DataCleaner:
    """
    Cleans and normalizes raw data for NCGN training.
    
    Handles:
    - Text normalization (lowercase, strip, etc.)
    - Relation type mapping
    - Noise filtering (low confidence, duplicates)
    - Entity resolution
    """
    
    # Relation type mappings from various sources to NCGN types
    RELATION_MAP = {
        # ConceptNet relations
        "relatedto": "associated",
        "isa": "is_a",
        "hasa": "has_property",
        "partof": "part_of",
        "usedfor": "used_for",
        "capableof": "can_do",
        "desires": "desires",
        "causesdesire": "causes",
        "causes": "causes",
        "hassubevent": "temporal_next",
        "atlocation": "located_at",
        "hasproperty": "has_property",
        # Wikidata/DBpedia
        "instance of": "is_a",
        "subclass of": "is_a",
        "eats": "eats",
        "consumes": "eats",
        "feeds on": "eats",
    }
    
    # Common noise patterns to filter
    NOISE_PATTERNS = [
        r"^\s*$",  # Empty
        r"^[0-9]+$",  # Pure numbers
        r"^.{1,2}$",  # Too short
        r"[^\w\s\-]",  # Special characters
    ]
    
    # Property-type relations (result in boolean properties)
    PROPERTY_RELATIONS = {
        "is_edible", "is_animate", "is_alive", "is_solid",
        "is_dangerous", "is_safe", "can_move", "has_legs"
    }
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self.seen_facts: Set[str] = set()
        self.entity_aliases: Dict[str, str] = {}
    
    def clean_text(self, text: str) -> str:
        """Normalize text: lowercase, strip, remove extra whitespace."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[_\-]+', '_', text)
        return text
    
    def is_noise(self, text: str) -> bool:
        """Check if text matches any noise pattern."""
        for pattern in self.NOISE_PATTERNS:
            if re.match(pattern, text):
                return True
        return False
    
    def normalize_relation(self, relation: str) -> str:
        """Map external relation types to NCGN types."""
        relation = self.clean_text(relation).replace(" ", "")
        return self.RELATION_MAP.get(relation, "associated")
    
    def resolve_entity(self, entity: str) -> str:
        """Resolve entity aliases to canonical form."""
        entity = self.clean_text(entity)
        return self.entity_aliases.get(entity, entity)
    
    def add_alias(self, alias: str, canonical: str):
        """Register an entity alias."""
        self.entity_aliases[self.clean_text(alias)] = self.clean_text(canonical)
    
    def fact_hash(self, subject: str, relation: str, obj: str) -> str:
        """Generate unique hash for deduplication."""
        content = f"{subject}|{relation}|{obj}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clean_fact(self, raw: RawFact) -> Optional[CleanedFact]:
        """Clean a raw fact. Returns None if fact should be filtered."""
        # Check confidence
        if raw.confidence < self.min_confidence:
            return None
        
        # Clean text
        subject = self.resolve_entity(raw.subject)
        obj = self.resolve_entity(raw.object)
        relation = self.normalize_relation(raw.relation)
        
        # Filter noise
        if self.is_noise(subject) or self.is_noise(obj):
            return None
        
        # Deduplicate
        fact_id = self.fact_hash(subject, relation, obj)
        if fact_id in self.seen_facts:
            return None
        self.seen_facts.add(fact_id)
        
        # Check if this is a property relation
        is_property = relation in self.PROPERTY_RELATIONS
        
        return CleanedFact(
            subject=subject,
            relation=relation,
            object=obj,
            confidence=raw.confidence,
            is_property=is_property,
            property_value=True  # Default to True for properties
        )
    
    def clean_batch(self, raws: List[RawFact]) -> List[CleanedFact]:
        """Clean a batch of raw facts, filtering duplicates and noise."""
        cleaned = []
        for raw in raws:
            result = self.clean_fact(raw)
            if result:
                cleaned.append(result)
        return cleaned


class HuggingFaceLoader:
    """
    Loads datasets from HuggingFace Hub.
    
    Supports structured knowledge datasets like:
    - ConceptNet
    - Atomic
    - Custom relation extraction datasets
    """
    
    def __init__(self):
        self.datasets_available = False
        try:
            from datasets import load_dataset
            self.load_dataset = load_dataset
            self.datasets_available = True
        except ImportError:
            print("Warning: 'datasets' library not installed. HuggingFace loading disabled.")
    
    def load_conceptnet(self, max_examples: int = 10000) -> List[RawFact]:
        """Load ConceptNet data from HuggingFace."""
        if not self.datasets_available:
            return []
        
        try:
            # Try loading conceptnet subset
            dataset = self.load_dataset("conceptnet5", "conceptnet5", split="train", streaming=True)
            
            facts = []
            for i, example in enumerate(dataset):
                if i >= max_examples:
                    break
                
                facts.append(RawFact(
                    subject=example.get("arg1", ""),
                    relation=example.get("rel", ""),
                    object=example.get("arg2", ""),
                    confidence=example.get("weight", 0.5),
                    source=DataSource.HUGGINGFACE,
                    raw_text=example.get("sentence", ""),
                    metadata={"dataset": "conceptnet5"}
                ))
            
            return facts
        except Exception as e:
            print(f"Warning: Could not load ConceptNet: {e}")
            return []
    
    def load_atomic(self, max_examples: int = 5000) -> List[RawFact]:
        """Load ATOMIC commonsense knowledge."""
        if not self.datasets_available:
            return []
        
        try:
            dataset = self.load_dataset("atomic", split="train", streaming=True)
            
            facts = []
            relations = ["xNeed", "xIntent", "xWant", "xEffect", "oEffect"]
            
            for i, example in enumerate(dataset):
                if i >= max_examples:
                    break
                
                event = example.get("event", "")
                for rel in relations:
                    targets = example.get(rel, [])
                    for target in targets:
                        if target and target != "none":
                            facts.append(RawFact(
                                subject=event,
                                relation=rel.lower(),
                                object=target,
                                confidence=0.6,
                                source=DataSource.HUGGINGFACE,
                                metadata={"dataset": "atomic"}
                            ))
            
            return facts
        except Exception as e:
            print(f"Warning: Could not load ATOMIC: {e}")
            return []
    
    def load_custom_dataset(
        self,
        dataset_name: str,
        subject_field: str,
        relation_field: str,
        object_field: str,
        split: str = "train",
        max_examples: int = 5000
    ) -> List[RawFact]:
        """Load a custom HuggingFace dataset with specified field mapping."""
        if not self.datasets_available:
            return []
        
        try:
            dataset = self.load_dataset(dataset_name, split=split, streaming=True)
            
            facts = []
            for i, example in enumerate(dataset):
                if i >= max_examples:
                    break
                
                facts.append(RawFact(
                    subject=str(example.get(subject_field, "")),
                    relation=str(example.get(relation_field, "")),
                    object=str(example.get(object_field, "")),
                    confidence=0.5,
                    source=DataSource.HUGGINGFACE,
                    metadata={"dataset": dataset_name}
                ))
            
            return facts
        except Exception as e:
            print(f"Warning: Could not load {dataset_name}: {e}")
            return []


class WebKnowledgeLoader:
    """
    Loads knowledge from web APIs.
    
    Supports:
    - ConceptNet API
    - Wikidata SPARQL
    - Web search for context
    """
    
    def __init__(self):
        self.http_available = False
        try:
            import requests
            self.requests = requests
            self.http_available = True
        except ImportError:
            print("Warning: 'requests' library not installed. Web loading disabled.")
    
    def query_conceptnet(self, concept: str, limit: int = 50) -> List[RawFact]:
        """Query ConceptNet API for relations about a concept."""
        if not self.http_available:
            return []
        
        try:
            url = f"http://api.conceptnet.io/c/en/{concept}?limit={limit}"
            response = self.requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            facts = []
            
            for edge in data.get("edges", []):
                start = edge.get("start", {}).get("label", "")
                end = edge.get("end", {}).get("label", "")
                rel = edge.get("rel", {}).get("label", "")
                weight = edge.get("weight", 1.0)
                
                if start and end and rel:
                    facts.append(RawFact(
                        subject=start,
                        relation=rel,
                        object=end,
                        confidence=min(1.0, weight / 5.0),  # Normalize weight
                        source=DataSource.CONCEPTNET,
                        metadata={"api": "conceptnet"}
                    ))
            
            return facts
        except Exception as e:
            print(f"Warning: ConceptNet API error: {e}")
            return []
    
    def query_wikidata(self, concept: str, limit: int = 50) -> List[RawFact]:
        """Query Wikidata for facts about a concept."""
        if not self.http_available:
            return []
        
        # Simplified Wikidata query
        query = f"""
        SELECT ?item ?itemLabel ?property ?value ?valueLabel WHERE {{
          ?item rdfs:label "{concept}"@en .
          ?item ?property ?value .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """
        
        try:
            url = "https://query.wikidata.org/sparql"
            response = self.requests.get(
                url,
                params={"query": query, "format": "json"},
                timeout=15
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            facts = []
            
            for binding in data.get("results", {}).get("bindings", []):
                subj = binding.get("itemLabel", {}).get("value", "")
                rel = binding.get("property", {}).get("value", "").split("/")[-1]
                obj = binding.get("valueLabel", {}).get("value", "")
                
                if subj and obj:
                    facts.append(RawFact(
                        subject=subj,
                        relation=rel,
                        object=obj,
                        confidence=0.8,  # Wikidata is generally reliable
                        source=DataSource.WIKIDATA,
                        metadata={"api": "wikidata"}
                    ))
            
            return facts
        except Exception as e:
            print(f"Warning: Wikidata API error: {e}")
            return []


class TrainingDataGenerator:
    """
    Generates training examples from cleaned facts.
    
    Converts knowledge facts into NCGN training format.
    """
    
    def __init__(self, cleaner: DataCleaner):
        self.cleaner = cleaner
    
    def facts_to_lessons(
        self,
        facts: List[CleanedFact],
        lesson_name: str,
        max_per_lesson: int = 20
    ) -> List[dict]:
        """Convert cleaned facts to lesson format."""
        lessons = []
        examples = []
        
        for fact in facts:
            example = {
                "inputs": [{"id": fact.subject, "energy": 1.0}],
                "expected": [fact.object],
                "description": f"{fact.subject} {fact.relation} {fact.object}"
            }
            
            if fact.is_property:
                example["constraints"] = {
                    f"{fact.subject}.{fact.relation}": fact.property_value
                }
            
            examples.append(example)
            
            if len(examples) >= max_per_lesson:
                lessons.append({
                    "id": f"{lesson_name}_{len(lessons) + 1}",
                    "name": f"{lesson_name} Part {len(lessons) + 1}",
                    "type": "association",
                    "difficulty": 3,
                    "examples": examples
                })
                examples = []
        
        # Add remaining examples
        if examples:
            lessons.append({
                "id": f"{lesson_name}_{len(lessons) + 1}",
                "name": f"{lesson_name} Part {len(lessons) + 1}",
                "type": "association",
                "difficulty": 3,
                "examples": examples
            })
        
        return lessons
    
    def save_curriculum(self, lessons: List[dict], output_path: str):
        """Save lessons to a curriculum JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(lessons, f, indent=2)


class DataAcquisitionPipeline:
    """
    End-to-end pipeline for acquiring and preparing training data.
    
    Usage:
        pipeline = DataAcquisitionPipeline()
        pipeline.acquire_from_huggingface("conceptnet")
        pipeline.acquire_from_web(["dog", "cat", "meat", "fish"])
        pipeline.generate_curriculum("training/curricula/acquired_data.json")
    """
    
    def __init__(self, output_dir: str = "training/curricula"):
        self.output_dir = output_dir
        self.cleaner = DataCleaner()
        self.hf_loader = HuggingFaceLoader()
        self.web_loader = WebKnowledgeLoader()
        self.generator = TrainingDataGenerator(self.cleaner)
        self.raw_facts: List[RawFact] = []
        self.cleaned_facts: List[CleanedFact] = []
    
    def acquire_from_huggingface(
        self,
        dataset_name: str = "conceptnet",
        max_examples: int = 5000
    ) -> int:
        """Acquire data from HuggingFace."""
        if dataset_name == "conceptnet":
            facts = self.hf_loader.load_conceptnet(max_examples)
        elif dataset_name == "atomic":
            facts = self.hf_loader.load_atomic(max_examples)
        else:
            facts = self.hf_loader.load_custom_dataset(
                dataset_name,
                subject_field="subject",
                relation_field="relation",
                object_field="object",
                max_examples=max_examples
            )
        
        self.raw_facts.extend(facts)
        return len(facts)
    
    def acquire_from_web(
        self,
        concepts: List[str],
        sources: List[str] = ["conceptnet"]
    ) -> int:
        """Acquire data from web APIs for given concepts."""
        count = 0
        for concept in concepts:
            if "conceptnet" in sources:
                facts = self.web_loader.query_conceptnet(concept)
                self.raw_facts.extend(facts)
                count += len(facts)
            
            if "wikidata" in sources:
                facts = self.web_loader.query_wikidata(concept)
                self.raw_facts.extend(facts)
                count += len(facts)
        
        return count
    
    def clean_data(self) -> int:
        """Clean all acquired raw facts."""
        self.cleaned_facts = self.cleaner.clean_batch(self.raw_facts)
        return len(self.cleaned_facts)
    
    def generate_curriculum(
        self,
        output_filename: str = "acquired_knowledge.json",
        max_per_lesson: int = 20
    ) -> str:
        """Generate curriculum from cleaned facts."""
        if not self.cleaned_facts:
            self.clean_data()
        
        lessons = self.generator.facts_to_lessons(
            self.cleaned_facts,
            lesson_name="acquired",
            max_per_lesson=max_per_lesson
        )
        
        output_path = os.path.join(self.output_dir, output_filename)
        self.generator.save_curriculum(lessons, output_path)
        
        return output_path
    
    def get_stats(self) -> Dict[str, int]:
        """Get pipeline statistics."""
        return {
            "raw_facts": len(self.raw_facts),
            "cleaned_facts": len(self.cleaned_facts),
            "noise_filtered": len(self.raw_facts) - len(self.cleaned_facts)
        }
