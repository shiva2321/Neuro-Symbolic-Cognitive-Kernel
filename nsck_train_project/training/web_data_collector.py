"""
Web Data Collector - Fetch high-quality training data from multiple free sources
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
import re

# Try to import requests, but provide fallback
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not available. Install with: pip install requests")


class DataSource:
    """Base class for data sources"""
    
    def __init__(self, name: str, cache_dir: Path):
        self.name = name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{name}_cache.json"
        self.rate_limit_delay = 1.0  # seconds between requests
        
    def fetch_data(self, limit: int = 100) -> List[str]:
        """Fetch data from source. Override in subclasses."""
        raise NotImplementedError
        
    def load_cache(self) -> List[str]:
        """Load cached data if available"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[{self.name}] Loaded {len(data)} cached sentences")
                    return data
            except Exception as e:
                print(f"[{self.name}] Cache load error: {e}")
        return []
        
    def save_cache(self, data: List[str]):
        """Save data to cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[{self.name}] Cached {len(data)} sentences")
        except Exception as e:
            print(f"[{self.name}] Cache save error: {e}")


class SimpleWikipediaSource(DataSource):
    """Fetch data from Simple Wikipedia (easier English, more reliable)"""
    
    def __init__(self, cache_dir: Path):
        super().__init__("simple_wikipedia", cache_dir)
        self.api_base = "https://simple.wikipedia.org/api/rest_v1/page/summary/"
        self.topics = [
            # Science
            "Physics", "Chemistry", "Biology", "Astronomy", "Geology",
            "Evolution", "DNA", "Cell_(biology)", "Photosynthesis",
            # History
            "World_War_II", "Ancient_Egypt", "Roman_Empire", "Industrial_Revolution",
            "Cold_War", "Space_Race", "Renaissance", "French_Revolution",
            # Geography
            "Earth", "Ocean", "Mountain", "River", "Climate", "Ecosystem",
            "Amazon_rainforest", "Sahara", "Antarctica",
            # Technology
            "Computer", "Internet", "Artificial_intelligence", "Electricity",
            "Telephone", "Television", "Radio", "Airplane",
            # Math
            "Mathematics", "Algebra", "Geometry", "Calculus", "Statistics",
            # Culture
            "Language", "Music", "Art", "Literature", "Philosophy",
            "Democracy", "Economy", "Education", "Medicine",
            # Nature
            "Animal", "Plant", "Mammal", "Bird", "Fish", "Insect",
            "Tree", "Flower", "Bacteria", "Virus"
        ]
        
    def fetch_data(self, limit: int = 100) -> List[str]:
        """Fetch summaries from Simple Wikipedia"""
        if not HAS_REQUESTS:
            return self.load_cache()
            
        cached = self.load_cache()
        if len(cached) >= limit:
            return cached[:limit]
            
        sentences = []
        topics_to_fetch = self.topics[:limit // 3]  # Each topic yields ~3 sentences
        
        print(f"[{self.name}] Fetching {len(topics_to_fetch)} topics...")
        
        for i, topic in enumerate(topics_to_fetch):
            try:
                url = f"{self.api_base}{topic}"
                headers = {
                    'User-Agent': 'Educational AI Research Bot/1.0 (educational purposes)',
                    'Accept': 'application/json'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    extract = data.get('extract', '')
                    
                    # Split into sentences
                    topic_sentences = self._extract_sentences(extract)
                    sentences.extend(topic_sentences)
                    
                    print(f"  [{i+1}/{len(topics_to_fetch)}] {topic}: {len(topic_sentences)} sentences")
                    
                elif response.status_code == 404:
                    print(f"  [{i+1}/{len(topics_to_fetch)}] {topic}: Not found (404)")
                else:
                    print(f"  [{i+1}/{len(topics_to_fetch)}] {topic}: HTTP {response.status_code}")
                    
                # Rate limiting
                time.sleep(self.rate_limit_delay + random.uniform(0, 0.5))
                
            except Exception as e:
                print(f"  [{i+1}/{len(topics_to_fetch)}] {topic}: Error - {e}")
                
        if sentences:
            # Combine with cached data and deduplicate
            all_sentences = list(set(cached + sentences))
            self.save_cache(all_sentences)
            return all_sentences
        else:
            print(f"[{self.name}] Fetch failed, using cache ({len(cached)} sentences)")
            return cached
            
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract clean sentences from text"""
        # Remove parenthetical clarifications
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter
        clean = []
        for s in sentences:
            s = s.strip()
            # Keep sentences that are informative (10-200 chars, has content words)
            if 10 <= len(s) <= 200 and any(c.isalpha() for c in s):
                clean.append(s)
                
        return clean[:5]  # Limit sentences per topic


class QuotesSource(DataSource):
    """Curated famous quotes and sayings"""
    
    def __init__(self, cache_dir: Path):
        super().__init__("quotes", cache_dir)
        
    def fetch_data(self, limit: int = 100) -> List[str]:
        """Return curated quotes"""
        quotes = [
            # Science quotes
            "The important thing is not to stop questioning.",
            "Science is organized knowledge. Wisdom is organized life.",
            "The good thing about science is that it's true whether or not you believe in it.",
            "Somewhere, something incredible is waiting to be known.",
            "Science knows no country, because knowledge belongs to humanity.",
            
            # Philosophy
            "I think, therefore I am.",
            "The unexamined life is not worth living.",
            "Knowledge is power.",
            "Time is what we want most, but what we use worst.",
            "Life must be understood backward, but it must be lived forward.",
            
            # History
            "Those who cannot remember the past are condemned to repeat it.",
            "History is written by the victors.",
            "The only thing we learn from history is that we learn nothing from history.",
            
            # Wisdom
            "The only way to do great work is to love what you do.",
            "Innovation distinguishes between a leader and a follower.",
            "Stay hungry, stay foolish.",
            "Success is not final, failure is not fatal: it is the courage to continue that counts.",
            
            # Technology
            "Technology is best when it brings people together.",
            "The advance of technology is based on making it fit in so that you don't really even notice it.",
            "Any sufficiently advanced technology is indistinguishable from magic.",
            
            # Nature
            "In every walk with nature one receives far more than he seeks.",
            "Look deep into nature, and then you will understand everything better.",
            "Nature does not hurry, yet everything is accomplished.",
        ]
        
        self.save_cache(quotes)
        return quotes


class FactsDatabase(DataSource):
    """Expanded curated facts database"""
    
    def __init__(self, cache_dir: Path):
        super().__init__("facts_db", cache_dir)
        
    def fetch_data(self, limit: int = 100) -> List[str]:
        """Return expanded curated facts"""
        facts = []
        
        # Physics & Chemistry (50 facts)
        facts.extend([
            "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "Light travels at about 186,000 miles per second in a vacuum.",
            "E=mc² is Einstein's famous equation relating mass and energy.",
            "Newton's first law states that objects in motion stay in motion unless acted upon by a force.",
            "Gravity is the force that attracts two bodies toward each other.",
            "The electromagnetic spectrum includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays.",
            "Atoms are the basic building blocks of matter.",
            "Water is composed of two hydrogen atoms and one oxygen atom (H2O).",
            "The periodic table organizes elements by atomic number and chemical properties.",
            "Temperature is a measure of the average kinetic energy of particles.",
            "Absolute zero is -273.15 degrees Celsius or 0 Kelvin.",
            "Sound travels through air at approximately 343 meters per second at room temperature.",
            "Electricity is the flow of electric charge through a conductor.",
            "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
            "The strong nuclear force holds atomic nuclei together.",
            "Friction is the force that opposes motion between surfaces in contact.",
            "Momentum is the product of mass and velocity.",
            "Energy cannot be created or destroyed, only transformed.",
            "Wavelength and frequency are inversely related in wave motion.",
            "The Earth's magnetic field protects us from solar radiation.",
        ])
        
        # Biology (50 facts)
        facts.extend([
            "DNA carries genetic information in all living organisms.",
            "Cells are the basic functional units of life.",
            "Photosynthesis is the process by which plants convert light into chemical energy.",
            "Mitochondria are the powerhouses of the cell.",
            "The human body has approximately 37 trillion cells.",
            "Evolution occurs through natural selection over generations.",
            "Proteins are made of amino acids linked together.",
            "The heart pumps blood throughout the circulatory system.",
            "The brain contains approximately 86 billion neurons.",
            "Chlorophyll is the green pigment that captures light for photosynthesis.",
            "Respiration converts glucose and oxygen into energy, water, and carbon dioxide.",
            "Enzymes are biological catalysts that speed up chemical reactions.",
            "The human genome contains about 20,000-25,000 genes.",
            "Bacteria are single-celled microorganisms found everywhere on Earth.",
            "Viruses require host cells to reproduce.",
            "The immune system defends the body against pathogens.",
            "Ecosystems are communities of living organisms interacting with their environment.",
            "Biodiversity refers to the variety of life forms in an ecosystem.",
            "Symbiosis is a close relationship between two different species.",
            "Metamorphosis is the process of transformation in some animals.",
        ])
        
        # History (50 facts)
        facts.extend([
            "World War II lasted from 1939 to 1945.",
            "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century.",
            "The Industrial Revolution began in Britain in the late 18th century.",
            "The Roman Empire fell in 476 CE.",
            "The Great Wall of China was built over many centuries to protect against invasions.",
            "The American Declaration of Independence was signed in 1776.",
            "The French Revolution began in 1789.",
            "The Cold War was a period of tension between the US and Soviet Union from 1947 to 1991.",
            "The first human landed on the Moon on July 20, 1969.",
            "The printing press was invented by Johannes Gutenberg around 1440.",
            "The ancient Egyptians built pyramids as tombs for pharaohs.",
            "Julius Caesar was assassinated in 44 BCE.",
            "The Magna Carta was signed in 1215, limiting royal power in England.",
            "Christopher Columbus reached the Americas in 1492.",
            "The Berlin Wall fell in 1989, symbolizing the end of the Cold War.",
            "The United Nations was founded in 1945 after World War II.",
            "The steam engine revolutionized transportation and industry.",
            "Ancient Greece is considered the birthplace of democracy.",
            "The Black Death killed millions in Europe during the 14th century.",
            "The Silk Road was an ancient trade route connecting Asia and Europe.",
        ])
        
        # Geography (50 facts)
        facts.extend([
            "Mount Everest is the tallest mountain on Earth at 8,849 meters.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "The Sahara is the largest hot desert in the world.",
            "The Amazon rainforest produces about 20% of Earth's oxygen.",
            "Antarctica is the coldest continent on Earth.",
            "The Nile River is traditionally considered the longest river at over 6,600 kilometers.",
            "The Earth's crust is divided into tectonic plates that slowly move.",
            "The atmosphere is composed mainly of nitrogen (78%) and oxygen (21%).",
            "The Grand Canyon was carved by the Colorado River over millions of years.",
            "Tropical rainforests contain more than half of Earth's plant and animal species.",
            "The Earth is approximately 4.5 billion years old.",
            "Volcanoes form when magma from Earth's interior reaches the surface.",
            "Glaciers are massive bodies of ice that flow slowly over land.",
            "The water cycle involves evaporation, condensation, and precipitation.",
            "Continents drift apart at a rate of about 2-5 centimeters per year.",
            "The Great Barrier Reef is the world's largest coral reef system.",
            "Mountains form through tectonic plate collisions.",
            "The Dead Sea is the lowest point on Earth's land surface.",
            "Earthquakes occur along fault lines where tectonic plates meet.",
            "The hydrosphere contains all the water on Earth.",
        ])
        
        # Technology & Computing (50 facts)
        facts.extend([
            "The first electronic computer was built in the 1940s.",
            "The Internet began as ARPANET in 1969.",
            "Binary code uses only two digits: 0 and 1.",
            "Artificial intelligence aims to create machines that can think and learn.",
            "The World Wide Web was invented by Tim Berners-Lee in 1989.",
            "Computers process information using transistors.",
            "Cloud computing stores data on remote servers accessible via the Internet.",
            "Machine learning enables computers to learn from data without explicit programming.",
            "Encryption secures data by encoding it so only authorized parties can read it.",
            "The first mobile phone call was made in 1973.",
            "Silicon is the primary material used in computer chips.",
            "Quantum computers use quantum bits or qubits instead of classical bits.",
            "The first programming language was Fortran, developed in 1957.",
            "GPS satellites enable precise location tracking anywhere on Earth.",
            "Fiber optic cables transmit data using pulses of light.",
            "Social media platforms connect billions of people worldwide.",
            "Blockchain technology creates secure, distributed ledgers.",
            "3D printing builds objects layer by layer from digital models.",
            "Virtual reality creates immersive simulated environments.",
            "Moore's Law observes that computing power doubles approximately every two years.",
        ])
        
        # Mathematics (30 facts)
        facts.extend([
            "Pi is the ratio of a circle's circumference to its diameter, approximately 3.14159.",
            "The Pythagorean theorem states that a² + b² = c² in right triangles.",
            "Zero was invented as a number around 500 CE in India.",
            "Prime numbers are divisible only by 1 and themselves.",
            "The golden ratio is approximately 1.618 and appears throughout nature.",
            "Infinity is a concept of something without any limit.",
            "Fractals are patterns that repeat at different scales.",
            "The Fibonacci sequence starts with 0, 1, and each number is the sum of the previous two.",
            "Probability measures the likelihood of an event occurring.",
            "Calculus deals with rates of change and accumulation.",
            "Euler's number e is approximately 2.71828.",
            "Statistics analyzes and interprets data.",
            "Geometry studies shapes, sizes, and spatial relationships.",
            "Algebra uses symbols to represent numbers in equations.",
            "Complex numbers include real and imaginary parts.",
        ])
        
        # Random additional facts
        facts.extend([
            "The Sun is a medium-sized star at the center of our solar system.",
            "Photons are particles of light.",
            "Water expands when it freezes, unlike most substances.",
            "The human eye can distinguish about 10 million different colors.",
            "Lightning is a discharge of electricity in the atmosphere.",
        ])
        
        self.save_cache(facts)
        return facts[:limit]


class WebDataCollector:
    """Main collector that orchestrates multiple data sources"""
    
    def __init__(self, cache_dir: str = "web_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize all data sources
        self.sources = {
            'facts_db': FactsDatabase(self.cache_dir),
            'quotes': QuotesSource(self.cache_dir),
            'simple_wiki': SimpleWikipediaSource(self.cache_dir),
        }
        
    def collect_all(self, target_count: int = 500) -> Dict[str, List[str]]:
        """Collect data from all sources"""
        print(f"\n{'='*80}")
        print("WEB DATA COLLECTION")
        print(f"{'='*80}")
        print(f"Target: {target_count} training sentences\n")
        
        all_data = {}
        total_collected = 0
        
        # Collect from each source
        for name, source in self.sources.items():
            print(f"\nSource: {name}")
            print("-" * 40)
            
            try:
                data = source.fetch_data(limit=target_count // len(self.sources))
                all_data[name] = data
                total_collected += len(data)
                print(f"✓ Collected: {len(data)} sentences")
            except Exception as e:
                print(f"✗ Error: {e}")
                all_data[name] = []
                
        # Deduplicate across sources
        print(f"\n{'='*80}")
        all_sentences = []
        seen = set()
        
        for source_name, sentences in all_data.items():
            for s in sentences:
                s_clean = s.strip().lower()
                if s_clean not in seen and len(s) >= 10:
                    seen.add(s_clean)
                    all_sentences.append(s)
                    
        print(f"Total Unique Sentences: {len(all_sentences)}")
        print(f"Target Progress: {len(all_sentences)}/{target_count} ({100*len(all_sentences)//target_count}%)")
        
        # Save combined corpus
        corpus_file = self.cache_dir / "combined_training_corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'collected_at': datetime.now().isoformat(),
                    'total_sentences': len(all_sentences),
                    'sources': {k: len(v) for k, v in all_data.items()}
                },
                'sentences': all_sentences
            }, f, indent=2, ensure_ascii=False)
            
        print(f"\n✓ Saved to: {corpus_file}")
        print(f"{'='*80}\n")
        
        return all_data
        
    def get_training_data(self) -> List[str]:
        """Get all collected data for training"""
        corpus_file = self.cache_dir / "combined_training_corpus.json"
        
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['sentences']
        else:
            print("No cached corpus found. Run collect_all() first.")
            return []


def main():
    """Test the web data collector"""
    collector = WebDataCollector()
    
    # Collect data from all sources
    data = collector.collect_all(target_count=500)
    
    # Show samples from each source
    print("\nSAMPLE DATA:")
    print("="*80)
    for source_name, sentences in data.items():
        if sentences:
            print(f"\n{source_name.upper()} (showing 3 of {len(sentences)}):")
            for i, s in enumerate(sentences[:3], 1):
                print(f"  {i}. {s[:100]}...")
                
    # Get combined training data
    training_data = collector.get_training_data()
    print(f"\n{'='*80}")
    print(f"READY FOR TRAINING: {len(training_data)} sentences")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
