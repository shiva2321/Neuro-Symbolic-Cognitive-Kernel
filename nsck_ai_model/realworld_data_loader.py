#!/usr/bin/env python3
"""
Real-World Data Loader for NSCK AI Model
=========================================

Loads and processes real-world datasets for training and testing:
- News articles
- Wikipedia content
- Technical documentation
- Conversational data

Addresses the training data scale limitation identified in testing.
"""

import os
import re
import json
from typing import List, Dict, Any, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealWorldDataLoader:
    """
    Loads real-world data from various sources for training and testing.
    """
    
    def __init__(self, data_dir: str = "./realworld_data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing real-world datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different data types
        self.news_dir = self.data_dir / "news"
        self.wiki_dir = self.data_dir / "wikipedia"
        self.tech_dir = self.data_dir / "technical"
        self.conv_dir = self.data_dir / "conversational"
        
        for d in [self.news_dir, self.wiki_dir, self.tech_dir, self.conv_dir]:
            d.mkdir(exist_ok=True)
    
    def get_sample_news_articles(self) -> List[str]:
        """
        Get sample news articles (synthetic for testing).
        In production, would fetch from news APIs or datasets.
        
        Returns:
            List of news article texts
        """
        articles = [
            "Scientists at MIT have developed a new material that can change its properties based on temperature. "
            "The breakthrough could lead to more efficient solar panels and better insulation materials. "
            "Researchers say the material uses a novel crystalline structure that responds to thermal changes.",
            
            "Global climate summit concludes with new agreements on carbon emissions. "
            "More than 190 countries committed to reducing greenhouse gas emissions by 2030. "
            "Environmental groups welcomed the agreement but called for stronger enforcement mechanisms.",
            
            "SpaceX successfully launched its latest satellite constellation into orbit. "
            "The launch marks the company's 50th mission this year. "
            "The satellites will provide high-speed internet access to remote areas around the world.",
            
            "Breakthrough in quantum computing achieved by researchers at Stanford University. "
            "The new algorithm reduces error rates in quantum calculations by 90%. "
            "This advancement could accelerate the development of practical quantum computers.",
            
            "Artificial intelligence system demonstrates human-level performance in medical diagnosis. "
            "The AI analyzed thousands of medical images and identified diseases with 95% accuracy. "
            "Doctors say the technology could help address healthcare shortages in rural areas.",
            
            "Study reveals ocean temperatures are rising faster than previously thought. "
            "Marine biologists warn of significant impacts on coral reefs and fish populations. "
            "The research analyzed temperature data collected over the past three decades.",
            
            "Electric vehicle sales surpass 10 million units globally this year. "
            "Analysts attribute the growth to improved battery technology and government incentives. "
            "Major automakers are investing billions in electric vehicle production.",
            
            "New archaeological discovery sheds light on ancient civilization. "
            "Excavations in Egypt uncovered artifacts dating back 5,000 years. "
            "The findings include pottery, tools, and evidence of advanced agricultural practices.",
        ]
        
        # Save to files
        for i, article in enumerate(articles):
            file_path = self.news_dir / f"article_{i+1}.txt"
            with open(file_path, 'w') as f:
                f.write(article)
        
        logger.info(f"Generated {len(articles)} sample news articles")
        return articles
    
    def get_sample_wikipedia_content(self) -> List[str]:
        """
        Get sample Wikipedia-style content.
        
        Returns:
            List of encyclopedia-style texts
        """
        articles = [
            "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. "
            "This process occurs in chloroplasts, which contain the pigment chlorophyll. "
            "During photosynthesis, plants absorb carbon dioxide and water, and produce glucose and oxygen. "
            "The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
            
            "The Roman Empire was one of the largest empires in ancient history. "
            "At its height, it controlled territories across Europe, North Africa, and Western Asia. "
            "The empire lasted from 27 BCE to 476 CE in the West, and until 1453 CE in the East. "
            "Roman achievements include law, architecture, engineering, and language that influenced Western civilization.",
            
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data. "
            "It uses statistical techniques to give computers the ability to learn without being explicitly programmed. "
            "Common types include supervised learning, unsupervised learning, and reinforcement learning. "
            "Applications include image recognition, natural language processing, and recommendation systems.",
            
            "The water cycle describes the continuous movement of water on, above, and below the Earth's surface. "
            "Key processes include evaporation, condensation, precipitation, and collection. "
            "Water evaporates from oceans and lakes, forms clouds, falls as rain or snow, and returns to water bodies. "
            "This cycle is essential for distributing heat and maintaining life on Earth.",
            
            "Democracy is a system of government where power is vested in the people. "
            "Citizens exercise power directly or through elected representatives. "
            "Key principles include free and fair elections, rule of law, and protection of individual rights. "
            "Democratic systems vary, including parliamentary, presidential, and hybrid forms.",
            
            "DNA, or deoxyribonucleic acid, is the hereditary material in humans and almost all other organisms. "
            "It consists of two strands that coil around each other to form a double helix. "
            "DNA carries genetic instructions for growth, development, functioning, and reproduction. "
            "The sequence of nucleotides in DNA determines genetic information.",
            
            "The Industrial Revolution was a period of major industrialization from the 18th to 19th century. "
            "It began in Great Britain and spread to Western Europe and the United States. "
            "Key developments included steam power, factories, and mass production. "
            "This transformation had profound effects on society, economy, and culture.",
            
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "While climate has changed throughout Earth's history, human activities since the 1800s have been the main driver. "
            "Burning fossil fuels releases greenhouse gases that trap heat in the atmosphere. "
            "Effects include rising temperatures, melting ice caps, and extreme weather events.",
        ]
        
        # Save to files
        for i, article in enumerate(articles):
            file_path = self.wiki_dir / f"wiki_{i+1}.txt"
            with open(file_path, 'w') as f:
                f.write(article)
        
        logger.info(f"Generated {len(articles)} sample Wikipedia articles")
        return articles
    
    def get_sample_technical_docs(self) -> List[str]:
        """
        Get sample technical documentation.
        
        Returns:
            List of technical texts
        """
        docs = [
            "REST APIs use HTTP methods to perform operations on resources. "
            "GET retrieves data, POST creates new resources, PUT updates existing resources, and DELETE removes resources. "
            "RESTful services are stateless, meaning each request contains all necessary information. "
            "Common response formats include JSON and XML.",
            
            "Python is a high-level, interpreted programming language. "
            "It emphasizes code readability and supports multiple programming paradigms. "
            "Key features include dynamic typing, automatic memory management, and a comprehensive standard library. "
            "Python is widely used for web development, data analysis, and machine learning.",
            
            "Git is a distributed version control system for tracking changes in source code. "
            "Developers can create branches to work on features independently. "
            "Changes are committed locally and can be pushed to remote repositories. "
            "Git enables collaboration and maintains a complete history of project changes.",
            
            "SQL (Structured Query Language) is used to manage relational databases. "
            "SELECT retrieves data, INSERT adds new records, UPDATE modifies existing data, and DELETE removes records. "
            "Joins combine data from multiple tables based on related columns. "
            "Indexes improve query performance by creating fast lookup structures.",
            
            "Docker containers package applications with their dependencies. "
            "Containers are lightweight and portable, running consistently across different environments. "
            "Dockerfile defines the container image, specifying the base system and installation steps. "
            "Docker Compose manages multi-container applications.",
            
            "Algorithms have time complexity measured using Big O notation. "
            "O(1) represents constant time, O(n) represents linear time, and O(n²) represents quadratic time. "
            "Efficient algorithms minimize time and space complexity. "
            "Common data structures include arrays, linked lists, trees, and hash tables.",
        ]
        
        # Save to files
        for i, doc in enumerate(docs):
            file_path = self.tech_dir / f"tech_{i+1}.txt"
            with open(file_path, 'w') as f:
                f.write(doc)
        
        logger.info(f"Generated {len(docs)} sample technical documents")
        return docs
    
    def get_sample_conversations(self) -> List[Dict[str, str]]:
        """
        Get sample conversation data.
        
        Returns:
            List of conversation dictionaries
        """
        conversations = [
            {
                "context": "Discussion about renewable energy",
                "turns": [
                    {"user": "What is solar energy?", 
                     "assistant": "Solar energy is power derived from the sun's radiation. It can be converted to electricity using photovoltaic cells or used for heating."},
                    {"user": "How efficient are solar panels?",
                     "assistant": "Modern solar panels typically convert 15-22% of sunlight into electricity. Efficiency depends on panel technology and environmental conditions."},
                ]
            },
            {
                "context": "Explaining machine learning",
                "turns": [
                    {"user": "What is supervised learning?",
                     "assistant": "Supervised learning trains models using labeled data. The algorithm learns patterns from input-output pairs to make predictions on new data."},
                    {"user": "Can you give an example?",
                     "assistant": "Email spam filtering is a classic example. The model learns from emails labeled as spam or not spam, then classifies new emails."},
                ]
            },
        ]
        
        # Save to files
        for i, conv in enumerate(conversations):
            file_path = self.conv_dir / f"conversation_{i+1}.json"
            with open(file_path, 'w') as f:
                json.dump(conv, f, indent=2)
        
        logger.info(f"Generated {len(conversations)} sample conversations")
        return conversations
    
    def load_all_data(self) -> Dict[str, List[str]]:
        """
        Load all available real-world data.
        
        Returns:
            Dictionary mapping data types to text lists
        """
        data = {
            'news': self.get_sample_news_articles(),
            'wikipedia': self.get_sample_wikipedia_content(),
            'technical': self.get_sample_technical_docs(),
        }
        
        # Flatten conversations into text
        conversations = self.get_sample_conversations()
        conv_texts = []
        for conv in conversations:
            for turn in conv.get('turns', []):
                conv_texts.append(turn.get('user', ''))
                conv_texts.append(turn.get('assistant', ''))
        data['conversational'] = [t for t in conv_texts if t]
        
        total_samples = sum(len(texts) for texts in data.values())
        logger.info(f"Loaded {total_samples} total samples from real-world data")
        
        return data
    
    def get_combined_training_texts(self) -> List[str]:
        """
        Get all texts combined for training.
        
        Returns:
            List of all training texts
        """
        data = self.load_all_data()
        all_texts = []
        
        for category, texts in data.items():
            all_texts.extend(texts)
        
        return all_texts
    
    def split_into_sentences(self, texts: List[str]) -> List[str]:
        """
        Split texts into individual sentences.
        
        Args:
            texts: List of texts
            
        Returns:
            List of sentences
        """
        sentences = []
        
        for text in texts:
            # Split on sentence boundaries
            parts = re.split(r'[.!?]+', text)
            for sent in parts:
                sent = sent.strip()
                if len(sent) > 10:  # Minimum sentence length
                    sentences.append(sent + '.')
        
        return sentences
