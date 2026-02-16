#!/usr/bin/env python3
"""
Expanded Real-World Data Loader for NSCK AI Model
=================================================

Expanded dataset loader with 100+ diverse training samples.

Target: Scale from 30 samples to 100+ samples across multiple domains.

Categories:
- News articles (28 samples)
- Wikipedia content (28 samples)
- Technical documentation (24 samples)
- Conversational data (20+ conversations)
"""

import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExpandedRealWorldDataLoader:
    """
    Loads expanded real-world data (100+ samples) for comprehensive training.
    """
    
    def __init__(self, data_dir: str = "./realworld_data_expanded"):
        """Initialize expanded data loader."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.news_dir = self.data_dir / "news"
        self.wiki_dir = self.data_dir / "wikipedia"
        self.tech_dir = self.data_dir / "technical"
        self.conv_dir = self.data_dir / "conversational"
        
        for d in [self.news_dir, self.wiki_dir, self.tech_dir, self.conv_dir]:
            d.mkdir(exist_ok=True)
    
    def get_expanded_news_articles(self) -> List[str]:
        """
        Get 28 news articles across diverse topics.
        
        Returns:
            List of news article texts
        """
        articles = [
            # Technology (7 articles)
            "Researchers at MIT developed a new quantum computer chip that operates at room temperature. "
            "The breakthrough eliminates the need for expensive cooling systems. "
            "This could make quantum computing more accessible to businesses and researchers worldwide.",
            
            "Apple announced a new augmented reality headset with advanced AI capabilities. "
            "The device uses machine learning to understand user gestures and environment. "
            "Industry experts predict this will revolutionize how people interact with digital content.",
            
            "Cybersecurity experts discovered a major vulnerability in commonly used encryption software. "
            "The flaw affects millions of devices worldwide. "
            "Companies are rushing to release security patches to protect user data.",
            
            "Tesla unveiled a new electric semi-truck with 500-mile range. "
            "The vehicle uses advanced battery technology and autonomous driving features. "
            "Major logistics companies have already placed orders for hundreds of units.",
            
            "Google's AI system achieved breakthrough in protein folding prediction. "
            "The technology could accelerate drug discovery and medical research. "
            "Scientists are using the system to design new treatments for diseases.",
            
            "Microsoft launched a new cloud computing service for small businesses. "
            "The platform offers AI-powered tools for data analysis and automation. "
            "Early adopters report 40% improvement in operational efficiency.",
            
            "Samsung developed flexible display technology for foldable smartphones. "
            "The screens can be folded thousands of times without degradation. "
            "This innovation opens new possibilities for mobile device design.",
            
            # Science (7 articles)
            "NASA's James Webb telescope discovered potentially habitable exoplanets. "
            "The planets orbit a star 100 light-years away from Earth. "
            "Scientists detected signatures of water vapor in their atmospheres.",
            
            "Medical researchers found a new treatment for Alzheimer's disease. "
            "Clinical trials showed significant improvement in patient cognitive function. "
            "The therapy targets protein buildup in brain cells.",
            
            "Climate scientists reported accelerating ice melt in Antarctica. "
            "Satellite data shows ice loss has doubled in the past decade. "
            "Rising sea levels threaten coastal cities around the world.",
            
            "Biologists discovered a new species of deep-sea fish near hydrothermal vents. "
            "The creature has unique adaptations for surviving extreme pressure and temperature. "
            "This finding expands our understanding of life in extreme environments.",
            
            "Chemists created a new catalyst that converts CO2 into useful fuels. "
            "The process is efficient and could help reduce greenhouse gas emissions. "
            "Industrial applications are being tested at pilot facilities.",
            
            "Astronomers detected mysterious radio signals from distant galaxy. "
            "The signals repeat in a regular pattern over 16 days. "
            "Researchers are working to determine their origin and meaning.",
            
            "Geologists found evidence of ancient ocean on Mars. "
            "Rock samples contain minerals that form only in presence of water. "
            "The discovery suggests Mars once had conditions suitable for life.",
            
            # Business/Economy (7 articles)
            "Global stock markets reached record highs amid economic recovery. "
            "Investors showed confidence in technology and renewable energy sectors. "
            "Analysts predict continued growth despite inflation concerns.",
            
            "Major retail chain announced shift to fully renewable energy. "
            "The company plans to power all stores with solar and wind by 2025. "
            "This move is expected to reduce operating costs by 20%.",
            
            "Cryptocurrency markets experienced significant volatility this week. "
            "Bitcoin prices fluctuated between $40,000 and $50,000. "
            "Regulatory uncertainty continues to impact investor sentiment.",
            
            "Unemployment rates dropped to lowest level in five years. "
            "Job growth was strongest in healthcare and technology sectors. "
            "Economists credit government stimulus and business reopenings.",
            
            "International trade agreement signed between major economies. "
            "The deal reduces tariffs on agricultural and manufactured goods. "
            "Businesses expect this will boost exports and create jobs.",
            
            "Oil prices surged due to supply chain disruptions. "
            "Production cuts and increased demand drove prices higher. "
            "Consumers face rising costs for gasoline and heating.",
            
            "Real estate market shows signs of cooling after pandemic boom. "
            "Home prices stabilized as interest rates increased. "
            "First-time buyers find more opportunities in the market.",
            
            # Health (7 articles)
            "New cancer immunotherapy shows promising results in trials. "
            "The treatment helps immune system recognize and attack tumor cells. "
            "Patients experienced longer survival rates with fewer side effects.",
            
            "Mental health services expanded to meet growing demand. "
            "Telehealth platforms make therapy more accessible to rural areas. "
            "Experts emphasize importance of early intervention and support.",
            
            "Researchers developed a blood test for early Alzheimer's detection. "
            "The test identifies biomarkers years before symptoms appear. "
            "Early diagnosis could enable more effective treatment interventions.",
            
            "WHO declared end to global pandemic emergency status. "
            "Vaccination efforts and natural immunity reduced severe cases. "
            "Health officials warn continued vigilance is necessary.",
            
            "New study links regular exercise to reduced dementia risk. "
            "Adults who exercised 150 minutes weekly showed better cognitive health. "
            "Physical activity promotes brain plasticity and neurogenesis.",
            
            "Scientists created synthetic organs using 3D bioprinting. "
            "The technology could revolutionize organ transplantation. "
            "Clinical trials are underway for kidney and liver tissues.",
            
            "Diabetes prevention program shows long-term effectiveness. "
            "Lifestyle interventions reduced disease incidence by 58%. "
            "The program focuses on diet, exercise, and behavioral changes.",
        ]
        
        # Save to files
        for i, article in enumerate(articles):
            file_path = self.news_dir / f"news_{i+1:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(article)
        
        logger.info(f"Generated {len(articles)} news articles")
        return articles
    
    def get_expanded_wikipedia_content(self) -> List[str]:
        """
        Get 28 Wikipedia-style articles across diverse topics.
        
        Returns:
            List of encyclopedia-style texts
        """
        articles = [
            # Science topics (7)
            "Photosynthesis is the process by which plants convert light energy into chemical energy. "
            "This occurs in chloroplasts containing chlorophyll pigment. "
            "Plants absorb carbon dioxide and water to produce glucose and oxygen. "
            "The overall equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.",
            
            "Quantum mechanics describes behavior of matter at atomic and subatomic scales. "
            "Key principles include wave-particle duality and uncertainty principle. "
            "Particles exist in superposition until measured. "
            "Applications include semiconductors, lasers, and quantum computing.",
            
            "Evolution is the change in heritable characteristics of populations over generations. "
            "Natural selection acts on variation within populations. "
            "Beneficial traits become more common through differential reproduction. "
            "Evidence includes fossil records, comparative anatomy, and DNA analysis.",
            
            "Plate tectonics theory explains movement of Earth's lithosphere. "
            "Tectonic plates float on the semi-fluid asthenosphere. "
            "Their interactions cause earthquakes, volcanoes, and mountain formation. "
            "Continental drift was first proposed by Alfred Wegener in 1912.",
            
            "Cellular respiration releases energy from glucose molecules. "
            "The process occurs in mitochondria through three main stages. "
            "Glycolysis, Krebs cycle, and electron transport chain produce ATP. "
            "Aerobic respiration yields approximately 36-38 ATP molecules per glucose.",
            
            "Black holes are regions where gravity is so strong light cannot escape. "
            "They form when massive stars collapse at end of life cycle. "
            "Event horizon marks the point of no return. "
            "Supermassive black holes exist at centers of most galaxies.",
            
            "Antibiotics are medicines that fight bacterial infections. "
            "They work by killing bacteria or preventing their reproduction. "
            "Different classes target different bacterial processes. "
            "Antibiotic resistance is a growing global health concern.",
            
            # History topics (7)
            "The Roman Empire was one of the largest empires in ancient history. "
            "At peak, it controlled territories across Europe, North Africa, and Asia. "
            "The empire lasted from 27 BCE to 476 CE in the West. "
            "Roman achievements include law, architecture, engineering, and language.",
            
            "World War II was a global conflict lasting from 1939 to 1945. "
            "It involved most nations organized into Allied and Axis powers. "
            "The war resulted in 70-85 million deaths. "
            "It ended with Allied victory and establishment of United Nations.",
            
            "The Industrial Revolution transformed economies from agrarian to industrial. "
            "It began in Great Britain in the late 18th century. "
            "Key developments included steam power, factories, and mass production. "
            "The revolution had profound social and economic effects worldwide.",
            
            "The Renaissance was a period of cultural rebirth in Europe. "
            "It spanned from the 14th to 17th centuries. "
            "The movement emphasized humanism, art, and scientific inquiry. "
            "Notable figures include Leonardo da Vinci and Michelangelo.",
            
            "The Cold War was a period of geopolitical tension after World War II. "
            "It lasted from 1947 to 1991 between Western and Eastern blocs. "
            "The conflict involved ideological, economic, and military competition. "
            "It ended with the dissolution of the Soviet Union.",
            
            "Ancient Egypt was a civilization along the Nile River. "
            "It lasted for over 3,000 years from 3100 BCE. "
            "Egyptians built pyramids as tombs for pharaohs. "
            "Their achievements include hieroglyphic writing and advanced mathematics.",
            
            "The French Revolution was a period of radical social change. "
            "It began in 1789 with storming of the Bastille. "
            "The revolution overthrew the monarchy and established a republic. "
            "It influenced democratic movements worldwide.",
            
            # Technology topics (7)
            "Machine learning is a subset of artificial intelligence. "
            "It enables computers to learn from data without explicit programming. "
            "Common types include supervised, unsupervised, and reinforcement learning. "
            "Applications include image recognition and natural language processing.",
            
            "The Internet is a global network of interconnected computers. "
            "It uses TCP/IP protocol suite for communication. "
            "World Wide Web is an information system accessed via the Internet. "
            "The Internet has transformed communication, commerce, and information access.",
            
            "Blockchain is a distributed ledger technology. "
            "It records transactions across multiple computers securely. "
            "Each block contains cryptographic hash of previous block. "
            "Applications include cryptocurrencies and supply chain management.",
            
            "Neural networks are computing systems inspired by biological brains. "
            "They consist of interconnected nodes organized in layers. "
            "Networks learn by adjusting connection weights through training. "
            "Deep learning uses neural networks with many layers.",
            
            "Cloud computing provides on-demand access to computing resources. "
            "Resources include servers, storage, and applications. "
            "Services are delivered over the Internet from data centers. "
            "Benefits include scalability, flexibility, and cost efficiency.",
            
            "Virtual reality creates immersive simulated environments. "
            "Users wear headsets with stereoscopic displays. "
            "Motion tracking enables natural interaction with virtual objects. "
            "Applications include gaming, training, and therapy.",
            
            "5G is the fifth generation of cellular network technology. "
            "It offers faster speeds and lower latency than 4G. "
            "The technology uses higher frequency radio waves. "
            "5G enables Internet of Things and autonomous vehicles.",
            
            # Other topics (7)
            "Climate change refers to long-term shifts in temperatures and weather patterns. "
            "Human activities since the 1800s have been the main driver. "
            "Burning fossil fuels releases greenhouse gases trapping heat. "
            "Effects include rising temperatures, melting ice, and extreme weather.",
            
            "Democracy is a system where power is vested in the people. "
            "Citizens exercise power directly or through elected representatives. "
            "Key principles include free elections, rule of law, and individual rights. "
            "Democratic systems vary including parliamentary and presidential forms.",
            
            "DNA, or deoxyribonucleic acid, is the hereditary material in organisms. "
            "It consists of two strands forming a double helix. "
            "DNA carries genetic instructions for growth and reproduction. "
            "The sequence of nucleotides determines genetic information.",
            
            "The water cycle describes continuous movement of water on Earth. "
            "Key processes include evaporation, condensation, and precipitation. "
            "Water evaporates from oceans, forms clouds, and falls as rain. "
            "This cycle is essential for distributing heat and maintaining life.",
            
            "Economics studies production, distribution, and consumption of goods. "
            "Microeconomics examines individual and business decisions. "
            "Macroeconomics analyzes economy-wide phenomena. "
            "Key concepts include supply, demand, and market equilibrium.",
            
            "Shakespeare was an English playwright and poet. "
            "He lived from 1564 to 1616 during the Elizabethan era. "
            "His works include 39 plays and 154 sonnets. "
            "Notable plays include Hamlet, Macbeth, and Romeo and Juliet.",
            
            "Buddhism is a religion and philosophy based on teachings of Buddha. "
            "It originated in ancient India in the 5th century BCE. "
            "Core beliefs include Four Noble Truths and Eightfold Path. "
            "Practices emphasize meditation, ethics, and wisdom.",
        ]
        
        # Save to files
        for i, article in enumerate(articles):
            file_path = self.wiki_dir / f"wiki_{i+1:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(article)
        
        logger.info(f"Generated {len(articles)} Wikipedia articles")
        return articles
    
    def get_expanded_technical_docs(self) -> List[str]:
        """
        Get 24 technical documentation samples.
        
        Returns:
            List of technical texts
        """
        docs = [
            # Programming (8)
            "REST APIs use HTTP methods to perform operations on resources. "
            "GET retrieves data, POST creates resources, PUT updates, DELETE removes. "
            "RESTful services are stateless with each request containing all information. "
            "Common response formats include JSON and XML.",
            
            "Python is a high-level interpreted programming language. "
            "It emphasizes code readability with significant indentation. "
            "Python supports multiple paradigms including object-oriented and functional. "
            "The language has a comprehensive standard library.",
            
            "Git is a distributed version control system. "
            "Developers create branches to work on features independently. "
            "Changes are committed locally and pushed to remote repositories. "
            "Git maintains complete history of all project changes.",
            
            "JavaScript is a programming language that runs in web browsers. "
            "It enables interactive web pages and dynamic content. "
            "Modern JavaScript includes ES6+ features like arrow functions and promises. "
            "Node.js allows JavaScript to run on servers.",
            
            "Docker containers package applications with their dependencies. "
            "Containers are lightweight and portable across environments. "
            "Dockerfile defines the container image and installation steps. "
            "Docker Compose manages multi-container applications.",
            
            "SQL is used to manage relational databases. "
            "SELECT retrieves data, INSERT adds records, UPDATE modifies, DELETE removes. "
            "Joins combine data from multiple tables based on relationships. "
            "Indexes improve query performance by creating fast lookups.",
            
            "Object-oriented programming organizes code into objects. "
            "Objects encapsulate data and methods that operate on data. "
            "Key concepts include inheritance, polymorphism, and encapsulation. "
            "Classes define blueprints for creating objects.",
            
            "React is a JavaScript library for building user interfaces. "
            "It uses component-based architecture for reusable UI elements. "
            "React employs virtual DOM for efficient updates. "
            "Hooks enable functional components to use state and effects.",
            
            # Data Science (8)
            "Data preprocessing prepares raw data for analysis. "
            "Steps include cleaning, normalization, and feature engineering. "
            "Missing values must be handled through imputation or removal. "
            "Outliers can be detected using statistical methods.",
            
            "Linear regression models relationships between variables. "
            "It assumes linear relationship between inputs and output. "
            "The model minimizes sum of squared residuals. "
            "R-squared measures proportion of variance explained.",
            
            "Classification algorithms assign data to predefined categories. "
            "Common algorithms include decision trees, SVM, and neural networks. "
            "Performance is measured using accuracy, precision, and recall. "
            "Cross-validation prevents overfitting to training data.",
            
            "Clustering groups similar data points together. "
            "K-means algorithm partitions data into k clusters. "
            "Each point belongs to cluster with nearest centroid. "
            "The algorithm iterates until centroids stabilize.",
            
            "Deep learning uses neural networks with multiple layers. "
            "Networks automatically learn hierarchical feature representations. "
            "Training requires large datasets and computational resources. "
            "Applications include computer vision and speech recognition.",
            
            "Natural language processing analyzes and understands human language. "
            "Techniques include tokenization, part-of-speech tagging, and parsing. "
            "Word embeddings represent words as dense vectors. "
            "Transformer models have achieved state-of-the-art results.",
            
            "Time series analysis examines data points over time. "
            "Components include trend, seasonality, and noise. "
            "ARIMA models predict future values based on past patterns. "
            "Applications include stock prediction and demand forecasting.",
            
            "Feature engineering creates new variables from existing data. "
            "Good features improve model performance significantly. "
            "Techniques include binning, encoding, and interaction terms. "
            "Domain knowledge guides effective feature creation.",
            
            # Systems/Infrastructure (8)
            "Load balancers distribute traffic across multiple servers. "
            "They improve application availability and reliability. "
            "Common algorithms include round-robin and least connections. "
            "Health checks ensure traffic only goes to healthy servers.",
            
            "Microservices architecture decomposes applications into small services. "
            "Each service is independently deployable and scalable. "
            "Services communicate through APIs or message queues. "
            "Benefits include flexibility and fault isolation.",
            
            "Kubernetes orchestrates containerized applications. "
            "It automates deployment, scaling, and management. "
            "Pods are smallest deployable units containing containers. "
            "Services provide stable networking for pods.",
            
            "CI/CD pipelines automate software delivery. "
            "Continuous integration merges code changes frequently. "
            "Continuous deployment automatically releases to production. "
            "Automated testing ensures quality throughout pipeline.",
            
            "Message queues enable asynchronous communication between services. "
            "Producers send messages, consumers process them independently. "
            "Queues provide buffering and load leveling. "
            "Popular systems include RabbitMQ and Apache Kafka.",
            
            "Caching stores frequently accessed data in fast storage. "
            "It reduces database load and improves response times. "
            "Cache invalidation ensures data freshness. "
            "Common implementations include Redis and Memcached.",
            
            "Monitoring tracks system health and performance. "
            "Metrics include CPU usage, memory, and request rates. "
            "Logs provide detailed information about events. "
            "Alerts notify teams of issues requiring attention.",
            
            "Database indexing improves query performance. "
            "Indexes create data structures for fast lookups. "
            "B-tree indexes are most common for range queries. "
            "Index maintenance adds overhead to write operations.",
        ]
        
        # Save to files
        for i, doc in enumerate(docs):
            file_path = self.tech_dir / f"tech_{i+1:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(doc)
        
        logger.info(f"Generated {len(docs)} technical documents")
        return docs
    
    def get_expanded_conversations(self) -> List[Dict[str, Any]]:
        """
        Get 20+ conversation samples.
        
        Returns:
            List of conversation dictionaries
        """
        conversations = [
            # Science conversations (5)
            {
                "context": "Discussing photosynthesis",
                "turns": [
                    {"user": "What is photosynthesis?", 
                     "assistant": "Photosynthesis is how plants convert light energy into chemical energy. They absorb sunlight, carbon dioxide, and water to produce glucose and oxygen."},
                    {"user": "What part of the plant does this?",
                     "assistant": "Photosynthesis occurs in chloroplasts, which contain chlorophyll. Chlorophyll is the green pigment that captures light energy."},
                    {"user": "Why is oxygen produced?",
                     "assistant": "Oxygen is a byproduct of splitting water molecules during the light reactions. Plants release this oxygen into the atmosphere."},
                ]
            },
            {
                "context": "Learning about evolution",
                "turns": [
                    {"user": "How does natural selection work?",
                     "assistant": "Natural selection acts on variation in populations. Individuals with beneficial traits survive and reproduce more successfully, passing these traits to offspring."},
                    {"user": "What creates the variation?",
                     "assistant": "Variation comes from genetic mutations and recombination during sexual reproduction. These create differences in traits among individuals."},
                ]
            },
            {
                "context": "Understanding DNA",
                "turns": [
                    {"user": "What is DNA?",
                     "assistant": "DNA is the hereditary material that carries genetic instructions. It's a double helix made of nucleotides containing genetic information."},
                    {"user": "How is it copied?",
                     "assistant": "DNA replication uses enzymes to unwind the double helix and synthesize complementary strands. This ensures each cell receives complete genetic information."},
                ]
            },
            {
                "context": "Climate science",
                "turns": [
                    {"user": "What causes climate change?",
                     "assistant": "Climate change is primarily caused by greenhouse gas emissions from human activities. Burning fossil fuels releases CO2 that traps heat in the atmosphere."},
                    {"user": "What are the effects?",
                     "assistant": "Effects include rising temperatures, melting ice caps, sea level rise, and more extreme weather events. These changes impact ecosystems and human societies."},
                ]
            },
            {
                "context": "Astronomy basics",
                "turns": [
                    {"user": "How do stars form?",
                     "assistant": "Stars form from clouds of gas and dust that collapse under gravity. As material compresses, it heats up until nuclear fusion begins in the core."},
                    {"user": "What happens when they die?",
                     "assistant": "Star death depends on mass. Small stars become white dwarfs, while massive stars explode as supernovae and may form neutron stars or black holes."},
                ]
            },
            
            # Technology conversations (5)
            {
                "context": "Learning about machine learning",
                "turns": [
                    {"user": "What is machine learning?",
                     "assistant": "Machine learning enables computers to learn from data without explicit programming. Algorithms identify patterns and make predictions based on training examples."},
                    {"user": "What are the main types?",
                     "assistant": "The three main types are supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through rewards)."},
                ]
            },
            {
                "context": "Understanding cloud computing",
                "turns": [
                    {"user": "What is cloud computing?",
                     "assistant": "Cloud computing delivers computing resources over the internet. Instead of owning servers, you access storage, processing, and applications on-demand from cloud providers."},
                    {"user": "What are the benefits?",
                     "assistant": "Benefits include scalability, flexibility, and cost efficiency. You pay only for resources used and can scale up or down based on demand."},
                ]
            },
            {
                "context": "Cybersecurity basics",
                "turns": [
                    {"user": "What is encryption?",
                     "assistant": "Encryption transforms data into unreadable format using mathematical algorithms. Only those with the correct decryption key can access the original information."},
                    {"user": "How strong is it?",
                     "assistant": "Encryption strength depends on key length and algorithm. Modern encryption like AES-256 is effectively unbreakable with current technology when implemented correctly."},
                ]
            },
            {
                "context": "Web development",
                "turns": [
                    {"user": "What is a REST API?",
                     "assistant": "REST API is an interface for programs to communicate over HTTP. It uses standard HTTP methods like GET, POST, PUT, and DELETE to perform operations on resources."},
                    {"user": "Why use APIs?",
                     "assistant": "APIs enable different applications to integrate and share data. They provide standardized ways for software components to interact without sharing code."},
                ]
            },
            {
                "context": "Databases",
                "turns": [
                    {"user": "What is a database index?",
                     "assistant": "A database index is a data structure that improves query speed. It works like a book index, allowing fast lookups without scanning entire tables."},
                    {"user": "Are there downsides?",
                     "assistant": "Indexes use additional storage space and slow down write operations. Each insert or update must also update the index structures."},
                ]
            },
            
            # History conversations (5)
            {
                "context": "Roman Empire",
                "turns": [
                    {"user": "Tell me about the Roman Empire",
                     "assistant": "The Roman Empire was one of history's largest empires. At its peak, it controlled territories across Europe, North Africa, and Western Asia from 27 BCE to 476 CE."},
                    {"user": "What were their achievements?",
                     "assistant": "Romans excelled in engineering, law, and architecture. They built roads, aqueducts, and buildings that still stand. Roman law influenced modern legal systems."},
                ]
            },
            {
                "context": "Industrial Revolution",
                "turns": [
                    {"user": "What was the Industrial Revolution?",
                     "assistant": "The Industrial Revolution transformed economies from agriculture to industry. It began in Britain in the late 1700s with inventions like the steam engine."},
                    {"user": "How did it change society?",
                     "assistant": "It led to urbanization as people moved to cities for factory work. Living standards eventually improved, but early conditions were difficult with long hours and low wages."},
                ]
            },
            {
                "context": "World War II",
                "turns": [
                    {"user": "When was World War II?",
                     "assistant": "World War II lasted from 1939 to 1945. It involved most nations organized into Allied and Axis powers, resulting in 70-85 million deaths."},
                    {"user": "What were the causes?",
                     "assistant": "Causes included Treaty of Versailles resentment, economic depression, and rise of totalitarian regimes. Territorial expansion by Germany and Japan triggered the conflict."},
                ]
            },
            {
                "context": "Ancient Egypt",
                "turns": [
                    {"user": "How were pyramids built?",
                     "assistant": "Pyramids were built using massive workforce and simple tools. Workers quarried limestone blocks, transported them on sledges, and used ramps to position them."},
                    {"user": "Why build pyramids?",
                     "assistant": "Pyramids served as tombs for pharaohs. Egyptians believed in afterlife and wanted to ensure safe passage for their rulers with treasures and provisions."},
                ]
            },
            {
                "context": "Renaissance period",
                "turns": [
                    {"user": "What was the Renaissance?",
                     "assistant": "The Renaissance was a period of cultural rebirth in Europe from 14th to 17th centuries. It emphasized humanism, art, science, and classical learning."},
                    {"user": "Who were key figures?",
                     "assistant": "Key figures included Leonardo da Vinci (artist and inventor), Michelangelo (sculptor and painter), and Galileo (astronomer and physicist)."},
                ]
            },
            
            # General knowledge conversations (5+)
            {
                "context": "Healthy lifestyle",
                "turns": [
                    {"user": "How much exercise do I need?",
                     "assistant": "Health organizations recommend 150 minutes of moderate exercise weekly. This can be walking, swimming, or cycling spread throughout the week."},
                    {"user": "What about diet?",
                     "assistant": "A balanced diet includes fruits, vegetables, whole grains, and lean proteins. Limit processed foods, added sugars, and excessive sodium for better health."},
                ]
            },
            {
                "context": "Financial planning",
                "turns": [
                    {"user": "What is compound interest?",
                     "assistant": "Compound interest is earning interest on both principal and previously earned interest. Over time, this creates exponential growth in savings or investments."},
                    {"user": "How can I save money?",
                     "assistant": "Start by creating a budget to track income and expenses. Set aside savings first, then allocate for necessities and discretionary spending."},
                ]
            },
            {
                "context": "Environmental conservation",
                "turns": [
                    {"user": "How can I reduce my carbon footprint?",
                     "assistant": "You can reduce your carbon footprint by using public transportation, reducing meat consumption, conserving energy at home, and choosing sustainable products."},
                    {"user": "Why does it matter?",
                     "assistant": "Individual actions collectively make significant impact. Reducing carbon emissions helps slow climate change and protects the environment for future generations."},
                ]
            },
            {
                "context": "Learning a language",
                "turns": [
                    {"user": "What's the best way to learn a language?",
                     "assistant": "Immersion is most effective - surround yourself with the language through media, conversation, and practice. Consistent daily practice is more important than duration."},
                    {"user": "How long does it take?",
                     "assistant": "It depends on language similarity to your native language and practice intensity. Basic conversational fluency typically takes 6-12 months of consistent study."},
                ]
            },
            {
                "context": "Music theory basics",
                "turns": [
                    {"user": "What is a musical scale?",
                     "assistant": "A musical scale is a collection of notes arranged in ascending or descending order. The most common is the major scale with its characteristic pattern of whole and half steps."},
                    {"user": "What about chords?",
                     "assistant": "Chords are three or more notes played simultaneously. Major and minor chords are most common, each creating different emotional qualities in music."},
                ]
            },
        ]
        
        # Save to files
        for i, conv in enumerate(conversations):
            file_path = self.conv_dir / f"conversation_{i+1:03d}.json"
            with open(file_path, 'w') as f:
                json.dump(conv, f, indent=2)
        
        logger.info(f"Generated {len(conversations)} conversations")
        return conversations
    
    def load_all_data(self) -> Dict[str, List[str]]:
        """
        Load all 100+ samples.
        
        Returns:
            Dictionary mapping data types to text lists
        """
        data = {
            'news': self.get_expanded_news_articles(),
            'wikipedia': self.get_expanded_wikipedia_content(),
            'technical': self.get_expanded_technical_docs(),
        }
        
        # Flatten conversations
        conversations = self.get_expanded_conversations()
        conv_texts = []
        for conv in conversations:
            for turn in conv.get('turns', []):
                conv_texts.append(turn.get('user', ''))
                conv_texts.append(turn.get('assistant', ''))
        data['conversational'] = [t for t in conv_texts if t]
        
        total_samples = sum(len(texts) for texts in data.values())
        logger.info(f"Loaded {total_samples} total samples")
        
        return data
    
    def get_combined_training_texts(self) -> List[str]:
        """
        Get all texts combined for training.
        
        Returns:
            List of all training texts (100+)
        """
        data = self.load_all_data()
        all_texts = []
        
        for category, texts in data.items():
            all_texts.extend(texts)
        
        logger.info(f"Combined {len(all_texts)} training texts")
        return all_texts
