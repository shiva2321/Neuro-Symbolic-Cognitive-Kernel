"""
AI-Enhanced Training Module
Uses Claude and Gemini APIs to improve neural network training quality
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from dataclasses import dataclass

from core.graph_network import GraphNetwork
from core.training_engine import TrainingEngine, TrainingConfig
from utils.text_processor import TextProcessor


@dataclass
class AITrainingConfig:
    """Configuration for AI-enhanced training"""
    use_claude: bool = True
    use_gemini: bool = True
    claude_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Enhancement settings
    optimize_hyperparameters: bool = True
    improve_text_quality: bool = True
    validate_training_data: bool = True
    generate_synthetic_data: bool = False

    # AI assistance level
    assistance_level: str = "medium"  # low, medium, high
    max_ai_calls: int = 10


class AITrainingAssistant:
    """
    AI-powered training assistant using Claude and Gemini.
    Provides intelligent feedback and optimization suggestions.
    """

    def __init__(self, config: AITrainingConfig):
        self.config = config
        self.claude_client = None
        self.gemini_client = None

        self._initialize_ai_clients()

    def _initialize_ai_clients(self):
        """Initialize Claude and Gemini API clients"""

        # Initialize Claude (Anthropic)
        if self.config.use_claude and self.config.claude_api_key:
            try:
                import anthropic
                self.claude_client = anthropic.Anthropic(
                    api_key=self.config.claude_api_key
                )
                print("✅ Claude API initialized")
            except ImportError:
                print("⚠️  anthropic package not found. Install: pip install anthropic")
            except Exception as e:
                print(f"⚠️  Claude initialization failed: {e}")

        # Initialize Gemini (Google)
        if self.config.use_gemini and self.config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
                print("✅ Gemini API initialized")
            except ImportError:
                print("⚠️  google-generativeai package not found. Install: pip install google-generativeai")
            except Exception as e:
                print(f"⚠️  Gemini initialization failed: {e}")

    def analyze_training_data(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze training data quality using AI.

        Args:
            texts: List of training texts

        Returns:
            Analysis results with suggestions
        """
        print("\n🔍 Analyzing training data quality...")

        # Sample texts for analysis
        sample_size = min(10, len(texts))
        samples = texts[:sample_size]

        analysis = {
            'quality_score': 0.0,
            'issues': [],
            'suggestions': [],
            'data_characteristics': {}
        }

        # Use Claude for detailed analysis
        if self.claude_client:
            try:
                analysis_prompt = self._create_data_analysis_prompt(samples)
                claude_analysis = self._call_claude(analysis_prompt)
                analysis.update(self._parse_analysis_response(claude_analysis))
            except Exception as e:
                print(f"Claude analysis failed: {e}")

        # Use Gemini for additional insights
        if self.gemini_client:
            try:
                gemini_analysis = self._call_gemini(
                    f"Analyze this training data and provide insights: {samples[:3]}"
                )
                analysis['additional_insights'] = gemini_analysis
            except Exception as e:
                print(f"Gemini analysis failed: {e}")

        return analysis

    def optimize_hyperparameters(self,
                                 current_config: TrainingConfig,
                                 training_history: List[Dict]) -> TrainingConfig:
        """
        Use AI to suggest optimal hyperparameters.

        Args:
            current_config: Current training configuration
            training_history: Previous training results

        Returns:
            Optimized training configuration
        """
        print("\n⚙️  Optimizing hyperparameters with AI...")

        if not self.claude_client:
            return current_config

        try:
            prompt = f"""
            Analyze this neural network training configuration and suggest optimizations:
            
            Current Configuration:
            - Learning Rate: {current_config.learning_rate}
            - Batch Size: {current_config.batch_size}
            - Epochs: {current_config.num_epochs}
            - Weight Decay: {current_config.weight_decay}
            - Momentum: {current_config.momentum}
            
            Training History:
            {json.dumps(training_history[-3:], indent=2) if training_history else "No history"}
            
            Provide optimized values in JSON format:
            {{
                "learning_rate": float,
                "batch_size": int,
                "num_epochs": int,
                "weight_decay": float,
                "momentum": float,
                "reasoning": "explanation"
            }}
            """

            response = self._call_claude(prompt)
            optimized = self._parse_json_response(response)

            if optimized:
                print(f"📊 AI Suggestions:")
                print(f"   Learning Rate: {current_config.learning_rate} → {optimized.get('learning_rate', current_config.learning_rate)}")
                print(f"   Batch Size: {current_config.batch_size} → {optimized.get('batch_size', current_config.batch_size)}")
                print(f"   Epochs: {current_config.num_epochs} → {optimized.get('num_epochs', current_config.num_epochs)}")
                print(f"   Reasoning: {optimized.get('reasoning', 'N/A')}")

                # Apply optimizations
                new_config = TrainingConfig(
                    learning_rate=optimized.get('learning_rate', current_config.learning_rate),
                    batch_size=optimized.get('batch_size', current_config.batch_size),
                    num_epochs=optimized.get('num_epochs', current_config.num_epochs),
                    weight_decay=optimized.get('weight_decay', current_config.weight_decay),
                    momentum=optimized.get('momentum', current_config.momentum)
                )

                return new_config

        except Exception as e:
            print(f"⚠️  Hyperparameter optimization failed: {e}")

        return current_config

    def improve_training_text(self, text: str) -> str:
        """
        Use AI to improve text quality before training.

        Args:
            text: Original text

        Returns:
            Improved text
        """
        if not self.config.improve_text_quality:
            return text

        # Only improve if text is short enough
        if len(text) > 2000:
            return text

        try:
            if self.gemini_client:
                prompt = f"""
                Improve this text for neural network training by:
                1. Fixing grammar and spelling
                2. Improving clarity
                3. Maintaining original meaning
                4. Keeping the same length approximately
                
                Original text:
                {text}
                
                Return only the improved text, no explanations.
                """

                improved = self._call_gemini(prompt)
                return improved if improved else text

        except Exception as e:
            print(f"Text improvement failed: {e}")

        return text

    def generate_synthetic_training_data(self,
                                        existing_texts: List[str],
                                        num_samples: int = 10) -> List[str]:
        """
        Generate synthetic training data based on existing examples.

        Args:
            existing_texts: Sample of existing training data
            num_samples: Number of synthetic samples to generate

        Returns:
            List of synthetic training texts
        """
        print(f"\n🤖 Generating {num_samples} synthetic training samples...")

        synthetic_data = []

        if not self.claude_client:
            return synthetic_data

        try:
            # Analyze existing data style
            sample_texts = existing_texts[:5]

            prompt = f"""
            Based on these example texts, generate {num_samples} similar training samples 
            that maintain the same style, tone, and structure:
            
            Examples:
            {json.dumps(sample_texts, indent=2)}
            
            Generate diverse, high-quality samples that would be good for training.
            Return as a JSON array of strings.
            """

            response = self._call_claude(prompt)
            generated = self._parse_json_response(response)

            if isinstance(generated, list):
                synthetic_data = generated[:num_samples]
                print(f"✅ Generated {len(synthetic_data)} synthetic samples")

        except Exception as e:
            print(f"⚠️  Synthetic data generation failed: {e}")

        return synthetic_data

    def evaluate_training_progress(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to evaluate training progress and provide feedback.

        Args:
            metrics: Training metrics

        Returns:
            Evaluation results with recommendations
        """
        print("\n📈 Evaluating training progress with AI...")

        evaluation = {
            'status': 'unknown',
            'recommendations': [],
            'issues': [],
            'continue_training': True
        }

        if not self.claude_client:
            return evaluation

        try:
            prompt = f"""
            Evaluate this neural network training progress:
            
            Metrics:
            {json.dumps(metrics, indent=2)}
            
            Provide evaluation in JSON format:
            {{
                "status": "excellent/good/poor",
                "recommendations": ["list of suggestions"],
                "issues": ["identified problems"],
                "continue_training": true/false,
                "explanation": "detailed analysis"
            }}
            """

            response = self._call_claude(prompt)
            result = self._parse_json_response(response)

            if result:
                evaluation.update(result)
                print(f"📊 Training Status: {evaluation.get('status', 'unknown')}")

                if evaluation.get('recommendations'):
                    print("💡 Recommendations:")
                    for rec in evaluation['recommendations']:
                        print(f"   • {rec}")

        except Exception as e:
            print(f"⚠️  Progress evaluation failed: {e}")

        return evaluation

    def suggest_training_improvements(self,
                                     graph: GraphNetwork,
                                     training_stats: Dict) -> List[str]:
        """
        Analyze graph structure and suggest training improvements.

        Args:
            graph: Current graph network
            training_stats: Training statistics

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        stats = graph.get_statistics()

        # Basic analysis
        if stats['num_nodes'] < 50:
            suggestions.append("Graph is very small. Consider adding more training data.")

        if stats['avg_edge_weight'] < 0.3:
            suggestions.append("Average edge weight is low. Increase training epochs or learning rate.")

        if stats['avg_node_degree'] < 2:
            suggestions.append("Nodes are poorly connected. Add more diverse training examples.")

        # AI-powered analysis
        if self.claude_client:
            try:
                prompt = f"""
                Analyze this graph neural network structure and suggest improvements:
                
                Statistics:
                - Nodes: {stats['num_nodes']}
                - Edges: {stats['num_edges']}
                - Avg Edge Weight: {stats['avg_edge_weight']:.4f}
                - Avg Node Degree: {stats['avg_node_degree']:.2f}
                
                Training Stats:
                {json.dumps(training_stats, indent=2)}
                
                Provide 3-5 specific, actionable suggestions to improve training.
                """

                response = self._call_claude(prompt)

                # Parse suggestions
                if response:
                    ai_suggestions = response.split('\n')
                    suggestions.extend([s.strip('- ').strip() for s in ai_suggestions if s.strip()])

            except Exception as e:
                print(f"AI suggestion generation failed: {e}")

        return suggestions[:5]  # Return top 5

    def _call_claude(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Claude API"""
        if not self.claude_client:
            return ""

        try:
            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Claude API error: {e}")
            return ""

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        if not self.gemini_client:
            return ""

        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""

    def _create_data_analysis_prompt(self, samples: List[str]) -> str:
        """Create prompt for data analysis"""
        return f"""
        Analyze these training data samples for a graph neural network:
        
        Samples:
        {json.dumps(samples, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "quality_score": 0-10,
            "issues": ["list of problems"],
            "suggestions": ["improvement suggestions"],
            "data_characteristics": {{
                "complexity": "low/medium/high",
                "diversity": "low/medium/high",
                "structure": "description"
            }}
        }}
        """

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse AI analysis response"""
        try:
            # Try to extract JSON
            result = self._parse_json_response(response)
            if result:
                return result
        except:
            pass

        # Fallback: basic parsing
        return {
            'quality_score': 7.0,
            'issues': [],
            'suggestions': [response] if response else []
        }

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from AI response"""
        try:
            # Try direct parsing
            return json.loads(response)
        except:
            pass

        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass

        return None


class AIEnhancedTrainingEngine(TrainingEngine):
    """
    Enhanced training engine with AI assistance.
    """

    def __init__(self,
                 graph: GraphNetwork,
                 config: Optional[TrainingConfig] = None,
                 ai_config: Optional[AITrainingConfig] = None):
        super().__init__(graph, config)

        self.ai_config = ai_config or AITrainingConfig()
        self.ai_assistant = AITrainingAssistant(self.ai_config)
        self.ai_call_count = 0

    def train_with_ai_assistance(self,
                                sequences: List[List[str]],
                                validate_sequences: Optional[List[List[str]]] = None) -> List:
        """
        Train with AI assistance and optimization.

        Args:
            sequences: Training sequences
            validate_sequences: Validation sequences

        Returns:
            Training metrics with AI insights
        """
        print("\n" + "="*70)
        print("AI-ENHANCED TRAINING")
        print("="*70)

        # Step 1: Analyze training data
        if self.ai_config.validate_training_data:
            sample_texts = [' '.join(seq) for seq in sequences[:10]]
            analysis = self.ai_assistant.analyze_training_data(sample_texts)

            print(f"\n📊 Data Quality Score: {analysis.get('quality_score', 'N/A')}/10")

            if analysis.get('issues'):
                print("⚠️  Issues found:")
                for issue in analysis['issues']:
                    print(f"   • {issue}")

            if analysis.get('suggestions'):
                print("💡 Suggestions:")
                for suggestion in analysis['suggestions']:
                    print(f"   • {suggestion}")

        # Step 2: Optimize hyperparameters
        if self.ai_config.optimize_hyperparameters and self.training_history:
            optimized_config = self.ai_assistant.optimize_hyperparameters(
                self.config,
                [vars(m) for m in self.training_history[-3:]]
            )
            self.config = optimized_config

        # Step 3: Generate synthetic data if requested
        if self.ai_config.generate_synthetic_data:
            sample_texts = [' '.join(seq) for seq in sequences[:5]]
            synthetic = self.ai_assistant.generate_synthetic_training_data(
                sample_texts,
                num_samples=min(10, len(sequences) // 10)
            )

            if synthetic:
                print(f"✅ Added {len(synthetic)} synthetic training samples")
                # Convert synthetic texts to sequences
                from utils.text_processor import TextProcessor
                processor = TextProcessor()
                for text in synthetic:
                    tokens = processor.tokenize(text)
                    if len(tokens) > 5:
                        sequences.append(tokens)

        # Step 4: Standard training
        print("\n🎓 Starting training with optimized parameters...")
        metrics = self.train_from_sequences(sequences, validate_sequences)

        # Step 5: Evaluate progress
        if metrics:
            final_metrics = vars(metrics[-1])
            evaluation = self.ai_assistant.evaluate_training_progress(final_metrics)

            print(f"\n{'='*70}")
            print("AI EVALUATION")
            print(f"{'='*70}")

            if evaluation.get('recommendations'):
                print("\n💡 AI Recommendations:")
                for rec in evaluation['recommendations']:
                    print(f"   • {rec}")

        # Step 6: Get improvement suggestions
        training_stats = {
            'final_loss': metrics[-1].loss if metrics else 0,
            'final_avg_weight': metrics[-1].avg_weight if metrics else 0,
            'total_updates': sum(m.num_updates for m in metrics)
        }

        suggestions = self.ai_assistant.suggest_training_improvements(
            self.graph,
            training_stats
        )

        if suggestions:
            print("\n🔧 Training Improvement Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")

        print(f"\n{'='*70}")
        print("AI-ENHANCED TRAINING COMPLETE")
        print(f"{'='*70}\n")

        return metrics


def setup_ai_training(claude_api_key: Optional[str] = None,
                     gemini_api_key: Optional[str] = None) -> AITrainingConfig:
    """
    Setup AI-enhanced training configuration.

    Args:
        claude_api_key: Anthropic Claude API key
        gemini_api_key: Google Gemini API key

    Returns:
        AI training configuration
    """
    # Try to get API keys from environment if not provided
    if not claude_api_key:
        claude_api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not gemini_api_key:
        gemini_api_key = os.environ.get('GOOGLE_API_KEY')

    config = AITrainingConfig(
        use_claude=claude_api_key is not None,
        use_gemini=gemini_api_key is not None,
        claude_api_key=claude_api_key,
        gemini_api_key=gemini_api_key,
        optimize_hyperparameters=True,
        improve_text_quality=True,
        validate_training_data=True,
        generate_synthetic_data=False,  # Set to True to generate synthetic data
        assistance_level="medium"
    )

    print("\n🤖 AI Training Assistant Configuration:")
    print(f"   Claude: {'✅ Enabled' if config.use_claude else '❌ Disabled'}")
    print(f"   Gemini: {'✅ Enabled' if config.use_gemini else '❌ Disabled'}")
    print(f"   Hyperparameter Optimization: {'✅' if config.optimize_hyperparameters else '❌'}")
    print(f"   Text Quality Improvement: {'✅' if config.improve_text_quality else '❌'}")
    print(f"   Data Validation: {'✅' if config.validate_training_data else '❌'}")
    print(f"   Synthetic Data Generation: {'✅' if config.generate_synthetic_data else '❌'}")

    return config

