import pickle
import sys
import random

try:
    import hypervec_py
except ImportError:
    pass # Anticipating runtime check in server if missing

class NeuroChatbot:
    def __init__(self, codebook_path="codebook.pkl"):
        self.codebook = {}
        if codebook_path:
            try:
                with open(codebook_path, "rb") as f:
                    self.codebook = pickle.load(f)
            except (FileNotFoundError, ValueError, Exception) as e:
                print(f"Warning: Could not load codebook ({e}). Running in simplified mode.")
                self.codebook = {}

    def get_embedding(self, text):
        # Naive text to VSA: XOR sum of character vectors or word vectors?
        # For this prototype, we just look for keywords.
        # Real VSA approach: Map words to vectors, bundle them.
        # We will create random vectors for words on the fly if not in codebook (not persistent)
        # or just random projection.
        
        # Simulating "Intent Classification"
        text = text.lower()
        if "snake" in text or "play" in text:
            return "INTENT_PLAY_SNAKE"
        else:
            return "INTENT_UNKNOWN"

    def process(self, message):
        # 1. Intent Classify via VSA (Mocked Logic via existing codebook concept)
        intent_key = self.get_embedding(message)
        
        if intent_key == "INTENT_PLAY_SNAKE":
            return "Starting Snake Game Control Sequence..."
        
        # 2. Fallback to GPT-2 Style Dummy
        return self.gpt_fallback(message)

    def gpt_fallback(self, message):
        # Child-safe filter
        unsafe_words = ["die", "kill", "stupid"]
        if any(w in message.lower() for w in unsafe_words):
            return "I cannot respond to that."
            
        responses = [
            "That is interesting.",
            "Tell me more about the neural symbolism.",
            "I am calculating the free energy of that statement.",
            "Have you fed the snake hypervectors today?"
        ]
        return random.choice(responses)

if __name__ == "__main__":
    bot = NeuroChatbot(None)
    print(bot.process("Let's play snake"))
