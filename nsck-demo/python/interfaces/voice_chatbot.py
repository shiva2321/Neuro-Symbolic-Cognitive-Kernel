"""
NSCK Voice Chatbot (Phase 1.4 Demo)
===================================
Integrates VoiceInterface, DialogueManager, and LanguageModule.
Demonstrates the full communication loop.
"""

import numpy as np
from python.interfaces.voice_interface import VoiceInterface
from python.core.language.dialogue_manager import DialogueManager
from python.core.language.language_module import LanguageModule
from python.core.language.lingua_cortex import get_lingua_cortex

class MockHomeostasis:
    def __init__(self):
        self.drives = {"hunger": 0.3, "pain": 0.0}
    
    def get_emotional_tone(self) -> str:
        if self.drives["hunger"] > 0.7:
            return "angry" # Hungry agents are irritable
        elif self.drives["pain"] > 0.5:
            return "sad"
        return "neutral"

def main():
    print("=== NSCK Voice Chatbot Simulation ===")
    
    # 1. Setup Components
    cortex = get_lingua_cortex()
    cortex.learn_text_snippet("food eat move garden")
    
    lang = LanguageModule()
    voice = VoiceInterface()
    homeostasis = MockHomeostasis()
    
    # Mock Engine (idle)
    dm = DialogueManager(None, lang)
    
    print("\n[System]: Voice Chatbot Ready.")
    
    # Simulate Interaction 1: Normal Greeting
    print("\n--- Interaction 1: Calm User ---")
    # Low energy, mid pitch
    user_audio = np.sin(2 * np.pi * 150 * np.linspace(0, 1, 16000)) * 0.05
    
    audio_data = voice.listen(user_audio)
    print(f"[User Voice Detected]: Emotion={audio_data['emotion']}")
    
    # In real world, we'd use audio_data['text']. 
    # For simulation, we'll use a fixed prompt.
    user_text = "Where is the food?"
    print(f"User: {user_text}")
    
    response_text = dm.process_turn(user_text)
    
    # Agent speaks back
    # Tone modulated by Homeostasis
    agent_tone = homeostasis.get_emotional_tone()
    voice.speak(response_text, emotion=agent_tone)
    
    # Simulate Interaction 2: High Hunger (Agent State Change)
    print("\n--- Interaction 2: Hungry Agent ---")
    homeostasis.drives["hunger"] = 0.8 # Boost hunger
    
    user_text = "Why haven't you moved?"
    print(f"User: {user_text}")
    
    response_text = dm.process_turn(user_text)
    
    agent_tone = homeostasis.get_emotional_tone()
    voice.speak(response_text, emotion=agent_tone)

if __name__ == "__main__":
    main()
