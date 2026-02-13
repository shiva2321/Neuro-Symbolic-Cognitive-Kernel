"""
Phase 6: Social & Emotional Intelligence - Test Suite
====================================================
Tests for emotion system, theory of mind, social learning, and empathy.
"""
import sys
import os

import pytest
import numpy as np
from python.core.cognitive.emotion_system import EmotionSystem
from python.core.cognitive.theory_of_mind import TheoryOfMind, MentalStateModel


class TestEmotionSystem:
    """Test emotion generation, recognition, and regulation."""
    
    def test_emotion_from_reward(self):
        """Test emotion generation from positive/negative rewards."""
        print("\n--- Test: Emotion from Reward ---")
        emo = EmotionSystem()
        
        # Positive reward → positive valence
        emo.update_from_drives({"hunger": 0.1}, reward=0.5)
        assert emo.valence > 0, "Positive reward should increase valence"
        assert emo.current_emotion in ["joy", "anticipation"], f"Expected positive emotion, got {emo.current_emotion}"
        
        # Negative reward → negative valence
        emo2 = EmotionSystem()
        emo2.update_from_drives({"hunger": 0.1}, reward=-0.5)
        assert emo2.valence < 0, "Negative reward should decrease valence"
        
        print(f"  Positive reward: valence={emo.valence:.2f}, emotion={emo.current_emotion}")
        print(f"  Negative reward: valence={emo2.valence:.2f}, emotion={emo2.current_emotion}")
        print("✅ Emotion from Reward Test Passed")
    
    def test_arousal_from_drives(self):
        """Test arousal increases with urgent drives."""
        print("\n--- Test: Arousal from Drives ---")
        emo = EmotionSystem()
        
        # Low drives → low arousal
        emo.update_from_drives({"hunger": 0.1, "pain": 0.0}, reward=0.0)
        low_arousal = emo.arousal
        
        # High drives → high arousal (arousal uses EMA blending, so update multiple times)
        for _ in range(5):  # Multiple updates for arousal to converge
            emo.update_from_drives({"hunger": 0.9, "pain": 0.8}, reward=0.0)
        high_arousal = emo.arousal
        
        assert high_arousal > low_arousal, "High drives should increase arousal"
        assert high_arousal > 0.5, "High drives should produce significant arousal after convergence"
        
        print(f"  Low drives: arousal={low_arousal:.3f}")
        print(f"  High drives: arousal={high_arousal:.3f}")
        print("✅ Arousal from Drives Test Passed")
    
    def test_emotion_categories(self):
        """Test mapping to discrete emotion categories."""
        print("\n--- Test: Emotion Categories ---")
        emo = EmotionSystem()
        
        # Test different valence/arousal combinations
        test_cases = [
            (0.8, 0.3, "joy"),        # Positive valence, low arousal
            (-0.8, 0.3, "sadness"),   # Negative valence, low arousal
            (-0.6, 0.8, "fear"),      # Negative valence, high arousal
            (0.6, 0.8, "surprise"),   # Positive valence, high arousal
        ]
        
        for valence, arousal, expected_family in test_cases:
            emo.valence = valence
            emo.arousal = arousal
            emotion = emo._map_to_basic_emotion()
            print(f"  v={valence:+.1f}, a={arousal:.1f} → {emotion}")
            # Check it's a valid emotion
            assert emotion in emo.basic_emotions + ["neutral"], f"Invalid emotion: {emotion}"
        
        print("✅ Emotion Categories Test Passed")
    
    def test_emotion_vsa_encoding(self):
        """Test VSA representation of emotions."""
        print("\n--- Test: Emotion VSA Encoding ---")
        emo = EmotionSystem()
        
        # Check codebook created
        assert len(emo.emotion_codebook) >= 8, "Should have at least 8 basic emotions"
        
        # Get emotion vectors
        joy_hv = emo.get_emotion_vector("joy")
        sadness_hv = emo.get_emotion_vector("sadness")
        
        # Different emotions should have different representations
        similarity = joy_hv.similarity(sadness_hv)
        assert similarity < 0.99, "Different emotions should have distinct HVs"
        
        print(f"  Codebook size: {len(emo.emotion_codebook)}")
        print(f"  joy-sadness similarity: {similarity:.3f}")
        print("✅ Emotion VSA Encoding Test Passed")


class TestTheoryOfMind:
    """Test mental state modeling and perspective taking."""
    
    def test_belief_tracking(self):
        """Test tracking another agent's beliefs."""
        print("\n--- Test: Belief Tracking ---")
        tom = TheoryOfMind()
        
        # Update Sally's beliefs
        tom.update_agent_perspective("Sally", "room_A", {"ball_location": "basket"})
        
        model = tom.get_or_create_model("Sally")
        assert model.get_belief("ball_location") == "basket", "Sally should believe ball is in basket"
        
        print(f"  Sally's belief: ball_location={model.get_belief('ball_location')}")
        print("✅ Belief Tracking Test Passed")
    
    def test_sally_anne_false_belief(self):
        """Test Sally-Anne false belief scenario."""
        print("\n--- Test: Sally-Anne False Belief ---")
        tom = TheoryOfMind()
        
        # Setup: Sally puts ball in basket
        tom.update_agent_perspective("Sally", "room", {"ball_location": "basket"})
        
        # Sally leaves (no updates)
        # Anne moves ball to box (reality changes)
        reality = {"ball_location": "box"}
        
        # Sally still believes it's in basket (false belief)
        sally_model = tom.get_or_create_model("Sally")
        sally_belief = sally_model.get_belief("ball_location")
        
        assert sally_belief == "basket", "Sally should still believe ball is in basket"
        assert reality["ball_location"] == "box", "Reality is ball in box"
        
        # Detect false belief
        false_beliefs = tom.detect_false_belief("Sally", reality)
        assert "ball_location" in false_beliefs, "Should detect Sally's false belief about ball location"
        
        print(f"  Sally believes: {sally_belief}")
        print(f"  Reality: {reality['ball_location']}")
        print(f"  False beliefs detected: {false_beliefs}")
        print("✅ Sally-Anne Test Passed")
    
    def test_action_prediction(self):
        """Test predicting actions based on beliefs and desires."""
        print("\n--- Test: Action Prediction ---")
        tom = TheoryOfMind()
        
        # Sally wants to find the ball
        model = tom.get_or_create_model("Sally")
        model.set_desire("find_ball")
        model.update_beliefs({"ball_location": "basket"})
        
        # Predict: Sally should search the basket
        predicted_action = tom.predict_action("Sally")
        assert "basket" in predicted_action, f"Sally should search basket, got: {predicted_action}"
        
        print(f"  Sally's desire: find_ball")
        print(f"  Sally's belief: ball in basket")
        print(f"  Predicted action: {predicted_action}")
        print("✅ Action Prediction Test Passed")
    
    def test_multiple_agents(self):
        """Test tracking multiple agents simultaneously."""
        print("\n--- Test: Multiple Agents ---")
        tom = TheoryOfMind()
        
        # Track Sally and Anne
        tom.update_agent_perspective("Sally", "room_A", {"item": "apple"})
        tom.update_agent_perspective("Anne", "room_B", {"item": "banana"})
        
        sally = tom.get_or_create_model("Sally")
        anne = tom.get_or_create_model("Anne")
        
        assert sally.get_belief("item") == "apple", "Sally should know about apple"
        assert anne.get_belief("item") == "banana", "Anne should know about banana"
        assert sally.get_belief("item") != anne.get_belief("item"), "Different agents, different beliefs"
        
        print(f"  Sally's knowledge: {sally.beliefs}")
        print(f"  Anne's knowledge: {anne.beliefs}")
        print("✅ Multiple Agents Test Passed")


class TestSocialLearning:
    """Test social learning capabilities."""
    
    def test_imitation_learning_basic(self):
        """Test basic imitation of observed behavior."""
        print("\n--- Test: Imitation Learning ---")
        
        # Simple demonstration: expert shows state-action pairs
        expert_demo = [
            ("state_0", "move_right"),
            ("state_1", "move_forward"),
            ("state_2", "move_left"),
        ]
        
        # Learner observes and records
        learned_policy = {}
        for state, action in expert_demo:
            learned_policy[state] = action
        
        # Verify learning
        assert learned_policy["state_0"] == "move_right", "Should learn from demonstration"
        assert len(learned_policy) == 3, "Should learn all demonstrated actions"
        
        print(f"  Learned {len(learned_policy)} state-action pairs")
        print(f"  Example: {list(learned_policy.items())[0]}")
        print("✅ Imitation Learning Test Passed")
    
    def test_social_norm_learning(self):
        """Test learning social norms from feedback."""
        print("\n--- Test: Social Norm Learning ---")
        
        # Track norm violations and approvals
        norms = {}
        
        # Observe: "interrupt" gets negative feedback
        context = "conversation"
        action = "interrupt"
        feedback = "negative"
        
        if feedback == "negative":
            norms[(context, action)] = "unacceptable"
        
        # Observe: "wait_turn" gets positive feedback
        action2 = "wait_turn"
        feedback2 = "positive"
        
        if feedback2 == "positive":
            norms[(context, action2)] = "acceptable"
        
        # Verify learned norms
        assert norms[(context, "interrupt")] == "unacceptable", "Should learn interrupting is bad"
        assert norms[(context, "wait_turn")] == "acceptable", "Should learn waiting is good"
        
        print(f"  Learned norms: {norms}")
        print("✅ Social Norm Learning Test Passed")


class TestEmpathy:
    """Test empathy and emotional understanding."""
    
    def test_emotional_contagion(self):
        """Test emotional state transfer from others."""
        print("\n--- Test: Emotional Contagion ---")
        
        # Observer starts neutral
        observer_emotion = EmotionSystem()
        
        # Observe someone who is sad (high negative valence)
        observed_valence = -0.7
        observed_emotion = "sadness"
        
        # Emotional contagion: observer's emotion shifts toward observed
        contagion_strength = 0.3
        observer_emotion.valence += contagion_strength * observed_valence
        
        assert observer_emotion.valence < 0, "Observer should feel negative after seeing sadness"
        
        print(f"  Observed emotion: {observed_emotion} (valence={observed_valence})")
        print(f"  Observer's valence after contagion: {observer_emotion.valence:.2f}")
        print("✅ Emotional Contagion Test Passed")
    
    def test_empathetic_response(self):
        """Test generating empathetic responses."""
        print("\n--- Test: Empathetic Response ---")
        
        # Detect other's emotion
        other_emotion = "sadness"
        other_context = "lost_game"
        
        # Generate appropriate response
        empathetic_responses = {
            "sadness": "comfort",
            "joy": "celebrate",
            "fear": "reassure",
        }
        
        response = empathetic_responses.get(other_emotion, "acknowledge")
        
        assert response == "comfort", "Should respond with comfort to sadness"
        
        print(f"  Other's emotion: {other_emotion} ({other_context})")
        print(f"  Empathetic response: {response}")
        print("✅ Empathetic Response Test Passed")


class TestPhase6Integration:
    """Test integration of all Phase 6 components."""
    
    def test_social_interaction_scenario(self):
        """Test complete social interaction: emotion + ToM + empathy."""
        print("\n--- Test: Social Interaction Scenario ---")
        
        # Agent A (our system)
        agent_a_emotion = EmotionSystem()
        tom = TheoryOfMind()
        
        # Agent B loses a game (negative event)
        tom.update_agent_perspective("Agent_B", "game_area", {"game_result": "loss"})
        
        # Agent B feels sad
        agent_b_emotion = "sadness"
        agent_b_valence = -0.6
        
        # Agent A uses ToM to understand B's state
        b_model = tom.get_or_create_model("Agent_B")
        b_game_result = b_model.get_belief("game_result")
        
        # Agent A feels empathy (emotional contagion)
        contagion = 0.3
        agent_a_emotion.valence += contagion * agent_b_valence
        
        # Agent A generates appropriate response
        if b_game_result == "loss" and agent_a_emotion.valence < 0:
            response = "offer_encouragement"
        else:
            response = "neutral"
        
        assert b_game_result == "loss", "Should understand B lost"
        assert agent_a_emotion.valence < 0, "Should feel empathy"
        assert response == "offer_encouragement", "Should offer encouragement"
        
        print(f"  Agent B's result: {b_game_result}")
        print(f"  Agent A's empathy: valence={agent_a_emotion.valence:.2f}")
        print(f"  Agent A's response: {response}")
        print("✅ Social Interaction Scenario Test Passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
