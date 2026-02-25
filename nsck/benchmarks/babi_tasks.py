"""20 hand-crafted bAbI-style QA tasks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BABI_TASKS = [
    (["Mary went to the garden.", "John went to the kitchen."], "Where is Mary?", "garden"),
    (["The ball is in the kitchen.", "Mary picked up the ball.", "Mary went to the bedroom."], "Where is the ball?", "bedroom"),
    (["Daniel went back to the hallway.", "Sandra moved to the garden."], "Where is Sandra?", "garden"),
    (["John picked up the apple.", "John went to the office."], "Where is the apple?", "office"),
    (["Mary got the football.", "Mary went to the bedroom."], "Where is the football?", "bedroom"),
    (["The cats are afraid of dogs.", "Dogs are afraid of mice."], "What are cats afraid of?", "dogs"),
    (["Mice are small animals.", "Dogs are large animals."], "Are mice small?", "yes"),
    (["The office is north of the bedroom.", "The bedroom is north of the bathroom."], "What is north of the bedroom?", "office"),
    (["John grabbed the milk.", "John discarded the milk."], "Does John have the milk?", "no"),
    (["Mary is a queen.", "Queens are royalty."], "Is Mary royalty?", "yes"),
    (["The dog ran to the park.", "The dog found a stick."], "What did the dog find?", "stick"),
    (["John went to the park.", "John saw a cat."], "What did John see?", "cat"),
    (["Alice has 3 apples.", "Bob gives Alice 2 apples."], "How many apples does Alice have?", "5"),
    (["The red ball is on the table.", "The blue ball is under the table."], "Where is the red ball?", "table"),
    (["John is tired.", "When people are tired they sleep."], "What does John do?", "sleep"),
    (["Mary traveled to the cinema.", "Mary met John at the cinema."], "Where did Mary meet John?", "cinema"),
    (["The box contains a key.", "Mary opened the box."], "What did Mary find?", "key"),
    (["Today is Monday.", "Tomorrow is Tuesday."], "What day is today?", "Monday"),
    (["Dogs are mammals.", "Cats are mammals."], "Are dogs mammals?", "yes"),
    (["John is in the kitchen.", "The kitchen is in the house."], "Where is John?", "kitchen"),
]


def run_babi_benchmark(engine=None) -> float:
    """Run bAbI-style tasks. Returns % correct (0-100)."""
    try:
        if engine is None:
            from python.core.reasoning.cognitive_engine import CognitiveEngine
            engine = CognitiveEngine()
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        tkl = TextKnowledgeLearner(engine.semantic_memory)
        correct = 0
        for story, question, answer in BABI_TASKS:
            try:
                for sentence in story:
                    tkl.learn(sentence)
                state = {"text": question, "task": "babi"}
                result = engine.decide(state, "babi")
                action_str = str(result.chosen_action).lower()
                if answer.lower() in action_str:
                    correct += 1
            except Exception:
                pass
        return correct / len(BABI_TASKS) * 100
    except Exception:
        return 0.0
