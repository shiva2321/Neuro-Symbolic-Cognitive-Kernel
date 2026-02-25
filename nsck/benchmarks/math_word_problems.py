"""50 arithmetic word problems benchmark for NSCK."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MATH_PROBLEMS = [
    ("What is 3 plus 5?", 8),
    ("What is 10 minus 4?", 6),
    ("What is 6 times 7?", 42),
    ("What is 20 divided by 4?", 5),
    ("What is 15 plus 8?", 23),
    ("What is 100 minus 37?", 63),
    ("What is 9 times 9?", 81),
    ("What is 144 divided by 12?", 12),
    ("What is 25 plus 75?", 100),
    ("What is 50 minus 19?", 31),
    ("If Alice has 3 apples and Bob gives her 2, how many does she have?", 5),
    ("If there are 10 birds and 3 fly away, how many are left?", 7),
    ("A store has 24 items. If 6 are sold, how many remain?", 18),
    ("John has 5 marbles. He gets 4 more. How many does he have?", 9),
    ("There are 8 cats and 5 dogs. How many animals are there?", 13),
    ("What is 2 plus 2?", 4),
    ("What is 100 divided by 10?", 10),
    ("What is 7 times 8?", 56),
    ("What is 50 plus 50?", 100),
    ("What is 99 minus 9?", 90),
    ("What is 3 times 3?", 9),
    ("What is 12 plus 12?", 24),
    ("What is 30 minus 15?", 15),
    ("What is 4 times 5?", 20),
    ("What is 60 divided by 3?", 20),
    ("What is 11 plus 11?", 22),
    ("What is 100 minus 1?", 99),
    ("What is 6 times 6?", 36),
    ("What is 81 divided by 9?", 9),
    ("What is 45 plus 55?", 100),
    ("What is 7 plus 8?", 15),
    ("What is 20 minus 7?", 13),
    ("What is 5 times 9?", 45),
    ("What is 72 divided by 8?", 9),
    ("What is 33 plus 17?", 50),
    ("What is 80 minus 35?", 45),
    ("What is 11 times 11?", 121),
    ("What is 48 divided by 6?", 8),
    ("What is 16 plus 24?", 40),
    ("What is 90 minus 45?", 45),
    ("What is 3 times 12?", 36),
    ("What is 64 divided by 8?", 8),
    ("What is 27 plus 13?", 40),
    ("What is 55 minus 25?", 30),
    ("What is 8 times 7?", 56),
    ("What is 36 divided by 4?", 9),
    ("What is 19 plus 21?", 40),
    ("What is 75 minus 25?", 50),
    ("What is 9 times 11?", 99),
    ("What is 100 divided by 5?", 20),
]


def run_math_benchmark(engine=None) -> float:
    """Run math word problems benchmark. Returns % correct (0-100)."""
    from python.core.reasoning.math_reasoning import MathReasoner
    mr = MathReasoner()
    correct = 0
    for text, expected in MATH_PROBLEMS:
        try:
            result = mr.solve_word_problem(text)
            answer = result.get("answer") if result else None
            if answer is None:
                try:
                    answer = mr.solve_expression(text)
                except Exception:
                    pass
            if answer is not None and abs(float(answer) - float(expected)) < 0.5:
                correct += 1
        except Exception:
            pass
    return correct / len(MATH_PROBLEMS) * 100
