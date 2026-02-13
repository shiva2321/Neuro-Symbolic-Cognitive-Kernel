import numpy as np
from python.core.vsa.hypervec_py import HyperVector

class ConceptMapper:
    """
    Translates raw SNN output IDs into a bundle of Human Concepts.
    Enables the Brain to 'explain' why it recognized a digit.
    """
    def __init__(self):
        # 1. ATOMIC CONCEPT CODEBOOK
        self.concepts = {
            "ODD": HyperVector(101),
            "EVEN": HyperVector(102),
            "PRIME": HyperVector(103),
            "CURVED": HyperVector(201),
            "SHARP": HyperVector(202),
            "HORIZONTAL": HyperVector(203),
            "VERTICAL": HyperVector(204)
        }

        # 2. ALPHANUMERIC PROPERTY MAP
        # 0-9: Indices 0-9
        # A-Z: Indices 10-35
        # a-z: Indices 36-61
        self.char_map = {
            # Digits
            0: ["EVEN", "CURVED"], 1: ["VERTICAL"], 2: ["EVEN", "HORIZONTAL", "CURVED"],
            3: ["ODD", "CURVED", "HORIZONTAL"], 4: ["EVEN", "SHARP", "VERTICAL"],
            5: ["ODD", "HORIZONTAL", "CURVED"], 6: ["EVEN", "CURVED"],
            7: ["ODD", "HORIZONTAL", "SHARP"], 8: ["EVEN", "CURVED"], 9: ["ODD", "CURVED"],
            
            # Letters (Example Grounding)
            10: ["LETTER", "UPPERCASE", "SHARP"], # A
            11: ["LETTER", "UPPERCASE", "CURVED", "VERTICAL"], # B
            12: ["LETTER", "UPPERCASE", "CURVED"], # C
            13: ["LETTER", "UPPERCASE", "CURVED", "VERTICAL"], # D
            14: ["LETTER", "UPPERCASE", "HORIZONTAL", "VERTICAL"], # E
            15: ["LETTER", "UPPERCASE", "HORIZONTAL", "VERTICAL"], # F
            16: ["LETTER", "UPPERCASE", "CURVED"], # G
            17: ["LETTER", "UPPERCASE", "SHARP", "VERTICAL"] # H (and so on...)
        }
        
        # Populate the rest dynamically for the prototype
        # In a real system, each would have its own specific semantic vector.
        for i in range(18, 36):
            if i not in self.char_map: self.char_map[i] = ["LETTER", "UPPERCASE"]
        for i in range(36, 62):
            self.char_map[i] = ["LETTER", "LOWERCASE"]

    def get_explanation(self, char_id):
        if char_id not in self.char_map:
            return "UNKNOWN CONCEPT"

        properties = self.char_map[char_id]
        
        # Human Readable Labels for UI
        char_label = ""
        if char_id < 10: char_label = str(char_id)
        elif char_id < 36: char_label = chr(ord('A') + char_id - 10)
        else: char_label = chr(ord('a') + char_id - 36)

        return f"['{char_label}'] " + " + ".join(properties)

    def get_conceptual_vector(self, char_id):
        """Returns a bundled HyperVector representing the character's meaning."""
        if char_id not in self.char_map:
            return None
            
        props = self.char_map[char_id]
        # Map strings to Atomic Concepts
        vecs = []
        for p in props:
            if p in self.concepts:
                vecs.append(self.concepts[p])
        
        if not vecs: return None
        
        res = vecs[0]
        for v in vecs[1:]:
            res = res.bundle(v)
        return res

if __name__ == "__main__":
    mapper = ConceptMapper()
    print(f"Index 10 Explanation: {mapper.get_explanation(10)}")
    print(f"Index 36 Explanation: {mapper.get_explanation(36)}")
