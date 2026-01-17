import os
import sys
import time
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.flash import FlashColony

BRAIN_FILE = "main_brain.dat"

# Dynamic Vocabulary Management
class WordCortex:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 1
        self.brain = None
        self._connect_brain()

    def _connect_brain(self):
        if not os.path.exists(BRAIN_FILE):
            print(f"🧠 GIVING BIRTH: Creating {BRAIN_FILE}...")
            self.brain = FlashColony(BRAIN_FILE)
            # Start small, expand dynamically
            self.brain.create_new(initial_nodes=100)
            print("✓ Brain Born.")
        else:
            print(f"🧠 WAKING UP: Loading {BRAIN_FILE}...")
            self.brain = FlashColony(BRAIN_FILE)
            self.brain.connect_to_file()
            # We need to rebuild the vocab map from previous sessions
            # (In a real app, we'd save vocab.json, but here we relearn fast)
            print("✓ Brain Online.")

    def get_token_id(self, word, create_new=True):
        word = word.upper()
        if word in self.word_to_id:
            return self.word_to_id[word]

        if create_new:
            new_id = self.next_id
            self.word_to_id[word] = new_id
            self.id_to_word[new_id] = word
            self.next_id += 1

            # Ensure physical brain has space
            if new_id >= self.brain.max_nodes:
                self.brain.ensure_capacity(new_id + 50)

            # Initialize the new neuron
            self.brain.add_node(new_id, "hidden", threshold=0.5)
            return new_id
        return None

    def find_synapse_index(self, source_id, target_id):
        node = self.brain.get_node(source_id)
        if not node: return -1
        for idx, (tid, _, _, _) in enumerate(node.iter_synapses()):
            if tid == target_id: return idx
        return -1

    def learn_sequence(self, text):
        print(f"   Processing: {text[:40]}...")
        # Split by non-alphanumeric but keep punctuation as separate tokens if needed
        # Simple split: just words
        words = re.findall(r"[\w']+|[.,!?;]", text.upper())

        for i in range(len(words) - 1):
            curr_w = words[i]
            next_w = words[i+1]

            src_id = self.get_token_id(curr_w)
            tgt_id = self.get_token_id(next_w)

            # Check if connection exists
            idx = self.find_synapse_index(src_id, tgt_id)

            if idx == -1:
                # Create new connection
                self.brain.connect_binary(src_id, tgt_id, 1.0)
            else:
                # Strengthen existing
                curr_w_val = self.brain.get_synapse_weight(src_id, idx)
                new_w = min(5.0, curr_w_val + 0.5)
                self.brain.update_synapse_weight(src_id, idx, new_w)

        print(f"   ✓ Learned {len(words)} tokens.")

    def chat(self):
        print("\n💬 CHAT INTERFACE (Word Level)")
        print("   Type 'MACHINE' to see what happens.")
        print("-" * 50)

        while True:
            try: user_input = input("You: ").upper()
            except: break
            if user_input == "EXIT": break
            if not user_input: continue

            # Extract the last word the user typed
            words = re.findall(r"[\w']+|[.,!?;]", user_input)
            if not words: continue
            last_word = words[-1]

            curr_id = self.get_token_id(last_word, create_new=False)
            if not curr_id:
                print("   (I don't know that word yet)")
                continue

            sys.stdout.write("Bot: ")

            history = []

            # Generation Loop
            for _ in range(20): # Generate 20 words
                # 1. Fire Current Neuron
                self.brain.reset()
                self.brain.set_input(curr_id, 2.0)
                self.brain.step(0,0)

                # 2. Find Strongest Output
                best_id = -1
                max_pot = 0.1

                # We need to scan ALL active nodes (FlashColony tracks active_next)
                # But since we just stepped, we can check neighbor potentials via manual scan
                # Optimization: Scan only known vocab to save time
                candidates = []
                for nid in self.id_to_word:
                    if nid == curr_id: continue
                    pot = self.brain.get_node(nid).potential
                    if pot > 0.1:
                        candidates.append((nid, pot))

                candidates.sort(key=lambda x: x[1], reverse=True)

                if candidates:
                    # Pick top 1 (Winner Take All)
                    best_id = candidates[0][0]

                if best_id != -1:
                    word = self.id_to_word[best_id]

                    # Formatting: Don't put space before punctuation
                    if word in ".,!?;":
                        sys.stdout.write(word)
                    else:
                        sys.stdout.write(" " + word)
                    sys.stdout.flush()

                    curr_id = best_id
                    history.append(best_id)
                    time.sleep(0.1)

                    if word in ".!?": break
                else:
                    break
            print()

    def close(self):
        self.brain.close()

    def factory_reset(self):
        print("\n⚠️  FACTORY RESET...")
        self.brain.close()
        try: os.remove(BRAIN_FILE)
        except: pass
        self.__init__() # Re-init

if __name__ == "__main__":
    cortex = WordCortex()

    while True:
        print("\nMAIN MENU")
        print("1. Read Story")
        print("2. Chat")
        print("3. Factory Reset")
        print("4. Exit")
        choice = input("Select: ")

        if choice == "1":
            print("Paste story:")
            cortex.learn_sequence(input("> "))
        elif choice == "2":
            cortex.chat()
        elif choice == "3":
            cortex.factory_reset()
        elif choice == "4":
            cortex.close()
            break