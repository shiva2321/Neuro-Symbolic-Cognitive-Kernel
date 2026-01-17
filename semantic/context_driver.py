import os
import sys
import re
from storage.flash_colony import FlashColony

BRAIN_FILE = "semantic_brain.dat"

# GRAMMAR RULES
STOP_WORDS = {
    "THE", "A", "AN", "IT", "ITS", "AND", "BUT", "OR", "IN", "ON", "AT",
    "TO", "OF", "FOR", "WITH", "ABOUT", "LIKE", "UP", "DOWN"
}

# Verbs define the "Edge Type"
KNOWN_VERBS = {
    "IS", "ARE", "WAS", "WERE", "BE", "BEEN",
    "EAT", "EATS", "CHASE", "CHASES",
    "NEED", "NEEDS", "GROW", "GROWS",
    "WRITE", "WRITES", "CONTROL", "CONTROLS",
    "KEEP", "KEEPS", "HAS", "HAVE", "HAD",
    "SHINE", "SHINES", "ORBIT", "ORBITS",
    "FLY", "FLIES", "SWIM", "SWIMS",
    "INVENT", "INVENTS", "INVENTED",
    "CREATE", "CREATES", "CREATED",
    "BUILD", "BUILDS", "BUILT",
    "MAKE", "MAKES", "MADE",
    "DEVELOP", "DEVELOPS", "DEVELOPED",
    "DESIGN", "DESIGNS", "DESIGNED",
    "DISCOVER", "DISCOVERS", "DISCOVERED",
    "FOUND", "FOUNDED", "FOUNDER",
    "CAN", "USE", "USES", "USED",
    "RUN", "RUNS"
}

class SemanticBrain:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 1
        self.brain = None
        self._connect_brain()

    def _connect_brain(self):
        if not os.path.exists(BRAIN_FILE):
            print(f"BIRTH: Creating {BRAIN_FILE}...")
            self.brain = FlashColony(BRAIN_FILE)
            self.brain.create_new(initial_nodes=5000)
        else:
            print(f"WAKING: Loading {BRAIN_FILE}...")
            self.brain = FlashColony(BRAIN_FILE)
            self.brain.connect_to_file()
            print("Brain Online.")

    def get_token_id(self, word, create=True):
        word = word.upper()
        if word in self.word_to_id: return self.word_to_id[word]
        if create:
            new_id = self.next_id
            self.word_to_id[word] = new_id
            self.id_to_word[new_id] = word
            self.next_id += 1
            if new_id >= self.brain.max_nodes:
                self.brain.ensure_capacity(new_id + 100)
            self.brain.add_node(new_id, "hidden", threshold=0.1)
            return new_id
        return None

    def create_rdf_link(self, subj, verb, obj):
        """
        Creates a Typed Connection.
        The 'Verb' is not a node anymore. It is the TYPE of the wire.
        Normalizes verb forms (WRITES -> WRITE, INVENTED -> INVENT).
        """
        # Normalize verb to base form
        base_verb = self._normalize_verb(verb)

        sid = self.get_token_id(subj)
        oid = self.get_token_id(obj)
        vid = self.get_token_id(base_verb) # We need ID to store in float field

        # 1. Forward Link: Subject --[Verb]--> Object
        # We store the Verb_ID in the 'trace' field (index 2 in struct)
        self._inject_synapse(sid, oid, weight=1.0, type_id=vid)

        # 2. Reverse Link: Object --[Is_Verb_Of]--> Subject
        # We mark reverse links with a negative Verb_ID
        self._inject_synapse(oid, sid, weight=1.0, type_id=-vid)

    def _normalize_verb(self, verb):
        """Normalize verb to base form (WRITES -> WRITE, INVENTED -> INVENT)."""
        verb = verb.upper()
        # Remove common suffixes carefully
        if verb.endswith('IES'):  # FLIES -> FLY
            return verb[:-3] + 'Y'
        elif verb.endswith('ES') and len(verb) > 3:  # WRITES -> WRITE (but not ES -> E)
            return verb[:-1]
        elif verb.endswith('ED') and len(verb) > 3:  # INVENTED -> INVENT
            if verb[-3] == verb[-4]:  # STOPPED -> STOP
                return verb[:-3]
            return verb[:-2] if verb[-3:-2] not in 'AEIOU' else verb[:-1]
        elif verb.endswith('ING') and len(verb) > 4:  # WRITING -> WRITE
            return verb[:-3]
        elif verb.endswith('S') and not verb.endswith('SS') and len(verb) > 2:  # RUNS -> RUN
            return verb[:-1]
        return verb

    def _inject_synapse(self, src_id, tgt_id, weight, type_id):
        node = self.brain.get_node(src_id)

        # Check if link exists
        idx = -1
        for i, (tid, w, trace, perm) in enumerate(node.iter_synapses()):
            if tid == tgt_id:
                # Check if it's the SAME relationship type
                if int(trace) == int(type_id):
                    idx = i
                break

        # HACK: modifying FlashColony internals to write 'trace' manually
        import struct
        SYNAPSE_FMT = 'I f f f'
        SYNAPSE_SIZE = 16

        if idx == -1:
            # Create new
            if node.edge_count >= self.brain.max_edges_per_node: return
            offset = node.synapse_start_offset + (node.edge_count * SYNAPSE_SIZE)
            # PACK: Target, Weight, TYPE_ID (Trace), Perm
            data = struct.pack(SYNAPSE_FMT, tgt_id, weight, float(type_id), 1.0)
            self.brain.mem[offset: offset + SYNAPSE_SIZE] = data

            node.edge_count += 1
            node.write_to_disk()
        else:
            # Update existing
            pass # Already exists

    def learn_rdf(self, text):
        print(f"\nLEARNING: RDF Triples (Typed Edges)...")
        # Pre-normalization
        text = text.replace(".", " .").upper()
        tokens = text.split()

        i = 0
        while i < len(tokens):
            word = tokens[i]
            if word in KNOWN_VERBS:
                verb = word

                # Look behind for Subject
                subj = None
                j = i - 1
                while j >= 0 and tokens[j] in STOP_WORDS:
                    j -= 1
                if j >= 0 and tokens[j] != ".":
                    subj = tokens[j]

                # Look ahead for Object
                obj = None
                k = i + 1
                while k < len(tokens) and tokens[k] in STOP_WORDS:
                    k += 1
                if k < len(tokens) and tokens[k] != ".":
                    obj = tokens[k]

                if subj and subj != ".":
                    if obj and obj != ".":
                        # Subject-Verb-Object triple
                        print(f"   Triple: {subj} --[{verb}]--> {obj}")
                        self.create_rdf_link(subj, verb, obj)
                    else:
                        # Subject-Verb (intransitive) - create self-referential or special node
                        # E.g., "BIRDS FLY" -> BIRDS --[FLY]--> BIRDS or BIRDS --[CAN]--> FLY
                        # Store as: SUBJECT --[VERB]--> VERB_ACTION node
                        verb_action = f"{verb}_ACTION"
                        print(f"   Triple: {subj} --[CAN]--> {verb_action}")
                        self.create_rdf_link(subj, "CAN", verb_action)
            i += 1
        print("Graph Updated.")

    def query(self, question):
        print(f"\nQ: {question}")
        tokens = re.findall(r"[\w']+", question.upper())

        # 1. Identify Components (Improved NLP-style parsing)
        q_verb = None
        q_entity = None
        question_words = {"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}

        # Extract verb and entity
        for i, w in enumerate(tokens):
            if w in KNOWN_VERBS:
                q_verb = w
            elif w not in STOP_WORDS and w not in question_words:
                # Try to normalize verb forms (INVENTED -> INVENT, WRITING -> WRITE)
                if not q_verb:
                    # Check if this word looks like a verb (ends with ED, ING, S)
                    base_word = w.rstrip('EDINGSZ')
                    if base_word and len(base_word) > 2:
                        # Try to find matching known verb
                        for known_verb in KNOWN_VERBS:
                            if w.startswith(known_verb) or known_verb.startswith(base_word):
                                q_verb = known_verb
                                break
                        # If still no verb, treat this word as a potential verb
                        if not q_verb and not q_entity:
                            # Store as candidate verb
                            q_verb = w

                # Collect entity (the main noun being asked about)
                if w != q_verb:
                    if not q_entity or len(w) > len(q_entity):
                        q_entity = w

        # Special case: single-word queries like "load" - treat as entity
        if len(tokens) == 1 and not q_verb:
            q_entity = tokens[0]
            # Try to infer verb from context (what relates to this entity?)
            q_verb = "IS"  # Default fallback

        # Special case: "What VERB?" queries (e.g., "What flies?")
        if q_verb and not q_entity:
            # Normalize verb properly (FLIES -> FLY, WRITES -> WRITE)
            base_verb = self._normalize_verb(q_verb)
            # Look for entities that CAN do this verb
            q_entity = f"{base_verb}_ACTION"
            q_verb = "CAN"
            print(f"   [Intransitive verb detected: searching for what can {base_verb.lower()}]")

        if not q_entity:
            print("   Parser failed (No clear entity found)")
            print(f"   TIP: Try format like 'What is X?' or 'Who created Y?'")
            return

        # If no verb found, try common question patterns
        if not q_verb:
            # "Who X Y" -> "Who IS X" or "Who VERB Y"
            if "WHO" in tokens:
                q_verb = "IS"  # "Who is X?" or "What is X?"
            elif "WHAT" in tokens:
                q_verb = "IS"
            else:
                q_verb = "IS"  # Default fallback

        print(f"   [Parsed: Verb='{q_verb}', Entity='{q_entity}']")

        # Normalize verb before searching
        q_verb_normalized = self._normalize_verb(q_verb)
        print(f"   [Logic: Find X where X --[{q_verb_normalized}]--> {q_entity} OR {q_entity} --[{q_verb_normalized}]--> X]")

        eid = self.get_token_id(q_entity, False)
        vid = self.get_token_id(q_verb_normalized, False)

        if not eid or not vid:
            print("   Unknown concepts.")
            return

        # 2. SEARCH THE GRAPH
        # Strategy: We look at the Entity's neighbors.
        # We look for a wire labeled with the Verb ID.

        node = self.brain.get_node(eid)
        found = []

        for tid, w, type_id, _ in node.iter_synapses():
            # Check if this wire is the Verb we are looking for
            # If we asked "What eats worms?", we are at WORMS.
            # The wire from BIRD -> WORM is type EAT.
            # The reverse wire WORM -> BIRD is type -EAT.

            # Case A: "What eats worms?" (Reverse lookup)
            # We want X --[EAT]--> WORMS.
            # So at WORMS, we look for incoming link type -EAT.
            if int(type_id) == -int(vid):
                found.append(self.id_to_word[tid])

            # Case B: "What do birds eat?" (Forward lookup)
            # We are at BIRDS. We look for outgoing link type EAT.
            elif int(type_id) == int(vid):
                found.append(self.id_to_word[tid])

        if found:
            print(f"   ANSWER: {', '.join(found)}")
        else:
            print("   No match found.")

    def factory_reset(self):
        print("\n⚠️  FACTORY RESET...")
        self.brain.close()
        try: os.remove(BRAIN_FILE)
        except: pass
        self.__init__()