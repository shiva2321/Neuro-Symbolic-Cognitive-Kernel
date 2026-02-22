"""Frame Semantics module for NSCK V3 - FrameNet-inspired semantic frames."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import python.core.vsa.hypervec_shim as hypervec_rs

HyperVector = hypervec_rs.HyperVector


class Frame:
    def __init__(self, name: str, roles: Dict[str, str], evoking_words: List[str]):
        self.name = name
        self.roles = roles  # role_name -> description
        self.evoking_words = evoking_words
        self.name_hv = HyperVector(hash(name) % (2**32))
        self.role_hvs: Dict[str, HyperVector] = {
            role: HyperVector(hash(f"{name}:{role}") % (2**32))
            for role in roles
        }

    def fill(self, fillers: Dict[str, HyperVector]) -> HyperVector:
        """XOR-bind each role with its filler, then bundle all."""
        bound = None
        for role, filler_hv in fillers.items():
            if role in self.role_hvs:
                binding = self.role_hvs[role].xor(filler_hv)
                bound = binding if bound is None else bound.bundle(binding)
        if bound is None:
            return self.name_hv
        return self.name_hv.bundle(bound)

    def extract_filler(self, filled_hv: HyperVector, role: str) -> HyperVector:
        """Unbind a role HV to recover filler."""
        if role not in self.role_hvs:
            raise KeyError(f"Unknown role: {role}")
        return filled_hv.xor(self.role_hvs[role])

    def __repr__(self):
        return f"Frame({self.name}, roles={list(self.roles.keys())})"


class FrameLibrary:
    def __init__(self):
        self._frames: List[Frame] = []
        self._verb_index: Dict[str, Frame] = {}
        self._build_frames()

    def _build_frames(self):
        frame_defs = [
            ("COMMERCIAL_TRANSACTION", {"buyer":"entity buying","seller":"entity selling","goods":"item sold","money":"payment"}, ["buy","sell","purchase","trade","pay","cost","spend","acquire","rent","lease"]),
            ("MOTION", {"mover":"entity in motion","source":"origin","destination":"endpoint","path":"route"}, ["go","move","travel","walk","run","fly","drive","arrive","depart","leave","come","return","approach"]),
            ("COMMUNICATION", {"speaker":"communicator","message":"content","addressee":"recipient","channel":"medium"}, ["say","tell","speak","communicate","announce","report","explain","describe","mention","write","read","ask"]),
            ("CAUSATION", {"cause":"causal agent","effect":"result","manner":"how caused"}, ["cause","lead","result","trigger","produce","generate","create","induce","force","make","enable","allow"]),
            ("CATEGORIZATION", {"item":"entity categorized","category":"class","basis":"classification criterion"}, ["is","are","classify","categorize","group","type","kind","sort","class","define","identify"]),
            ("POSSESSION", {"owner":"possessor","owned":"possessed entity"}, ["have","own","possess","hold","keep","contain","include","belong"]),
            ("LOCATION", {"figure":"located entity","ground":"reference location","relation":"spatial relation"}, ["be","locate","place","position","reside","stay","sit","stand","lie","exist"]),
            ("CHANGE", {"entity":"thing changing","initial_state":"before state","final_state":"after state","cause":"change agent"}, ["change","transform","become","turn","convert","evolve","develop","grow","shrink","increase","decrease","alter","modify"]),
            ("PERCEPTION", {"perceiver":"entity perceiving","stimulus":"perceived entity","manner":"mode of perception"}, ["see","hear","feel","smell","taste","observe","notice","detect","sense","perceive","watch","listen"]),
            ("COGNITION", {"cognizer":"thinking entity","content":"thought content","manner":"cognitive mode"}, ["think","know","believe","understand","remember","forget","learn","realize","recognize","consider","assume","suppose"]),
            ("EMOTION", {"experiencer":"entity feeling","stimulus":"cause of emotion","emotion":"felt state"}, ["feel","love","hate","fear","enjoy","dislike","like","want","desire","hope","worry","trust"]),
            ("CREATION", {"creator":"making entity","creation":"made artifact","material":"used material","method":"creation method"}, ["make","build","create","construct","design","develop","write","paint","compose","produce","form","generate"]),
            ("DESTRUCTION", {"destroyer":"agent","patient":"destroyed entity","manner":"destruction method"}, ["destroy","break","damage","ruin","kill","end","eliminate","remove","delete","erase"]),
            ("TRANSFER", {"donor":"giver","recipient":"receiver","theme":"transferred entity"}, ["give","send","transfer","deliver","hand","pass","donate","award","assign","provide","supply"]),
            ("USE", {"agent":"user","instrument":"tool used","purpose":"goal of use"}, ["use","apply","employ","utilize","operate","deploy","implement","exercise"]),
            ("COMPARISON", {"standard":"reference entity","comparee":"compared entity","attribute":"compared quality"}, ["compare","contrast","similar","different","like","unlike","same","equal","greater","less","more","fewer"]),
            ("EXISTENCE", {"entity":"existing thing","location":"place of existence","time":"time of existence"}, ["exist","be","live","occur","happen","appear","emerge","arise","vanish","disappear"]),
            ("JUDGMENT", {"judge":"evaluating entity","evaluee":"evaluated entity","criterion":"evaluation standard"}, ["judge","evaluate","assess","rate","grade","criticize","praise","blame","approve","condemn"]),
            ("REQUEST", {"requester":"asking entity","addressee":"asked entity","content":"requested action"}, ["ask","request","demand","require","order","instruct","command","urge","beg","plead"]),
            ("PREVENTION", {"agent":"preventing entity","event":"prevented event","manner":"prevention method"}, ["prevent","stop","block","avoid","hinder","obstruct","impede","prohibit","forbid","restrict"]),
        ]
        for name, roles, words in frame_defs:
            f = Frame(name, roles, words)
            self._frames.append(f)
            for w in words:
                self._verb_index[w.lower()] = f

    def find_frame(self, verb: str) -> Optional[Frame]:
        return self._verb_index.get(verb.lower())

    def get_all_frames(self) -> List[Frame]:
        return list(self._frames)
