import unittest
from python.core.language.frame_semantics import Frame, FrameLibrary
import python.core.vsa.hypervec_shim as hypervec_rs

class TestFrameSemantics(unittest.TestCase):
    def setUp(self):
        self.library = FrameLibrary()

    def test_frame_fill(self):
        frame = self.library.find_frame("buy")
        self.assertIsNotNone(frame)
        fillers = {
            "buyer": hypervec_rs.HyperVector(hash("alice") % (2**32)),
            "goods": hypervec_rs.HyperVector(hash("book") % (2**32)),
        }
        filled = frame.fill(fillers)
        self.assertIsNotNone(filled)

    def test_frame_extraction_roundtrip(self):
        frame = self.library.find_frame("buy")
        buyer_hv = hypervec_rs.HyperVector(hash("alice") % (2**32))
        fillers = {"buyer": buyer_hv}
        filled = frame.fill(fillers)
        recovered = frame.extract_filler(filled, "buyer")
        # XOR-bind then XOR-unbind: should be similar to original
        sim = buyer_hv.similarity(recovered)
        self.assertGreater(sim, 0.0)

    def test_frame_library_lookup_by_verb(self):
        frame = self.library.find_frame("sell")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.name, "COMMERCIAL_TRANSACTION")

    def test_frame_library_motion(self):
        frame = self.library.find_frame("travel")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.name, "MOTION")

    def test_frame_library_unknown_verb(self):
        frame = self.library.find_frame("xyzzy_nonexistent")
        self.assertIsNone(frame)

    def test_get_all_frames(self):
        frames = self.library.get_all_frames()
        self.assertGreaterEqual(len(frames), 10)

if __name__ == "__main__":
    unittest.main()
