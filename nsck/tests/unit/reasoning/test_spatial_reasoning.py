"""
Tests for NSCK Spatial Reasoning module (F2 from roadmap).
VSA-native 2D/3D spatial relation encoding and querying.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from python.core.reasoning.spatial_reasoning import (
    SpatialReasoner, PositionCodebook, SpatialRelationEncoder, SpatialPosition,
    _ADJACENT_MAX_DIST, _NEAR_DEFAULT_DIST,
)
import python.core.vsa.hypervec_shim as hypervec_rs


class TestPositionCodebook(unittest.TestCase):
    """VSA position encoding properties."""

    def setUp(self):
        self.cb = PositionCodebook()

    def test_encode_returns_hv(self):
        hv = self.cb.encode(1, 1)
        self.assertIsNotNone(hv)

    def test_same_position_same_hv(self):
        hv1 = self.cb.encode(3, 7)
        hv2 = self.cb.encode(3, 7)
        sim = float(hv1.similarity(hv2))
        self.assertAlmostEqual(sim, 1.0, places=3)

    def test_nearby_positions_more_similar_than_distant(self):
        hv_0_0 = self.cb.encode(0, 0)
        hv_1_0 = self.cb.encode(1, 0)
        hv_100_0 = self.cb.encode(100, 0)
        sim_near = float(hv_0_0.similarity(hv_1_0))
        sim_far = float(hv_0_0.similarity(hv_100_0))
        self.assertGreater(sim_near, sim_far,
            "Nearby positions should have higher HV similarity")

    def test_3d_encoding(self):
        hv = self.cb.encode(1, 2, 3)
        self.assertIsNotNone(hv)

    def test_z_axis_distinguishes_positions(self):
        hv_2d = self.cb.encode(0, 0, 0)
        hv_3d = self.cb.encode(0, 0, 5)
        sim = float(hv_2d.similarity(hv_3d))
        self.assertLess(sim, 1.0)  # Not identical

    def test_cache_works(self):
        hv1 = self.cb.encode(5, 5)
        hv2 = self.cb.encode(5, 5)
        self.assertIs(hv1, hv2)  # Same object (cached)


class TestSpatialPosition(unittest.TestCase):
    """SpatialPosition distance calculations."""

    def test_distance_2d(self):
        p1 = SpatialPosition(0, 0)
        p2 = SpatialPosition(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)

    def test_distance_zero(self):
        p1 = SpatialPosition(2, 3)
        p2 = SpatialPosition(2, 3)
        self.assertAlmostEqual(p1.distance_to(p2), 0.0)

    def test_distance_3d(self):
        p1 = SpatialPosition(0, 0, 0)
        p2 = SpatialPosition(1, 1, 1)
        import math
        self.assertAlmostEqual(p1.distance_to(p2), math.sqrt(3))


class TestSpatialRelationEncoder(unittest.TestCase):
    """VSA spatial relation encoding."""

    def setUp(self):
        self.enc = SpatialRelationEncoder()

    def test_role_hv_deterministic(self):
        hv1 = self.enc.role_hv("ABOVE")
        hv2 = self.enc.role_hv("ABOVE")
        self.assertAlmostEqual(float(hv1.similarity(hv2)), 1.0, places=3)

    def test_different_roles_differ(self):
        hv_above = self.enc.role_hv("ABOVE")
        hv_below = self.enc.role_hv("BELOW")
        sim = float(hv_above.similarity(hv_below))
        self.assertLess(sim, 0.9)  # Different roles → different HVs

    def test_encode_assertion_returns_hv(self):
        hv_cat = hypervec_rs.HyperVector(abs(hash("cat")) % (2**32))
        hv_table = hypervec_rs.HyperVector(abs(hash("table")) % (2**32))
        result = self.enc.encode_assertion(hv_cat, hv_table, "ABOVE")
        self.assertIsNotNone(result)


class TestSpatialReasonerBasic(unittest.TestCase):
    """Basic SpatialReasoner operations."""

    def setUp(self):
        self.sr = SpatialReasoner()
        # Classic 2D scene: cat above table, dog to the right of table
        self.sr.place("cat",   x=0, y=3)
        self.sr.place("table", x=0, y=0)
        self.sr.place("dog",   x=4, y=0)
        self.sr.place("box",   x=0, y=-3)

    def test_place_registers_entity(self):
        self.assertIn("cat", self.sr.list_entities())
        self.assertIn("table", self.sr.list_entities())

    def test_where_is_returns_position(self):
        pos = self.sr.where_is("cat")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.x, 0.0)
        self.assertAlmostEqual(pos.y, 3.0)

    def test_where_is_unknown_returns_none(self):
        self.assertIsNone(self.sr.where_is("unicorn"))

    def test_above_relation(self):
        rel = self.sr.get_relation("cat", "table")
        self.assertEqual(rel, "above", f"Expected 'above', got '{rel}'")

    def test_below_relation(self):
        rel = self.sr.get_relation("box", "table")
        self.assertEqual(rel, "below", f"Expected 'below', got '{rel}'")

    def test_right_of_relation(self):
        rel = self.sr.get_relation("dog", "table")
        self.assertEqual(rel, "right_of", f"Expected 'right_of', got '{rel}'")

    def test_left_of_relation(self):
        self.sr.place("plant", x=-4, y=0)
        rel = self.sr.get_relation("plant", "table")
        self.assertEqual(rel, "left_of")

    def test_adjacent_relation(self):
        self.sr.place("cup", x=0, y=1)  # distance 1 from table at (0,0)
        rel = self.sr.get_relation("cup", "table")
        self.assertEqual(rel, "adjacent_to")

    def test_unknown_entity_returns_none(self):
        self.assertIsNone(self.sr.get_relation("unicorn", "table"))

    def test_same_position_relation(self):
        self.sr.place("ghost", x=0, y=0)  # same as table
        rel = self.sr.get_relation("ghost", "table")
        self.assertEqual(rel, "at_same_position")


class TestSpatialReasonerMetric(unittest.TestCase):
    """Distance and find_near operations."""

    def setUp(self):
        self.sr = SpatialReasoner()
        self.sr.place("origin", x=0, y=0)
        self.sr.place("near1",  x=1, y=0)
        self.sr.place("near2",  x=0, y=2)
        self.sr.place("far",    x=10, y=10)

    def test_distance_correct(self):
        d = self.sr.distance("origin", "near1")
        self.assertAlmostEqual(d, 1.0)

    def test_distance_symmetric(self):
        d1 = self.sr.distance("origin", "far")
        d2 = self.sr.distance("far", "origin")
        self.assertAlmostEqual(d1, d2)

    def test_distance_unknown_returns_none(self):
        self.assertIsNone(self.sr.distance("unicorn", "origin"))

    def test_find_near_default_radius(self):
        near = self.sr.find_near("origin", radius=_NEAR_DEFAULT_DIST)
        self.assertIn("near1", near)
        self.assertIn("near2", near)
        self.assertNotIn("far", near)

    def test_find_near_excludes_anchor(self):
        near = self.sr.find_near("origin")
        self.assertNotIn("origin", near)

    def test_find_near_unknown_anchor_returns_empty(self):
        self.assertEqual(self.sr.find_near("unicorn"), [])

    def test_find_near_small_radius(self):
        near = self.sr.find_near("origin", radius=0.5)
        self.assertEqual(near, [])


class TestSpatialReasonerDirectional(unittest.TestCase):
    """find_above / find_below / find_left_of / find_right_of."""

    def setUp(self):
        self.sr = SpatialReasoner()
        self.sr.place("center", x=0, y=0)
        self.sr.place("north",  x=0, y=5)
        self.sr.place("south",  x=0, y=-5)
        self.sr.place("east",   x=5, y=0)
        self.sr.place("west",   x=-5, y=0)

    def test_find_above(self):
        above = self.sr.find_above("center")
        self.assertIn("north", above)
        self.assertNotIn("south", above)
        self.assertNotIn("east", above)

    def test_find_below(self):
        below = self.sr.find_below("center")
        self.assertIn("south", below)
        self.assertNotIn("north", below)

    def test_find_right_of(self):
        right = self.sr.find_right_of("center")
        self.assertIn("east", right)
        self.assertNotIn("west", right)

    def test_find_left_of(self):
        left = self.sr.find_left_of("center")
        self.assertIn("west", left)
        self.assertNotIn("east", left)


class TestSpatialReasonerVSA(unittest.TestCase):
    """VSA-based position similarity and assertion encoding."""

    def setUp(self):
        self.sr = SpatialReasoner()
        self.sr.place("cat",   x=0, y=3)
        self.sr.place("table", x=0, y=0)
        self.sr.place("far",   x=100, y=100)

    def test_position_similarity_same(self):
        self.sr.place("doppelganger", x=0, y=3)
        sim = self.sr.position_similarity("cat", "doppelganger")
        self.assertGreater(sim, 0.9)

    def test_position_similarity_nearby_vs_distant(self):
        self.sr.place("close", x=0, y=4)
        sim_near = self.sr.position_similarity("cat", "close")
        sim_far  = self.sr.position_similarity("cat", "far")
        self.assertGreater(sim_near, sim_far)

    def test_position_similarity_unknown_returns_none(self):
        self.assertIsNone(self.sr.position_similarity("unicorn", "cat"))

    def test_encode_scene_assertion_returns_hv(self):
        hv = self.sr.encode_scene_assertion("cat", "ABOVE", "table")
        self.assertIsNotNone(hv)

    def test_encode_scene_assertion_unknown_returns_none(self):
        hv = self.sr.encode_scene_assertion("unicorn", "ABOVE", "table")
        self.assertIsNone(hv)


class TestSpatialReasonerScene(unittest.TestCase):
    """scene_summary and move operations."""

    def test_scene_summary_empty(self):
        sr = SpatialReasoner()
        self.assertEqual(sr.scene_summary(), "Empty scene.")

    def test_scene_summary_nonempty(self):
        sr = SpatialReasoner()
        sr.place("cat", x=1, y=2)
        sr.place("dog", x=3, y=4)
        summary = sr.scene_summary()
        self.assertIn("cat", summary)
        self.assertIn("dog", summary)

    def test_move_updates_position(self):
        sr = SpatialReasoner()
        sr.place("entity", x=0, y=0)
        sr.move("entity", x=5, y=5)
        pos = sr.where_is("entity")
        self.assertAlmostEqual(pos.x, 5.0)
        self.assertAlmostEqual(pos.y, 5.0)

    def test_list_entities(self):
        sr = SpatialReasoner()
        sr.place("a", x=0, y=0)
        sr.place("b", x=1, y=1)
        entities = sr.list_entities()
        self.assertIn("a", entities)
        self.assertIn("b", entities)


class TestSpatialReasonerAllRelations(unittest.TestCase):
    """get_all_relations returns a list."""

    def test_all_relations_non_empty(self):
        sr = SpatialReasoner()
        sr.place("cat",   x=0, y=5)
        sr.place("table", x=0, y=0)
        rels = sr.get_all_relations("cat", "table")
        self.assertIsInstance(rels, list)
        self.assertIn("above", rels)

    def test_all_relations_near(self):
        sr = SpatialReasoner()
        sr.place("a", x=0, y=0)
        sr.place("b", x=1, y=0)
        rels = sr.get_all_relations("a", "b")
        self.assertIn("near", rels)

    def test_all_relations_far(self):
        sr = SpatialReasoner()
        sr.place("a", x=0, y=0)
        sr.place("b", x=50, y=50)
        rels = sr.get_all_relations("a", "b")
        self.assertIn("far", rels)


if __name__ == "__main__":
    unittest.main()
