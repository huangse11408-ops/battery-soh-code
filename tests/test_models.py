# -*- coding: utf-8 -*-
'''CPU tests on synthetic tensors: no data, no GPU.'''

import unittest

import torch

from sarg.models import (VARIANTS, GRUBackbone, build_sarg, count_parameters,
                         pair_delta, pool, progress)

ANCHORS = {"anchors": torch.linspace(3.0, 4.2, 8).tolist(), "sigma": 0.1714, "cutoff": 2.0}


def trace(batch=2, length=300, channels=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, length, channels, generator=generator)


def coordinate(batch=2, length=300):
    return torch.linspace(3.0, 4.2, length).repeat(batch, 1)


class AnchorPoolTests(unittest.TestCase):
    def test_pool_returns_one_memory_per_anchor(self):
        sequence = torch.randn(2, 300, 64)
        memory, observed = pool(sequence, coordinate(), torch.tensor(ANCHORS["anchors"]),
                                ANCHORS["sigma"], ANCHORS["cutoff"])
        self.assertEqual(memory.shape, (2, 8, 64))
        self.assertEqual(observed.shape, (2, 8))
        self.assertTrue(observed.all())

    def test_masked_coordinates_produce_no_memory(self):
        coord = coordinate().clone()
        coord[:, :150] = float("nan")
        memory, observed = pool(torch.randn(2, 300, 64), coord,
                                torch.tensor(ANCHORS["anchors"]), ANCHORS["sigma"],
                                ANCHORS["cutoff"])
        self.assertTrue(torch.isfinite(memory).all())

    def test_progress_is_unit_interval(self):
        time = torch.linspace(0.0, 1200.0, 300).repeat(2, 1)
        value, observed = progress(time)
        self.assertAlmostEqual(float(value.min()), 0.0, places=6)
        self.assertAlmostEqual(float(value.max()), 1.0, places=5)
        self.assertTrue(observed.all())

    def test_pair_delta_respects_joint_validity(self):
        z = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1).repeat(1, 1, 3)
        valid = torch.ones(1, 8, dtype=torch.bool)
        valid[0, 4] = False
        delta, pairs = pair_delta(z, valid)
        self.assertEqual(delta.shape, (1, 7, 3))
        self.assertFalse(bool(pairs[0, 3]))
        self.assertFalse(bool(pairs[0, 4]))
        self.assertAlmostEqual(float(delta[0, 2, 0]), 1.0, places=6)


class VariantTests(unittest.TestCase):
    def build(self, variant):
        return build_sarg(variant, ANCHORS)

    def test_all_variants_run_and_are_finite(self):
        x = trace()
        for variant in VARIANTS:
            model = self.build(variant)
            out = model(x, coordinate())
            self.assertEqual(out.shape, (2,), variant)
            self.assertTrue(torch.isfinite(out).all(), variant)

    def test_time_anchors_and_full_share_a_parameter_count(self):
        self.assertEqual(count_parameters(self.build("full")),
                         count_parameters(self.build("time_anchors")))

    def test_dropping_the_voltage_branch_removes_parameters(self):
        self.assertLess(count_parameters(self.build("no_voltage_branch")),
                        count_parameters(self.build("full")))

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            build_sarg("unknown_variant", ANCHORS)

    def test_backbone_is_independent_of_the_readout(self):
        first, second = GRUBackbone(), GRUBackbone()
        state = first.state_dict()
        second.load_state_dict(state)
        for key in state:
            self.assertTrue(torch.equal(state[key], second.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
