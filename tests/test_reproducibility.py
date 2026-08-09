from __future__ import annotations

import unittest

from scripts import verify_frozen_results as verify


class FrozenResultTests(unittest.TestCase):
    def test_manifest_hashes(self) -> None:
        verify.verify_hashes()

    def test_simulation(self) -> None:
        verify.verify_simulation()

    def test_all_poisson(self) -> None:
        verify.verify_all_poisson()

    def test_nba(self) -> None:
        verify.verify_nba()


if __name__ == "__main__":
    unittest.main()
