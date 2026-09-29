"""Sequence recovery must be exact or absent; a wrong sequence is worse than none."""
import pytest

pytest.importorskip("rdkit")
from aster.real.peptide_sequence import (AA_SMILES, CHAIN, build_peptide,
                                        sequence_from_smiles, side_chain_table)


def test_every_standard_residue_has_a_distinct_side_chain_key():
    table = side_chain_table()
    assert len(table) == 20
    assert set(table.values()) == set(AA_SMILES)
    assert set(CHAIN) == set(AA_SMILES)


@pytest.mark.parametrize("sequence", [
    "GAV", "PVP", "LIL", "ILV",  # the pairs that collide without the dummy atom
    "ACDEFGHIKLMNPQRSTVWY", "K", "GG", "PPP", "WYF", "TTSS",
])
def test_round_trip_recovers_the_sequence_it_built(sequence):
    assert sequence_from_smiles(build_peptide(sequence)) == sequence


def test_proline_and_valine_are_not_confused():
    # Both side chains are three carbons; only the attachment point separates them.
    assert sequence_from_smiles(build_peptide("AVA")) == "AVA"
    assert sequence_from_smiles(build_peptide("APA")) == "APA"
    assert build_peptide("AVA") != build_peptide("APA")


def test_leucine_and_isoleucine_are_not_confused():
    assert sequence_from_smiles(build_peptide("ALA")) == "ALA"
    assert sequence_from_smiles(build_peptide("AIA")) == "AIA"
    assert build_peptide("ALA") != build_peptide("AIA")


def test_modified_peptides_are_refused_rather_than_flattened():
    # Cyclic, N-methylated and D-substituted analogues have no L sequence.
    for smiles in ["O=C1NCC(=O)NCC(=O)NC1",                      # cyclic tripeptide
                   "CN(C(=O)[C@@H](N)C)[C@@H](C)C(=O)O",          # N-methylated
                   "N[C@H](C)C(=O)N[C@@H](C)C(=O)O",              # leading D-alanine
                   "N[C@@H](CCl)C(=O)N[C@@H](C)C(=O)O"]:          # nonstandard side chain
        assert sequence_from_smiles(smiles) is None


def test_non_peptides_and_junk_return_none():
    for smiles in ["", "c1ccccc1", "CCO", "not-a-smiles", "[Na+].[Cl-]"]:
        assert sequence_from_smiles(smiles) is None


def test_recovery_is_stereochemically_strict():
    # The all-L build is accepted; inverting one centre must not still pass.
    good = build_peptide("AFA")
    assert sequence_from_smiles(good) == "AFA"
    inverted = good.replace("[C@@H]", "[C@H]", 1)
    assert sequence_from_smiles(inverted) != "AFA"
