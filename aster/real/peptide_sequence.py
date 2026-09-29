"""Read an amino-acid sequence back out of a peptide SMILES, or refuse to.

Three PeptiVerse assays (toxicity, caco2, pampa) ship structures without any
sequence column, which leaves a protein language model with nothing to encode.
Where the structure really is a plain linear peptide of the twenty standard
L-residues, the sequence is recoverable; where it is a macrocycle, an
N-methylated backbone or a D-substituted analogue, it is not, and guessing the
nearest L sequence would feed the encoder a molecule that does not exist.

So acceptance is proved, not assumed: the recovered letters are rebuilt into a
linear all-L peptide and the canonical SMILES of that rebuild must equal the
canonical SMILES of the input. Anything else returns None. On the pinned
revision this recovers 42% of toxicity with zero disagreements against the
4,250 hemolysis rows that publish a sequence, and correctly recovers nothing
from caco2 and pampa, which are 100% macrocyclic.
"""
from __future__ import annotations

from functools import lru_cache

# Free-acid reference forms; side-chain keys are derived from these.
AA_SMILES = {
    "A": "N[C@@H](C)C(=O)O", "R": "N[C@@H](CCCNC(=N)N)C(=O)O",
    "N": "N[C@@H](CC(=O)N)C(=O)O", "D": "N[C@@H](CC(=O)O)C(=O)O",
    "C": "N[C@@H](CS)C(=O)O", "Q": "N[C@@H](CCC(=O)N)C(=O)O",
    "E": "N[C@@H](CCC(=O)O)C(=O)O", "G": "NCC(=O)O",
    "H": "N[C@@H](Cc1c[nH]cn1)C(=O)O", "I": "N[C@@H]([C@@H](C)CC)C(=O)O",
    "L": "N[C@@H](CC(C)C)C(=O)O", "K": "N[C@@H](CCCCN)C(=O)O",
    "M": "N[C@@H](CCSC)C(=O)O", "F": "N[C@@H](Cc1ccccc1)C(=O)O",
    "P": "OC(=O)[C@@H]1CCCN1", "S": "N[C@@H](CO)C(=O)O",
    "T": "N[C@@H]([C@H](C)O)C(=O)O", "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O", "V": "N[C@@H](C(C)C)C(=O)O",
}
# N-to-C chaining forms. Every ring closure opens and closes inside one residue,
# so the next residue can safely reuse the same ring-closure digit.
CHAIN = {
    "A": "N[C@@H](C)C(=O)", "R": "N[C@@H](CCCNC(=N)N)C(=O)",
    "N": "N[C@@H](CC(N)=O)C(=O)", "D": "N[C@@H](CC(=O)O)C(=O)",
    "C": "N[C@@H](CS)C(=O)", "Q": "N[C@@H](CCC(N)=O)C(=O)",
    "E": "N[C@@H](CCC(=O)O)C(=O)", "G": "NCC(=O)",
    "H": "N[C@@H](Cc1c[nH]cn1)C(=O)", "I": "N[C@@H]([C@@H](C)CC)C(=O)",
    "L": "N[C@@H](CC(C)C)C(=O)", "K": "N[C@@H](CCCCN)C(=O)",
    "M": "N[C@@H](CCSC)C(=O)", "F": "N[C@@H](Cc1ccccc1)C(=O)",
    "P": "N1CCC[C@H]1C(=O)", "S": "N[C@@H](CO)C(=O)",
    "T": "N[C@@H]([C@H](C)O)C(=O)", "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)", "V": "N[C@@H](C(C)C)C(=O)",
}
RESIDUE_SMARTS = "[NX3;H0,H1,H2][CX4;H1,H2][CX3](=[OX1])"


@lru_cache(maxsize=1)
def _residue_pattern():
    from rdkit import Chem
    return Chem.MolFromSmarts(RESIDUE_SMARTS)


def _side_chain_key(mol, n, ca, c):
    """Side chain keyed with the alpha carbon as a dummy atom.

    The dummy matters: drop it and the attachment point is lost, so valine's
    isopropyl and proline's ring both read as propane, and leucine and
    isoleucine both read as 2-methylbutane.
    """
    from rdkit import Chem
    keep, stack, seen = set(), [], {n, ca, c}
    for atom in mol.GetAtomWithIdx(ca).GetNeighbors():
        if atom.GetIdx() not in (n, c):
            stack.append(atom.GetIdx())
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        keep.add(index)
        for neighbour in mol.GetAtomWithIdx(index).GetNeighbors():
            j = neighbour.GetIdx()
            if j not in seen and j != n:  # proline's ring closes on the backbone N
                stack.append(j)
    editable = Chem.RWMol(mol)
    dummy = editable.GetAtomWithIdx(ca)
    dummy.SetAtomicNum(0)
    dummy.SetNoImplicit(True)
    dummy.SetNumExplicitHs(0)
    dummy.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolFragmentToSmiles(editable, atomsToUse=[ca] + sorted(keep),
                                    canonical=True)


@lru_cache(maxsize=1)
def side_chain_table() -> dict:
    """Side-chain key -> one-letter code, for the twenty standard residues."""
    from rdkit import Chem
    table = {}
    for letter, smiles in AA_SMILES.items():
        mol = Chem.MolFromSmiles(smiles)
        match = mol.GetSubstructMatch(_residue_pattern())
        if not match:
            raise ValueError(f"reference residue {letter} did not match the backbone pattern")
        n, ca, c, _ = match
        key = _side_chain_key(mol, n, ca, c)
        if key in table:
            raise ValueError(f"side-chain collision: {letter} vs {table[key]}")
        table[key] = letter
    return table


def build_peptide(sequence: str) -> str | None:
    """Canonical SMILES of the linear all-L peptide with a free C-terminus."""
    from rdkit import Chem, rdBase
    if not sequence or any(letter not in CHAIN for letter in sequence):
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles("".join(CHAIN[letter] for letter in sequence) + "O")
    return Chem.MolToSmiles(mol) if mol is not None else None


def _read_letters(mol) -> str | None:
    table = side_chain_table()
    matches = mol.GetSubstructMatches(_residue_pattern())
    if not matches:
        return None
    by_alpha = {}
    for n, ca, c, _ in matches:
        by_alpha.setdefault(ca, (n, ca, c))
    residues = list(by_alpha.values())
    nitrogen = {n: ca for n, ca, c in residues}
    following, preceding = {}, {}
    for n, ca, c in residues:
        for neighbour in mol.GetAtomWithIdx(c).GetNeighbors():
            j = neighbour.GetIdx()
            if j in nitrogen and j != n:
                following[ca] = nitrogen[j]
                preceding[nitrogen[j]] = ca
    starts = [ca for _, ca, _ in residues if ca not in preceding]
    if len(starts) != 1:
        return None  # branched, cyclic, or more than one chain
    order, seen, node = [], set(), starts[0]
    while node is not None and node not in seen:
        seen.add(node)
        order.append(node)
        node = following.get(node)
    if len(order) != len(residues):
        return None
    backbone = {ca: (n, c) for n, ca, c in residues}
    letters = []
    for ca in order:
        n, c = backbone[ca]
        letter = table.get(_side_chain_key(mol, n, ca, c))
        if letter is None:
            return None  # nonstandard residue
        letters.append(letter)
    return "".join(letters)


@lru_cache(maxsize=200000)
def sequence_from_smiles(smiles: str) -> str | None:
    """Recovered sequence, or None when the structure is not a plain L-peptide.

    Verified by reconstruction, so a returned sequence describes exactly the
    molecule that was passed in.
    """
    from rdkit import Chem, rdBase
    if not smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        letters = _read_letters(mol)
        if letters is None:
            return None
        return letters if build_peptide(letters) == Chem.MolToSmiles(mol) else None
