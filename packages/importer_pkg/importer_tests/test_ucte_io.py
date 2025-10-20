import os
from pathlib import Path

from toop_engine_importer.ucte_toolset.ucte_io import interpret_ucte, make_ucte, parse_ucte, specs


def test_parse_ucte():
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_uct_exporter_uct_file.uct")
    with open(ucte_file, "r") as f:
        contents = f.read()

    preamble, nodes, lines, trafo, trafo_reg, postamble = parse_ucte(contents)
    reconstructed = make_ucte(preamble, nodes, lines, trafo, trafo_reg, postamble)

    assert contents == reconstructed


def test_parse_pytest_ucte_file(ucte_file):
    with open(ucte_file, "r") as f:
        contents = f.read()

    preamble, nodes, lines, trafo, trafo_reg, postamble = parse_ucte(contents)
    reconstructed = make_ucte(preamble, nodes, lines, trafo, trafo_reg, postamble)

    assert contents == reconstructed


def test_interpret_ucte():
    ucte_file = Path(f"{os.path.dirname(__file__)}/files/test_uct_exporter_uct_file.uct")
    with open(ucte_file, "r") as f:
        contents = f.read()

    preamble, nodes, lines, trafo, trafo_reg, postamble = parse_ucte(contents)

    nodes, lines, trafos = interpret_ucte(nodes, lines, trafo, trafo_reg)

    for spec in specs["node"]:
        assert spec.name in nodes
        if spec.dtype != str:
            assert nodes[spec.name].dtype == spec.dtype

    for spec in specs["line"]:
        assert spec.name in lines
        if spec.dtype != str:
            assert lines[spec.name].dtype == spec.dtype

    for spec in specs["trafo"]:
        assert spec.name in trafos
        if spec.dtype != str:
            assert trafos[spec.name].dtype == spec.dtype

    for spec in specs["trafo_reg"]:
        assert spec.name in trafos
        if spec.dtype != str:
            assert trafos[spec.name].dtype == spec.dtype
