# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_importer.ucte_toolset.ucte_io import interpret_ucte, make_ucte, parse_ucte, specs


def test_parse_ucte(ucte_file_exporter_test):
    with open(ucte_file_exporter_test, "r") as f:
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


def test_interpret_ucte(ucte_file_exporter_test):
    with open(ucte_file_exporter_test, "r") as f:
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
