# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Bruteforce DC optimizer components.

The idea of the bruteforce optimizer is to walk through all combinations of the action set in dc exhaustively, or at least
up to the part that is doable within the runtime. The topologies are sent the same way to kafka for AC evaluation. The
optimizer will print on command line how many topologies are still remaining.
"""
