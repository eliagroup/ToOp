# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Exception hierarchy for the SpPS (Special Protection Scheme) engine."""


class SppsError(Exception):
    """Base class for every exception raised by the SpPS package.

    Catch :class:`SppsError` to cover any engine-specific failure without
    accidentally swallowing unrelated exceptions (e.g. ``KeyboardInterrupt``,
    ``ValueError`` from callers).
    """


class SppsPowerFlowError(SppsError):
    """The pandapower solver (``pp.runpp`` / ``pp.rundcpp``) failed.

    Raised by :func:`spps.engine.run_spps` when either:

    * the *initial* power flow (before the iteration loop) does not converge —
      in which case SpPS cannot run at all, or
    * an in-loop power flow fails and the caller asked for
      ``on_power_flow_error="raise"``.

    The underlying solver exception is attached as ``__cause__`` (via
    ``raise ... from exc``) so the original traceback is preserved.
    """
