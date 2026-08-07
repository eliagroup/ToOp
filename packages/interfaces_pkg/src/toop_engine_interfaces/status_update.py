# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The in-process progress reporting of the preprocessing routines.

This holds the callback that preprocessing routines invoke to report their progress, along with the
statistics they report next to a stage. Both are an internal interface between the routines and
whoever drives them, and the statistics only ever end up in the logs. They live here rather than in
a package of their own so that the importer and the DC solver can depend on them without depending
on each other.

The stages themselves are user-facing and are therefore defined alongside the heartbeat message in
toop_engine_interfaces.messages.preprocess.preprocess_heartbeat.
"""

import structlog
from beartype.typing import Optional, Protocol, TypeAlias
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import PreprocessStage

logger = structlog.get_logger(__name__)

NetworkDataStats: TypeAlias = dict[str, int]
# Size statistics of the grid at the time a preprocessing stage is entered, keyed by statistic name.


class StatusUpdateFn(Protocol):
    """The callback used to report progress through the preprocessing pipeline."""

    def __call__(
        self,
        stage: PreprocessStage,
        message: Optional[str],
        *,
        stats: Optional[NetworkDataStats] = None,
    ) -> None:
        """Report that a preprocessing stage was entered.

        Parameters
        ----------
        stage : PreprocessStage
            The stage that is being entered
        message : Optional[str]
            An optional message with more detail on the stage
        stats : Optional[NetworkDataStats]
            Size statistics of the network data as it looks when entering the stage, if available.
            Only stages that operate on network data report these.
        """
        ...


def empty_status_update_fn(
    stage: PreprocessStage, message: Optional[str], *, stats: Optional[NetworkDataStats] = None
) -> None:
    """Log an empty status update to logging.

    Use this function when no status_update_fn is provided.
    """
    stats_kwargs = {} if stats is None else {"network_stats": stats}
    if message is None:
        logger.info(f"Preprocessing stage {stage}", preprocess_stage=stage, **stats_kwargs)
    else:
        logger.info(f"Preprocessing stage {stage}, {message}", preprocess_stage=stage, message=message, **stats_kwargs)
