"""Internal helpers shared across asset-topology model modules."""

from copy import deepcopy

from beartype.typing import Any, Optional
from pydantic import BaseModel


def merged_round_trip_payload(
    model: BaseModel,
    update: Optional[dict[str, Any]],
    *,
    deep: bool = False,
) -> dict[str, Any]:
    """Merge model field values and requested updates for revalidation-aware copies.

    Parameters
    ----------
    model : BaseModel
        Model instance that should be copied.
    update : Optional[dict[str, Any]]
        Field updates to merge into the copied payload.
    deep : bool, default=False
        Whether nested values should be deep-copied before validation.

    Returns
    -------
    dict[str, Any]
        Payload ready to be passed through ``model_validate``.
    """
    payload = {field_name: getattr(model, field_name) for field_name in type(model).model_fields}
    if deep:
        payload = deepcopy(payload)
    if update:
        payload.update(update)
    return payload
