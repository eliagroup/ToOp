# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Typed JSON column adapters for SQLModel models."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

from pydantic import TypeAdapter
from sqlalchemy import JSON
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.type_api import TypeDecorator

DecodedType = TypeVar("DecodedType")


class TypedJson(TypeDecorator[Any], Generic[DecodedType]):
	"""Persist and decode Pydantic-compatible payloads through a JSON column.

	The adapter owns both directions of the conversion:

	- ``process_bind_param`` serializes typed payloads into plain JSON data that
	  SQLAlchemy can pass to the database driver.
	- ``process_result_value`` validates the raw JSON back into the configured
	  Python type.

	The ``decode`` helper is exposed explicitly so callers can reuse the same
	validated decoding logic outside ORM result loading as well.
	"""

	impl = JSON
	cache_ok = True

	def __init__(self, decoded_type: Any) -> None:
		"""Create an adapter for the given decoded Python type."""
		super().__init__()
		self._decoded_type = decoded_type
		self._type_adapter = cast(TypeAdapter[Any], TypeAdapter(decoded_type))

	def encode(self, value: DecodedType | None) -> Any:
		"""Convert a typed value into a plain JSON-compatible payload."""
		if value is None:
			return None
		return self._type_adapter.dump_python(value, mode="json")

	def decode(self, value: Any) -> DecodedType | None:
		"""Decode plain JSON data into the configured Python type."""
		if value is None:
			return None
		return self._type_adapter.validate_python(value)

	def process_bind_param(self, value: DecodedType | None, dialect: Dialect) -> Any:
		"""Serialize a typed value before sending it to the database."""
		del dialect
		return self.encode(value)

	def process_result_value(self, value: Any, dialect: Dialect) -> DecodedType | None:
		"""Validate database JSON data into the configured Python type."""
		del dialect
		return self.decode(value)