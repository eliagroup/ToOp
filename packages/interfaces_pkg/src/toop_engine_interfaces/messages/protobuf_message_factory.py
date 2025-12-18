# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module: protobuf_message_factory

This module provides utility functions for serializing and deserializing messages using the
MessageWrapper protobuf schema. It enables conversion between string messages and their
protobuf-encoded byte representations for efficient message exchange.

Functions
---------
- serialize_message(message: str) -> bytes
    Serializes a string message into bytes using the MessageWrapper protobuf schema.

- deserialize_message(msg_bytes: bytes) -> str
    Deserializes bytes into a string message using the MessageWrapper protobuf schema.
"""

from toop_engine_interfaces.messages.protobuf_schema.message_wrapper_pb2 import MessageWrapper


def serialize_message(message: str) -> bytes:
    """
    Serialize a message string into bytes using MessageWrapper.

    Parameters
    ----------
    message : str
        The message to be serialized.

    Returns
    -------
    bytes
        The serialized message as a bytes object.
    """
    mw = MessageWrapper(message=message)
    return mw.SerializeToString()


def deserialize_message(msg_bytes: bytes) -> str:
    """
    Deserializes a protobuf message from bytes and returns its string representation.

    Parameters
    ----------
    msg_bytes : bytes
        The serialized protobuf message as a byte string.

    Returns
    -------
    str
        The deserialized message as a string.
    """
    mw = MessageWrapper()
    mw.ParseFromString(msg_bytes)
    return mw.message
