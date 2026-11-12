"""A small stand-in for a PyMongo async database, for the offline Mongo repository tests.

It implements only what ``MongoRepository`` calls: ``replace_one`` (upsert), ``find_one``,
``find`` with ``sort``/``limit``/``to_list``, ``create_index`` and ``command``. Filters support
equality and ``$gt``/``$gte``/``$lt``/``$lte``. Like BSON it keeps datetimes to the
millisecond and returns them timezone-aware (UTC), and it copies documents both ways. It checks
the repository's logic, not the server: the ``-m live`` suite does that.
"""

import copy
from datetime import UTC, datetime
from typing import Any

from pymongo.errors import OperationFailure

_OPS = {
    "$gt": lambda a, b: a > b,
    "$gte": lambda a, b: a >= b,
    "$lt": lambda a, b: a < b,
    "$lte": lambda a, b: a <= b,
}


def bson_copy(value: Any) -> Any:
    """What a document looks like after a trip through BSON."""
    if isinstance(value, datetime):
        utc = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return utc.replace(microsecond=utc.microsecond // 1000 * 1000)
    if isinstance(value, dict):
        return {k: bson_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bson_copy(v) for v in value]
    return copy.copy(value)


def matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, wanted in query.items():
        have = doc.get(key)
        if isinstance(wanted, dict):
            if have is None or not all(_OPS[op](have, bound) for op, bound in wanted.items()):
                return False
        elif have != wanted:
            return False
    return True


class FakeCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def sort(self, keys: list[tuple[str, int]]) -> "FakeCursor":
        for key, direction in reversed(keys):  # stable sorts, last key first
            self._docs.sort(key=lambda d, k=key: d[k], reverse=direction < 0)
        return self

    def limit(self, count: int) -> "FakeCursor":
        self._docs = self._docs[:count]
        return self

    async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
        return [bson_copy(d) for d in self._docs]


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.docs: dict[Any, dict[str, Any]] = {}
        self.indexes: list[tuple[list[tuple[str, int]], dict[str, Any]]] = []
        self.index_conflict: int | None = None  # make create_index fail with this error code

    async def replace_one(
        self, query: dict[str, Any], doc: dict[str, Any], upsert: bool = False
    ) -> None:
        found = [key for key, d in self.docs.items() if matches(d, query)]
        if not found and not upsert:
            return
        self.docs[found[0] if found else doc["_id"]] = bson_copy(doc)

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        for doc in self.docs.values():
            if matches(doc, query):
                return bson_copy(doc)  # type: ignore[no-any-return]
        return None

    def find(self, query: dict[str, Any]) -> FakeCursor:
        return FakeCursor([d for d in self.docs.values() if matches(d, query)])

    async def create_index(self, keys: list[tuple[str, int]], **options: Any) -> str:
        if self.index_conflict is not None and "expireAfterSeconds" in options:
            code, self.index_conflict = self.index_conflict, None
            raise OperationFailure("index options conflict", code=code)
        self.indexes.append((keys, options))
        return "_".join(f"{k}_{d}" for k, d in keys)


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.commands: list[dict[str, Any]] = []

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection(name))

    async def command(self, command: dict[str, Any]) -> None:
        self.commands.append(command)
