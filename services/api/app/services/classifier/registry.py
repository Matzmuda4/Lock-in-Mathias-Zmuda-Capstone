"""
Module-level classifier registry.

Holds the active AbstractClassifier singleton and the shared ClassificationCache.
Set once during FastAPI lifespan startup; read from anywhere.

Why module-level rather than app.state?
  store.py and drift.py are pure-service modules — they have no access to the
  FastAPI Request object.  A module-level singleton is the standard pattern
  for services that need to be accessible without DI (same pattern as
  _upsert_counters in store.py).
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import AbstractClassifier
from .cache import ClassificationCache

log = logging.getLogger(__name__)

_classifier: Optional[AbstractClassifier] = None
_cache: ClassificationCache = ClassificationCache()


def set_classifier(clf: AbstractClassifier) -> None:
    global _classifier
    _classifier = clf
    log.info("Classifier registered: %s", type(clf).__name__)


def get_classifier() -> Optional[AbstractClassifier]:
    return _classifier


def get_cache() -> ClassificationCache:
    return _cache


def is_available() -> bool:
    return _classifier is not None
