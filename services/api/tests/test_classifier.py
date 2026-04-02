"""
Tests for the RF attentional-state classifier pipeline.

Structure
---------
TestFeatureExtractor       — pure feature vector construction (no DB, no model)
TestFullWindowGate         — is_full_window() rule
TestClassificationResult   — ClassificationResult invariants
TestRFClassifierMocked     — RFClassifier with a patched joblib bundle
TestClassificationEndpoint — GET /classifier/health + /attentional-state (MockClassifier)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.classifier.base import ATTENTIONAL_STATES, ClassificationResult
from app.services.classifier.feature_extractor import (
    FEATURE_COLS,
    FULL_WINDOW_BATCHES,
    build_feature_vector,
    is_full_window,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_packet_json(n_batches: int = 16, override: dict | None = None) -> dict:
    """
    Minimal packet_json that matches the structure produced by store._build_packet_json().
    All values are at their neutral/baseline defaults.
    """
    base = {
        "session_id": 1,
        "user_id": 1,
        "features": {
            "n_batches": n_batches,
            "idle_ratio_mean": 0.1,
            "stagnation_ratio": 0.2,
            "regress_rate_mean": 0.03,
            "panel_interaction_share": 0.0,
            "progress_velocity": 0.002,
            "pace_available": True,
            "focus_loss_rate": 0.05,
        },
        "z_scores": {
            "z_idle": 0.5,
            "z_skim": 0.0,
            "z_regress": 1.2,
            "z_pace": 0.0,
            "z_pause": 0.8,
            "z_jitter": 0.3,
            "z_burstiness": 0.4,
            "z_focus_loss": 0.6,
            "z_stagnation": 0.7,
            "z_mouse": 0.0,
        },
        "drift": {
            "drift_level": 0.3,
            "drift_ema": 0.25,
            "pace_ratio": 1.1,
            "pace_available": True,
            "engagement_score": 0.6,
            "disruption_score": 0.4,
        },
    }
    if override:
        for key, val in override.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                base[key].update(val)  # type: ignore[union-attr]
            else:
                base[key] = val
    return base


# ── TestFeatureExtractor ──────────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_vector_length_is_19(self):
        pkt = _make_packet_json()
        vec = build_feature_vector(pkt)
        assert len(vec) == 19, f"Expected 19 features, got {len(vec)}"

    def test_feature_cols_count_matches_vector(self):
        assert len(FEATURE_COLS) == 19

    def test_all_values_are_float(self):
        vec = build_feature_vector(_make_packet_json())
        assert all(isinstance(v, float) for v in vec), "All features must be float"

    def test_z_scores_extracted_correctly(self):
        pkt = _make_packet_json(override={"z_scores": {"z_idle": 2.5, "z_skim": 1.8}})
        vec = build_feature_vector(pkt)
        idx_z_idle  = list(FEATURE_COLS).index("z_idle")
        idx_z_skim  = list(FEATURE_COLS).index("z_skim")
        assert vec[idx_z_idle] == pytest.approx(2.5)
        assert vec[idx_z_skim] == pytest.approx(1.8)

    def test_n_batches_norm_full_window(self):
        vec = build_feature_vector(_make_packet_json(n_batches=16))
        idx = list(FEATURE_COLS).index("n_batches_norm")
        assert vec[idx] == pytest.approx(1.0)

    def test_n_batches_norm_partial_window(self):
        vec = build_feature_vector(_make_packet_json(n_batches=8))
        idx = list(FEATURE_COLS).index("n_batches_norm")
        assert vec[idx] == pytest.approx(0.5)

    def test_n_batches_norm_capped_at_1(self):
        # n_batches > 16 should be capped at 1.0
        vec = build_feature_vector(_make_packet_json(n_batches=32))
        idx = list(FEATURE_COLS).index("n_batches_norm")
        assert vec[idx] == pytest.approx(1.0)

    def test_pace_available_true_maps_to_1(self):
        pkt = _make_packet_json(override={"features": {"pace_available": True}})
        vec = build_feature_vector(pkt)
        idx = list(FEATURE_COLS).index("pace_available")
        assert vec[idx] == pytest.approx(1.0)

    def test_pace_available_false_maps_to_0(self):
        pkt = _make_packet_json(
            override={"features": {"pace_available": False}, "drift": {"pace_available": False}}
        )
        vec = build_feature_vector(pkt)
        idx = list(FEATURE_COLS).index("pace_available")
        assert vec[idx] == pytest.approx(0.0)

    def test_missing_z_scores_default_to_zero(self):
        pkt = _make_packet_json()
        pkt["z_scores"] = {}  # strip all z-scores
        vec = build_feature_vector(pkt)
        for col in FEATURE_COLS:
            if col.startswith("z_"):
                assert vec[list(FEATURE_COLS).index(col)] == pytest.approx(0.0)

    def test_missing_features_block_uses_defaults(self):
        pkt = {"session_id": 1, "user_id": 1}  # no features, z_scores, or drift
        vec = build_feature_vector(pkt)
        assert len(vec) == 19
        assert all(isinstance(v, float) for v in vec)

    def test_pace_ratio_falls_back_to_features_when_drift_none(self):
        pkt = _make_packet_json()
        pkt["drift"]["pace_ratio"] = None
        pkt["features"]["pace_ratio"] = 1.75
        vec = build_feature_vector(pkt)
        idx = list(FEATURE_COLS).index("pace_ratio")
        assert vec[idx] == pytest.approx(1.75)

    def test_feature_col_order_is_stable(self):
        """Feature order must never change without retraining the model."""
        expected_first_five = ["z_idle", "z_skim", "z_regress", "z_pace", "z_pause"]
        assert list(FEATURE_COLS[:5]) == expected_first_five

    def test_last_four_are_context_features(self):
        expected = ["n_batches_norm", "progress_velocity", "pace_available", "focus_loss_rate"]
        assert list(FEATURE_COLS[15:]) == expected


# ── TestFullWindowGate ────────────────────────────────────────────────────────

class TestFullWindowGate:
    def test_full_window_at_exactly_14_batches(self):
        assert is_full_window(_make_packet_json(n_batches=14)) is True

    def test_full_window_at_15_batches(self):
        # Live sessions reach 15 at most — this must fire
        assert is_full_window(_make_packet_json(n_batches=15)) is True

    def test_full_window_above_16_batches(self):
        assert is_full_window(_make_packet_json(n_batches=20)) is True

    def test_not_full_window_at_13_batches(self):
        assert is_full_window(_make_packet_json(n_batches=13)) is False

    def test_not_full_window_at_1_batch(self):
        assert is_full_window(_make_packet_json(n_batches=1)) is False

    def test_not_full_window_when_features_missing(self):
        assert is_full_window({}) is False

    def test_full_window_batches_constant_is_14(self):
        # Gate = 14 (≥ 28 s signal, one batch of jitter tolerance).
        # Normalization cap = 16 (matches training notebook — do not change without retraining).
        assert FULL_WINDOW_BATCHES == 14


# ── TestClassificationResult ──────────────────────────────────────────────────

class TestClassificationResult:
    def _make_result(self, **kwargs) -> ClassificationResult:
        defaults = dict(
            focused=0.7, drifting=0.2, hyperfocused=0.05,
            cognitive_overload=0.05, primary_state="focused",
            rationale="test", latency_ms=5, parse_ok=True,
        )
        defaults.update(kwargs)
        return ClassificationResult(**defaults)

    def test_distribution_valid_sums_to_one(self):
        r = self._make_result()
        assert r.distribution_valid() is True

    def test_distribution_invalid_does_not_sum_to_one(self):
        r = self._make_result(focused=0.5, drifting=0.5, hyperfocused=0.5)
        assert r.distribution_valid() is False

    def test_distribution_invalid_negative_value(self):
        r = self._make_result(focused=-0.1, drifting=0.8, hyperfocused=0.2, cognitive_overload=0.1)
        assert r.distribution_valid() is False

    def test_as_dict_contains_all_keys(self):
        r = self._make_result()
        d = r.as_dict()
        for key in ("focused", "drifting", "hyperfocused", "cognitive_overload",
                    "primary_state", "rationale", "latency_ms", "parse_ok"):
            assert key in d

    def test_as_intervention_context_contains_required_keys(self):
        r = self._make_result()
        ctx = r.as_intervention_context()
        assert "primary_state" in ctx
        assert "confidence" in ctx
        assert "distribution" in ctx
        assert "ambiguous" in ctx

    def test_ambiguous_flag_set_when_top_two_close(self):
        # focused=0.45 and drifting=0.40 are within 0.15 → ambiguous
        r = self._make_result(
            focused=0.45, drifting=0.40,
            hyperfocused=0.10, cognitive_overload=0.05,
        )
        ctx = r.as_intervention_context()
        assert ctx["ambiguous"] is True

    def test_ambiguous_flag_clear_when_dominant(self):
        # focused=0.90 → gap > 0.15 → not ambiguous
        r = self._make_result(
            focused=0.90, drifting=0.05,
            hyperfocused=0.03, cognitive_overload=0.02,
        )
        ctx = r.as_intervention_context()
        assert ctx["ambiguous"] is False

    def test_attentional_states_tuple_has_four_elements(self):
        assert len(ATTENTIONAL_STATES) == 4
        assert "focused" in ATTENTIONAL_STATES
        assert "drifting" in ATTENTIONAL_STATES
        assert "hyperfocused" in ATTENTIONAL_STATES
        assert "cognitive_overload" in ATTENTIONAL_STATES


# ── TestRFClassifierMocked ────────────────────────────────────────────────────

class TestRFClassifierMocked:
    """
    Tests for RFClassifier without requiring the actual pkl file.
    The joblib.load call is patched to return a synthetic bundle.
    """

    def _make_mock_bundle(self, proba: list[float] | None = None) -> dict:
        """Build a fake joblib bundle with a mock CalibratedClassifierCV."""
        proba = proba or [0.70, 0.15, 0.10, 0.05]  # focused dominant

        mock_clf = MagicMock()
        import numpy as np
        mock_clf.predict_proba.return_value = np.array([proba])

        return {
            "calibrated_clf": mock_clf,
            "rf_base":        MagicMock(),
            "feature_cols":   list(FEATURE_COLS),
            "state_keys":     list(ATTENTIONAL_STATES),
            "best_params":    {"n_estimators": 300},
            "seed":           42,
            "n_train":        1086,
        }

    @pytest.mark.asyncio
    async def test_classify_returns_correct_primary_state(self):
        bundle = self._make_mock_bundle([0.10, 0.75, 0.10, 0.05])  # drifting dominant
        with patch("app.services.classifier.rf_classifier.joblib") as mock_joblib, \
             patch("app.services.classifier.rf_classifier._SKLEARN_AVAILABLE", True), \
             patch.object(__import__("pathlib").Path, "exists", return_value=True):
            mock_joblib.load.return_value = bundle
            from app.services.classifier.rf_classifier import RFClassifier
            clf = RFClassifier.__new__(RFClassifier)
            clf._model_path = __import__("pathlib").Path("fake.pkl")
            clf._clf = bundle["calibrated_clf"]
            clf._state_keys = bundle["state_keys"]
            clf._loaded = True

            result = await clf.classify(_make_packet_json())

        assert result.primary_state == "drifting"
        assert result.parse_ok is True

    @pytest.mark.asyncio
    async def test_classify_probabilities_sum_to_one(self):
        bundle = self._make_mock_bundle([0.50, 0.25, 0.15, 0.10])
        with patch.object(__import__("pathlib").Path, "exists", return_value=True):
            from app.services.classifier.rf_classifier import RFClassifier
            clf = RFClassifier.__new__(RFClassifier)
            clf._model_path = __import__("pathlib").Path("fake.pkl")
            clf._clf = bundle["calibrated_clf"]
            clf._state_keys = bundle["state_keys"]
            clf._loaded = True

            result = await clf.classify(_make_packet_json())

        total = result.focused + result.drifting + result.hyperfocused + result.cognitive_overload
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected 1.0"

    @pytest.mark.asyncio
    async def test_classify_returns_fallback_when_not_loaded(self):
        from app.services.classifier.rf_classifier import RFClassifier
        clf = RFClassifier.__new__(RFClassifier)
        clf._model_path = __import__("pathlib").Path("nonexistent.pkl")
        clf._clf = None
        clf._loaded = False

        result = await clf.classify(_make_packet_json())

        assert result.parse_ok is False
        assert "not loaded" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_loaded(self):
        bundle = self._make_mock_bundle()
        from app.services.classifier.rf_classifier import RFClassifier
        clf = RFClassifier.__new__(RFClassifier)
        clf._model_path = __import__("pathlib").Path("fake.pkl")
        clf._clf = bundle["calibrated_clf"]
        clf._state_keys = bundle["state_keys"]
        clf._loaded = True

        ok = await clf.health_check()
        assert ok is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_not_loaded(self):
        from app.services.classifier.rf_classifier import RFClassifier
        clf = RFClassifier.__new__(RFClassifier)
        clf._loaded = False
        clf._clf = None

        ok = await clf.health_check()
        assert ok is False


# ── TestClassificationEndpoint ────────────────────────────────────────────────

@pytest.mark.asyncio
class TestClassificationEndpoint:
    """
    Integration tests for the classification API endpoints.
    Uses MockClassifier (no pkl required) via CLASSIFY_USE_MOCK=true.
    """

    async def test_classifier_health_disabled(self, api_client):
        """When classifier is disabled (default test config), health returns available=False."""
        resp = await api_client.get("/classifier/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert "cache_size" in data

    async def test_attentional_state_503_when_classifier_disabled(
        self, api_client, auth_headers
    ):
        """503 is returned when the classifier is disabled."""
        doc_resp = await api_client.post(
            "/documents/upload",
            headers=auth_headers,
            files={"file": ("test.pdf", _MINIMAL_PDF, "application/pdf")},
            data={"title": "test"},
        )
        assert doc_resp.status_code == 201
        doc_id = doc_resp.json()["id"]

        sess_resp = await api_client.post(
            "/sessions/start",
            headers=auth_headers,
            json={"document_id": doc_id, "mode": "adaptive", "name": "test"},
        )
        assert sess_resp.status_code == 201
        session_id = sess_resp.json()["id"]

        resp = await api_client.get(
            f"/sessions/{session_id}/attentional-state",
            headers=auth_headers,
        )
        assert resp.status_code == 503

    async def test_attentional_state_404_unknown_session(self, api_client, auth_headers):
        """404 for a session_id that does not belong to the user."""
        resp = await api_client.get(
            "/sessions/99999/attentional-state",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_classifier_health_with_mock_enabled(self, api_client):
        """With MockClassifier active, health returns available=True."""
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import set_classifier, _cache
        set_classifier(MockClassifier())
        try:
            resp = await api_client.get("/classifier/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is True
            assert data["classifier_type"] == "MockClassifier"
        finally:
            # Restore disabled state for other tests
            import app.services.classifier.registry as reg
            reg._classifier = None

    async def test_attentional_state_404_when_no_classification_yet(
        self, api_client, auth_headers
    ):
        """With classifier enabled but no packets classified yet, returns 404."""
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import set_classifier
        import app.services.classifier.registry as reg

        set_classifier(MockClassifier())
        try:
            doc_resp = await api_client.post(
                "/documents/upload",
                headers=auth_headers,
                files={"file": ("test.pdf", _MINIMAL_PDF, "application/pdf")},
                data={"title": "test"},
            )
            doc_id = doc_resp.json()["id"]

            sess_resp = await api_client.post(
                "/sessions/start",
                headers=auth_headers,
                json={"document_id": doc_id, "mode": "adaptive", "name": "test"},
            )
            session_id = sess_resp.json()["id"]

            resp = await api_client.get(
                f"/sessions/{session_id}/attentional-state",
                headers=auth_headers,
            )
            assert resp.status_code == 404
            assert "30 seconds" in resp.json()["detail"]
        finally:
            reg._classifier = None

    async def test_attentional_state_200_after_cache_populated(
        self, api_client, auth_headers
    ):
        """After the cache is populated, the endpoint returns 200 with a valid distribution."""
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import get_cache, set_classifier
        import app.services.classifier.registry as reg

        set_classifier(MockClassifier())
        try:
            doc_resp = await api_client.post(
                "/documents/upload",
                headers=auth_headers,
                files={"file": ("test.pdf", _MINIMAL_PDF, "application/pdf")},
                data={"title": "test"},
            )
            doc_id = doc_resp.json()["id"]

            sess_resp = await api_client.post(
                "/sessions/start",
                headers=auth_headers,
                json={"document_id": doc_id, "mode": "adaptive", "name": "test"},
            )
            session_id = sess_resp.json()["id"]

            # Manually inject a mock result into the cache
            mock_result = ClassificationResult(
                focused=0.70, drifting=0.20,
                hyperfocused=0.05, cognitive_overload=0.05,
                primary_state="focused",
                rationale="[MockClassifier] test",
                latency_ms=0, parse_ok=True,
            )
            get_cache().put(session_id, packet_seq=1, result=mock_result)

            resp = await api_client.get(
                f"/sessions/{session_id}/attentional-state",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()

            # Distribution keys
            dist = data["distribution"]
            assert set(dist.keys()) == {"focused", "drifting", "hyperfocused", "cognitive_overload"}

            # Probabilities sum to ≈ 1.0
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-3, f"Distribution sums to {total}"

            # Primary state matches argmax
            assert data["primary_state"] == max(dist, key=dist.get)

            # Confidence is max probability
            assert abs(data["confidence"] - max(dist.values())) < 1e-4

            # Intervention context is present
            assert data["intervention_context"] is not None
            assert "primary_state" in data["intervention_context"]
            assert "ambiguous" in data["intervention_context"]
        finally:
            reg._classifier = None
            get_cache().evict(session_id)  # type: ignore[possibly-undefined]


# ── TestClassifierStore ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestClassifierStore:
    """
    Tests for the DB persistence layer (classifier_store.py).
    Requires a live test DB — uses the shared conftest fixtures.
    """

    async def _create_session(self, api_client, auth_headers) -> int:
        doc_resp = await api_client.post(
            "/documents/upload",
            headers=auth_headers,
            files={"file": ("test.pdf", _MINIMAL_PDF, "application/pdf")},
            data={"title": "test"},
        )
        assert doc_resp.status_code == 201
        doc_id = doc_resp.json()["id"]

        sess_resp = await api_client.post(
            "/sessions/start",
            headers=auth_headers,
            json={"document_id": doc_id, "mode": "adaptive", "name": "test"},
        )
        assert sess_resp.status_code == 201
        return sess_resp.json()["id"]

    async def test_save_attentional_state_writes_row(self):
        """Direct unit test: save_attentional_state inserts a row."""
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.pool import NullPool
        from sqlalchemy import select
        from app.services.classifier.classifier_store import save_attentional_state
        from app.db.models import SessionAttentionalState

        # We use the test DB directly for unit-testing the store function
        # without going through the full HTTP stack.
        # (Session FK constraint means we need a real session_id — skip this
        #  test if we can't easily create one without the full app context.
        #  Instead verify the intervention_context structure is correct.)

        result = ClassificationResult(
            focused=0.80, drifting=0.10,
            hyperfocused=0.05, cognitive_overload=0.05,
            primary_state="focused",
            rationale="test", latency_ms=3, parse_ok=True,
        )
        packet_json = _make_packet_json()
        packet_json["drift"] = {"drift_level": 0.25, "drift_ema": 0.20}

        # Verify the intervention_context shape without hitting DB
        from app.services.classifier.classifier_store import save_attentional_state
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.add = MagicMock()

        row = await save_attentional_state(
            session_id=1,
            packet_seq=3,
            result=result,
            packet_json=packet_json,
            db=mock_db,  # type: ignore[arg-type]
        )

        # Row attributes
        assert row.primary_state == "focused"
        assert abs(row.prob_focused - 0.80) < 1e-4
        assert abs(row.prob_drifting - 0.10) < 1e-4
        assert abs(row.drift_level - 0.25) < 1e-4
        assert abs(row.drift_ema - 0.20) < 1e-4
        assert row.packet_seq == 3
        assert row.parse_ok is True

        # intervention_context is fully populated
        ctx = row.intervention_context
        assert ctx["primary_state"] == "focused"
        assert "confidence" in ctx
        assert "distribution" in ctx
        assert "ambiguous" in ctx
        assert ctx["drift_level"] == pytest.approx(0.25)
        assert ctx["drift_ema"] == pytest.approx(0.20)
        assert ctx["packet_seq"] == 3
        assert ctx["session_id"] == 1

        # db.add was called once
        mock_db.add.assert_called_once_with(row)

    async def test_intervention_context_ambiguous_when_close_probabilities(self):
        """Ambiguous flag is set when top-2 states are within 0.15."""
        from unittest.mock import MagicMock
        from app.services.classifier.classifier_store import save_attentional_state

        result = ClassificationResult(
            focused=0.42, drifting=0.38,
            hyperfocused=0.12, cognitive_overload=0.08,
            primary_state="focused",
            rationale="test", latency_ms=2, parse_ok=True,
        )
        mock_db = MagicMock()
        row = await save_attentional_state(1, 0, result, _make_packet_json(), mock_db)  # type: ignore
        assert row.intervention_context["ambiguous"] is True

    async def test_history_endpoint_404_when_empty(self, api_client, auth_headers):
        """History returns 404 when no rows exist yet."""
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import set_classifier
        import app.services.classifier.registry as reg

        set_classifier(MockClassifier())
        try:
            session_id = await self._create_session(api_client, auth_headers)
            resp = await api_client.get(
                f"/sessions/{session_id}/attentional-state/history",
                headers=auth_headers,
            )
            assert resp.status_code == 404
        finally:
            reg._classifier = None

    async def test_history_endpoint_returns_rows_after_db_insert(
        self, api_client, auth_headers
    ):
        """After DB insertion, history returns structured records with intervention_context."""
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import set_classifier
        from app.services.classifier.classifier_store import save_attentional_state
        from app.db.session import get_db
        import app.services.classifier.registry as reg

        set_classifier(MockClassifier())
        try:
            session_id = await self._create_session(api_client, auth_headers)

            # Insert rows directly via the store function
            result = ClassificationResult(
                focused=0.70, drifting=0.20,
                hyperfocused=0.05, cognitive_overload=0.05,
                primary_state="focused",
                rationale="test", latency_ms=5, parse_ok=True,
            )
            pkt = _make_packet_json()
            pkt["drift"] = {"drift_level": 0.30, "drift_ema": 0.25}

            async for db in get_db():
                await save_attentional_state(session_id, 0, result, pkt, db)
                await db.commit()
                break

            # Second row — drifting
            result2 = ClassificationResult(
                focused=0.20, drifting=0.65,
                hyperfocused=0.10, cognitive_overload=0.05,
                primary_state="drifting",
                rationale="test2", latency_ms=4, parse_ok=True,
            )
            pkt2 = _make_packet_json()
            pkt2["drift"] = {"drift_level": 0.55, "drift_ema": 0.50}
            async for db in get_db():
                await save_attentional_state(session_id, 1, result2, pkt2, db)
                await db.commit()
                break

            resp = await api_client.get(
                f"/sessions/{session_id}/attentional-state/history?limit=10",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()

            # Structure
            assert data["session_id"] == session_id
            assert data["total_records"] == 2

            # Records are newest first
            records = data["records"]
            assert len(records) == 2
            assert records[0]["primary_state"] == "drifting"   # inserted last
            assert records[1]["primary_state"] == "focused"

            # Each record has a complete intervention_context
            for rec in records:
                ctx = rec["intervention_context"]
                assert "primary_state" in ctx
                assert "confidence" in ctx
                assert "distribution" in ctx
                assert "ambiguous" in ctx
                assert "drift_level" in ctx
                assert "drift_ema" in ctx
                assert "packet_seq" in ctx

            # state_counts aggregation
            assert data["state_counts"]["focused"] == 1
            assert data["state_counts"]["drifting"] == 1

            # sustained=False (two different states)
            assert data["sustained"] is False
            assert data["sustained_state"] is None

        finally:
            reg._classifier = None

    async def test_history_sustained_flag_when_all_same_state(
        self, api_client, auth_headers
    ):
        """sustained=True and sustained_state set when all records share one state."""
        from app.services.classifier.classifier_store import save_attentional_state
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import set_classifier
        from app.db.session import get_db
        import app.services.classifier.registry as reg

        set_classifier(MockClassifier())
        try:
            session_id = await self._create_session(api_client, auth_headers)

            for seq in range(3):
                result = ClassificationResult(
                    focused=0.15, drifting=0.70,
                    hyperfocused=0.10, cognitive_overload=0.05,
                    primary_state="drifting",
                    rationale="test", latency_ms=4, parse_ok=True,
                )
                async for db in get_db():
                    await save_attentional_state(session_id, seq, result, _make_packet_json(), db)
                    await db.commit()
                    break

            resp = await api_client.get(
                f"/sessions/{session_id}/attentional-state/history?limit=3",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["sustained"] is True
            assert data["sustained_state"] == "drifting"
        finally:
            reg._classifier = None

    async def test_history_limit_parameter(self, api_client, auth_headers):
        """limit parameter restricts number of returned records."""
        from app.services.classifier.classifier_store import save_attentional_state
        from app.services.classifier.mock import MockClassifier
        from app.services.classifier.registry import set_classifier
        from app.db.session import get_db
        import app.services.classifier.registry as reg

        set_classifier(MockClassifier())
        try:
            session_id = await self._create_session(api_client, auth_headers)

            for seq in range(5):
                result = ClassificationResult(
                    focused=0.80, drifting=0.10,
                    hyperfocused=0.05, cognitive_overload=0.05,
                    primary_state="focused",
                    rationale="test", latency_ms=3, parse_ok=True,
                )
                async for db in get_db():
                    await save_attentional_state(session_id, seq, result, _make_packet_json(), db)
                    await db.commit()
                    break

            resp = await api_client.get(
                f"/sessions/{session_id}/attentional-state/history?limit=3",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            assert resp.json()["total_records"] == 3
        finally:
            reg._classifier = None


# ── Minimal PDF bytes (shared with other test modules) ────────────────────────

_MINIMAL_PDF = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)
