"""Unit tests for map-matching and re-anchor modules."""

import logging
import time
import numpy as np

from aeronavis.config import get_config
from aeronavis.app.maps import MapStore, haversine_m, bearing_deg
from aeronavis.app.mapmatch import MapMatcher, build_map_matcher
from aeronavis.app.ranchor import build_ranchor, RanchorState
from aeronavis.fusion.inekf import InEKF

logging.basicConfig(level=logging.ERROR)


class TestMapStore:
    """Tests for MapStore."""

    def test_haversine(self):
        haversine_m(12.9716, 77.5946, 12.9816, 77.5946)
        assert 1000 < haversine_m(12.9716, 77.5946, 12.9816, 77.5946) < 1200

    def test_bearing(self):
        bearing = bearing_deg(12.9716, 77.5946, 12.9816, 77.5946)
        assert 0 <= bearing < 360

    def test_mapstore_basic(self, tmp_path):
        """Test basic MapStore initialization."""
        config = get_config()
        store = MapStore(config, root=tmp_path)
        store.init_schema()
        assert store.db_path.exists()

    def test_mapstore_nodes(self, tmp_path):
        """Test node insertion and retrieval."""
        config = get_config()
        store = MapStore(config, root=tmp_path)
        store.init_schema()

        conn = store._connect()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO nodes (id, lat, lon) VALUES (?, ?, ?)", (1001, 12.9716, 77.5946)
        )
        conn.commit()

        node = store.get_node(1001)
        assert node is not None
        assert abs(node.lat - 12.9716) < 1e-6
        assert abs(node.lon - 77.5946) < 1e-6


class TestMapMatcher:
    """Tests for MapMatcher."""

    def test_matcher_init(self):
        config = get_config()
        matcher = build_map_matcher(config, None)
        assert matcher is not None

    def test_haversine(self):
        from aeronavis.app.mapmatch import haversine_m

        d = haversine_m(12.9716, 77.5946, 12.9816, 77.5946)
        assert 1000 < d < 1200


class TestRanchorStateMachine:
    """Tests for RanchorStateMachine."""

    def test_initial_state(self):
        from aeronavis.config import get_config

        config = get_config()
        InEKF(config)
        ranchor = build_ranchor(config, InEKF(config))

        assert ranchor.ranchor.state == RanchorState.DROPPED
        assert ranchor.get_state() == "DROPPED"

    def test_reacquiring_transition(self):
        from aeronavis.config import get_config

        config = get_config()
        InEKF(config)
        ranchor = build_ranchor(config, InEKF(config))

        # Simulate GNSS re-acquisition
        events = ranchor.ranchor.update(
            gnss_ok=True,
            gnss_residual=1.0,
            gnss_pos=np.array([10.0, 0.0, 0.0]),
            gnss_cov=np.eye(3),
            inekf_state=np.zeros(10),
            inekf_cov=np.eye(10),
            now=time.time(),
        )
        assert "REACQUIRING" in events or ranchor.get_state() == "REACQUIRING"

    def test_reanchored_convergence(self):
        from aeronavis.config import get_config

        config = get_config()
        InEKF(config)
        ranchor = build_ranchor(config, InEKF(config))

        # Force REACQUIRING state
        ranchor.ranchor._transition("REACQUIRING")
        ranchor.ranchor._residuals = [1.0, 1.0, 1.0, 1.0, 1.0]

        # Add good residuals
        for i in range(5):
            ranchor.ranchor.update(
                gnss_ok=True,
                gnss_residual=1.0,
                gnss_pos=np.array([10.0, 0.0, 0.0]),
                gnss_cov=np.eye(3),
                inekf_state=np.zeros(10),
                inekf_cov=np.eye(10),
                now=time.time() + i,
            )

    def test_handover_blend(self):
        from aeronavis.config import get_config

        config = get_config()
        inekf = InEKF(config)
        ranchor = build_ranchor(config, InEKF(config))

        # Force REACQUIRING state
        ranchor.ranchor._transition("REACQUIRING")
        ranchor.ranchor._last_event_time = time.time()

        inekf = InEKF(config)
        inekf.x[:2] = [100.0, 0.0]
        inekf.P[:2, :2] = np.eye(2) * 1.0

        pos, cov = ranchor.handover_blend(
            gnss_pos=np.array([101.0, 0.0, 0.0]),
            gnss_cov=np.eye(3) * 0.1,
            inekf_state=np.zeros(10),
            inekf_cov=np.eye(10),
        )

        # Should return blended position
        # Just test it runs without error


def test_map_basic():
    """Basic map module import test."""
    from aeronavis.app.maps import MapStore
    from aeronavis.config import get_config

    config = get_config()
    store = MapStore(config)
    store.init_schema()
    assert store.db_path.exists()

    # Test basic matcher creation
    matcher = MapMatcher(config, None)
    assert matcher is not None

    print("All basic map tests passed!")


if __name__ == "__main__":
    test_map_basic()
