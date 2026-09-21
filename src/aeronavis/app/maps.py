"""Offline OSM road graph store for NAV-X 3.0.

Downloads, parses, and caches OSM road network as a routable SQLite database.
No network access at runtime; all data is pre-fetched and cached.
"""

import logging
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from rtree import index

from aeronavis.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class MapNode:
    """OSM road network node."""

    id: int
    lat: float
    lon: float


@dataclass
class MapWay:
    """OSM way (road segment)."""

    id: int
    node_ids: List[int]
    tags: Dict[str, str]
    one_way: bool


@dataclass
class MapSegment:
    """Road segment between two nodes."""

    id: int
    n1: int
    n2: int
    length_m: float
    bearing_deg: float
    way_id: int
    one_way: bool


@dataclass
class TunnelPOI:
    """Tunnel point of interest for re-anchoring."""

    id: int
    name: str
    lat: float
    lon: float
    way_id: int


class MapStore:
    """Offline OSM road graph store with SQLite persistence."""

    def __init__(self, config=None, root: Optional[Path] = None):
        self.config = config or get_config()
        self.maps_dir = root or Path(self.config.paths.maps)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.maps_dir / "road_graph.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._rtree_idx: Optional[index.Index] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._rtree_idx:
            self._rtree_idx.close()
            self._rtree_idx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def is_ready(self) -> bool:
        """Check if map database exists and has data."""
        return self.db_path.exists()

    def init_schema(self):
        """Create database schema."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                lat REAL NOT NULL,
                lon REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ways (
                id INTEGER PRIMARY KEY,
                node_seq TEXT NOT NULL,  -- JSON array of node IDs
                tags TEXT NOT NULL,      -- JSON object
                one_way BOOLEAN DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY,
                n1 INTEGER NOT NULL,
                n2 INTEGER NOT NULL,
                length_m REAL NOT NULL,
                bearing_deg REAL NOT NULL,
                way_id INTEGER NOT NULL,
                one_way BOOLEAN DEFAULT 0,
                FOREIGN KEY (n1) REFERENCES nodes(id),
                FOREIGN KEY (n2) REFERENCES nodes(id),
                FOREIGN KEY (way_id) REFERENCES ways(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tunnels (
                id INTEGER PRIMARY KEY,
                name TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                way_id INTEGER,
                FOREIGN KEY (way_id) REFERENCES ways(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # R-tree shadow table for spatial queries
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_rtree
            USING rtree(id, minx, maxx, miny, maxy)
        """)

        conn.commit()
        logger.info("Map schema initialized")

    def build_rtree(self):
        """Build R-tree index for spatial queries on segments."""
        if self._rtree_idx:
            self._rtree_idx.close()

        self._rtree_idx = index.Index()
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT s.id, n1.lat, n1.lon, n2.lat, n2.lon
            FROM segments s
            JOIN nodes n1 ON s.n1 = n1.id
            JOIN nodes n2 ON s.n2 = n2.id
        """)

        for row in cursor:
            sid, lat1, lon1, lat2, lon2 = row
            minx = min(lon1, lon2)
            maxx = max(lon1, lon2)
            miny = min(lat1, lat2)
            maxy = max(lat1, lat2)
            self._rtree_idx.insert(sid, (minx, maxx, miny, maxy))

        logger.info(f"R-tree built with {self._rtree_idx.count()} segments")

    def query_segments_nearby(self, lat: float, lon: float, radius_m: float) -> List[int]:
        """Find segment IDs within radius_m of (lat, lon)."""
        if not self._rtree_idx:
            self.build_rtree()

        # Convert radius_m to approximate degrees
        deg_per_m = 1.0 / 111000.0  # rough approximation
        radius_deg = radius_m * deg_per_m

        minx = lon - radius_deg
        maxx = lon + radius_deg
        miny = lat - radius_deg
        maxy = lat + radius_deg

        return list(self._rtree_idx.intersection((minx, maxx, miny, maxy)))

    def get_segment(self, segment_id: int) -> Optional[MapSegment]:
        """Get segment by ID with node coordinates."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.n1, s.n2, s.length_m, s.bearing_deg,
                   s.way_id, s.one_way,
                   n1.lat as lat1, n1.lon as lon1,
                   n2.lat as lat2, n2.lon as lon2
            FROM segments s
            JOIN nodes n1 ON s.n1 = n1.id
            JOIN nodes n2 ON s.n2 = n2.id
            WHERE s.id = ?
        """,
            (segment_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return MapSegment(
            id=row["id"],
            n1=row["n1"],
            n2=row["n2"],
            length_m=row["length_m"],
            bearing_deg=row["bearing_deg"],
            way_id=row["way_id"],
            one_way=bool(row["one_way"]),
        )

    def get_node(self, node_id: int) -> Optional[MapNode]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, lat, lon FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return MapNode(id=row["id"], lat=row["lat"], lon=row["lon"])

    def get_tunnels(self) -> List[TunnelPOI]:
        """Get all tunnel POIs."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, lat, lon, way_id FROM tunnels")
        return [
            TunnelPOI(
                id=row["id"], name=row["name"], lat=row["lat"], lon=row["lon"], way_id=row["way_id"]
            )
            for row in cursor
        ]

    def count_segments(self) -> int:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM segments")
        return cursor.fetchone()[0]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters."""
    R = 6371000.0
    dlat = np.deg2rad(lat2 - lat1)
    np.deg2rad(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.deg2rad(lat1))
        * np.cos(np.deg2rad(lat2))
        * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing from point 1 to point 2 in degrees."""
    lat1_r = np.deg2rad(lat1)
    lat2_r = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return (np.rad2deg(bearing) + 360) % 360


def parse_osm_pbf(osm_path: Path, db_path: Path) -> Tuple[int, int, int]:
    """
    Parse OSM PBF file and populate SQLite database.
    Returns (num_nodes, num_ways, num_segments).
    """
    import osmium
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    class OSMHandler(osmium.SimpleHandler):
        def __init__(self, conn):
            super().__init__()
            self.conn = conn
            self.cursor = conn.cursor()
            self.nodes_count = 0
            self.ways_count = 0
            self.segments_count = 0
            self.node_cache = {}

        def node(self, n):
            self.node_cache[n.id] = (n.location.lat, n.location.lon)
            self.cursor.execute(
                "INSERT OR REPLACE INTO nodes (id, lat, lon) VALUES (?, ?, ?)",
                (n.id, n.location.lat, n.location.lon),
            )
            self.nodes_count += 1

        def way(self, w):
            if "highway" not in w.tags:
                return

            node_ids = [n.ref for n in w.nodes]
            if len(node_ids) < 2:
                return

            tags = dict(w.tags)
            one_way = tags.get("oneway", "no") in ("yes", "true", "1", "-1")

            # Insert way
            self.cursor.execute(
                "INSERT OR REPLACE INTO ways (id, node_seq, tags, one_way) VALUES (?, ?, ?, ?)",
                (w.id, str(node_ids), str(tags), int(one_way)),
            )
            self.ways_count += 1

            # Create segments between consecutive nodes
            for i in range(len(node_ids) - 1):
                n1, n2 = node_ids[i], node_ids[i + 1]
                if n1 not in self.node_cache or n2 not in self.node_cache:
                    continue
                lat1, lon1 = self.node_cache[n1]
                lat2, lon2 = self.node_cache[n2]
                length = haversine_m(lat1, lon1, lat2, lon2)
                bearing = bearing_deg(lat1, lon1, lat2, lon2)

                self.cursor.execute(
                    """INSERT INTO segments (n1, n2, length_m, bearing_deg, way_id, one_way)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (n1, n2, length, bearing, w.id, int(one_way)),
                )
                self.segments_count += 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    handler = OSMHandler(conn)
    handler.apply_file(str(osm_path))
    conn.commit()
    logger.info(
        f"Parsed OSM: {handler.nodes_count} nodes, "
        f"{handler.ways_count} ways, {handler.segments_count} segments"
    )
    return handler.nodes_count, handler.ways_count, handler.segments_count


def download_osm(city: str, out_path: Path) -> Path:
    """
    Download OSM extract for a city/region from Geofabrik.
    Returns path to downloaded file.
    """
    # Geofabrik URLs for common regions
    geofabrik_base = "https://download.geofabrik.de"
    urls = {
        "delhi": f"{geofabrik_base}/asia/india/delhi-latest.osm.pbf",
        "mumbai": f"{geofabrik_base}/asia/india/maharashtra-latest.osm.pbf",
        "bangalore": f"{geofabrik_base}/asia/india/karnataka-latest.osm.pbf",
        "hyderabad": f"{geofabrik_base}/asia/india/telangana-latest.osm.pbf",
        "chennai": f"{geofabrik_base}/asia/india/tamil-nadu-latest.osm.pbf",
        "kolkata": f"{geofabrik_base}/asia/india/west-bengal-latest.osm.pbf",
        "pune": f"{geofabrik_base}/asia/india/maharashtra-latest.osm.pbf",
    }

    url = urls.get(city.lower())
    if not url:
        raise ValueError(f"Unknown city '{city}'. Available: {list(urls.keys())}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logger.info(f"Using cached OSM file: {out_path}")
        return out_path

    logger.info(f"Downloading OSM for {city} from {url}")
    urllib.request.urlretrieve(url, out_path)
    logger.info(f"Downloaded to {out_path}")
    return out_path


def import_osm(
    city: str = None,
    osm_file: Path = None,
    config: Any = None,
) -> MapStore:
    """
    Import OSM data for a city or from a local file.
    Returns initialized MapStore.
    """
    config = config or get_config()
    maps_dir = Path(config.paths.maps)
    maps_dir.mkdir(parents=True, exist_ok=True)

    if osm_file is None and city is None:
        raise ValueError("Either city or osm_file must be provided")

    if osm_file is None:
        osm_file = maps_dir / f"{city}.osm.pbf"
        download_osm(city, osm_file)

    Path(config.paths.maps) / "road_graph.db"
    store = MapStore(config)
    store.init_schema()

    if not store.is_ready() or store.count_segments() == 0:
        logger.info(f"Parsing OSM file: {osm_file}")
        parse_osm_pbf(osm_file, Path(config.paths.maps) / "road_graph.db")

    store.build_rtree()
    return store


def main():
    """CLI entry point for map import."""
    import argparse

    parser = argparse.ArgumentParser(description="Import OSM map for AeroNavis")
    parser.add_argument("--city", help="City name (delhi, mumbai, etc.)")
    parser.add_argument("--file", help="Local OSM PBF file path")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    store = import_osm(
        city=args.city, osm_file=Path(args.file) if args.file else None, config=args.config
    )
    print(f"Map imported. Segments: {store.count_segments()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
