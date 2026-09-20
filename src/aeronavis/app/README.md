# aeronavis.app — maps, map-match, r-anchor

- **maps**: road-graph access for the footprint (OSM-style).
- **mapmatch**: HMM+geometry snap of filtered poses to the graph.
- **ranchor**: reference-anchor bookkeeping — known landmarks that can
  re-anchor a drifted dead-reckoned pose. Implemented in Parts 12–13.