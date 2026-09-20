# aeronavis.eval — metrics, blackout protocol, adaptation eval

- **metrics**: drift/error measures used at the leaderboard-adjacent level.
- **blackout**: deterministic outage protocol — drop GNSS at chosen timestamps,
  replay the filter, measure error growth vs ground truth.
- **adapt_eval**: quantify how quickly adaptation recovers after reacquire.
Implemented in Parts 9–11.