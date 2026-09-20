# aeronavis.predict — outage forecaster

Learns prior route pass-through statistics to predict GPS-blackout survivability
at `predict.horizons` seconds ahead, emitted as a success probability per
horizon. The `predict.threshold` gates whether the fusion solution should keep
driving dead-reckoned or slow down/wait for a fix. Implemented in Part 4.