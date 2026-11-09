# results_schema — versioned output contracts

`v1/` — schema of stage result files (JSON). Stage runners validate their
output against this contract, so downstream consumers (dashboards, the
science program) can rely on the fields. Bump the version when a field
changes meaning; never edit an old schema in place.
