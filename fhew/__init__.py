"""fhew: FH-EW gym core and sensor-aware generator-identification bank.

This package implements the conference-paper slice (Layer-2 reconnaissance
bank) of the Confidence-Adaptive Multi-Channel EW thesis, and is designed so
every component carries over unchanged into the journal / full-thesis system:

    * ``fhew.sensor``       - detection-level sensor model + frozen schema v1
    * ``fhew.channels``     - channel set
    * ``fhew.generators``   - hop-generator library behind a name-keyed registry
    * ``fhew.recovery``     - Berlekamp-Massey and gap-robust register recovery
    * ``fhew.likelihood``   - sensor-aware HMM forward likelihood (+ naive baseline)
    * ``fhew.experts``      - Expert interface, Hypothesis, H_0, the bank
    * ``fhew.stopping``     - MSPRT and anytime-valid e-process stopping rules
    * ``fhew.baselines``    - pairwise SPRT, fixed-length LRT, learned classifier
    * ``fhew.sim``          - end-to-end slot loop (paper Section 9)
    * ``fhew.metrics``      - identification / calibration / next-hop metrics
    * ``fhew.config``       - config dataclasses + YAML loading
    * ``fhew.datagen``      - dataset generation and grounding ranges

The design rules that keep it reuse-ready (paper Section 12) are enforced in
code: a single Expert interface (``update`` / ``predict`` / ``posterior`` /
``stopped``), a pluggable H_0 likelihood, generators in a registry, config-
driven runs with fixed seeds, logged next-hop distributions, and a frozen
observation schema. The information-parity rule (no likelihood may use anything
the modelled sensor cannot deliver) is structural: the simulator hands the bank
only :class:`~fhew.sensor.Observation` objects and its own recce config.
"""

from fhew.channels import ChannelSet
from fhew.sensor import Detection, Observation, SensorModel
from fhew.predict import ChannelPredictor, PredictionScore, predict_over_episode

__all__ = [
    "ChannelSet",
    "SensorModel",
    "Observation",
    "Detection",
    "ChannelPredictor",
    "PredictionScore",
    "predict_over_episode",
    # Revision 3: confidence-adaptive mixture, regime monitor, allocation.
    "FixedShareMixture",
    "RegimeController",
    "EffectModel",
    "AllocationController",
    "__version__",
]


def __getattr__(name):
    # Lazy re-exports so importing fhew stays light (scipy etc. load on demand).
    if name in ("FixedShareMixture", "HypothesisExpert", "build_default_mixture"):
        import fhew.mixture as m
        return getattr(m, name)
    if name in ("RegimeController", "make_detector", "recovery_time"):
        import fhew.regime as r
        return getattr(r, name)
    if name in ("EffectModel", "AllocationController", "CDTController",
                "greedy_knapsack", "tones_from_confidence", "predictability"):
        import fhew.allocator as a
        return getattr(a, name)
    raise AttributeError(f"module 'fhew' has no attribute {name!r}")

__version__ = "0.1.0"

# Frozen schemas (paper Section 4: "Observation schema (frozen as v1)").
OBSERVATION_SCHEMA_VERSION = "v1"
SENSOR_MODEL_VERSION = "v1"
