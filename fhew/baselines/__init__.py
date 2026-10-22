"""Fair baselines for the identification task (paper Section 11).

Every baseline sees exactly what the method sees: the same lossy observation
stream and the same sensor-aware likelihoods. What differs is only the decision
machinery, which is the point of the comparison.

    * :class:`PairwiseSPRT`  -- the classic two-hypothesis SPRT generalised to a
      bank by a leader-vs-all rule; has **no** H_0, so it is forced to pick a
      generator even when the true one is absent.
    * :class:`FixedLengthTest` -- observe a fixed number of slots, then declare
      the MAP hypothesis; no early stopping.
    * :class:`LearnedClassifier` -- a feature-based softmax/MLP classifier
      trained offline (optional torch-LSTM drop-in in :mod:`fhew.baselines.lstm`).
"""

from fhew.baselines.common import LikelihoodTracker
from fhew.baselines.fixed_length import FixedLengthTest
from fhew.baselines.learned import LearnedClassifier, episode_features
from fhew.baselines.pairwise_sprt import PairwiseSPRT

__all__ = [
    "LikelihoodTracker",
    "PairwiseSPRT",
    "FixedLengthTest",
    "LearnedClassifier",
    "episode_features",
]
