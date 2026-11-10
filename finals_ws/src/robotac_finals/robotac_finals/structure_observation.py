"""Bounded feedback for live, measured structural deficits (no scene model)."""
from __future__ import annotations

import math
import numpy as np


def distinct_viewpoint(target_xy, candidate_xy, previous, minimum_angle=math.radians(20),
                       minimum_translation=.75):
    """A retry must add a new angle and location relative to every prior view."""
    vector = np.asarray(candidate_xy) - np.asarray(target_xy)
    for old in previous:
        old_vector = np.asarray(old) - np.asarray(target_xy)
        if np.linalg.norm(np.asarray(candidate_xy) - old) < minimum_translation:
            return False
        if np.linalg.norm(vector) < 1e-9 or np.linalg.norm(old_vector) < 1e-9:
            return False
        angle = abs(math.atan2(vector[1], vector[0]) - math.atan2(old_vector[1], old_vector[0]))
        if abs(math.atan2(math.sin(angle), math.cos(angle))) < minimum_angle:
            return False
    return True


class StructureObservation:
    """Keep target coordinates fixed across all attempts and changing nav grids."""
    def __init__(self, targets, baseline, source_ns, frame, anchor_xy,
                 max_attempts=3, threshold=.25):
        self.targets = np.array(targets, dtype=float, copy=True).reshape(-1, 3)
        self.baseline = np.array(baseline, dtype=float, copy=True).reshape(-1)
        if (not math.isfinite(threshold) or not .10 <= threshold <= .25 or
                int(max_attempts) != max_attempts or not 1 <= max_attempts <= 3 or
                np.asarray(anchor_xy).shape != (2,) or not np.isfinite(anchor_xy).all() or
                not len(self.targets) or len(self.targets) != len(self.baseline)
                or not np.isfinite(self.targets).all() or not np.isfinite(self.baseline).all()
                or not np.all(self.baseline > threshold + 1e-6) or source_ns <= 0):
            raise ValueError('Structural feedback requires finite, genuinely deficient measured targets')
        self.targets.setflags(write=False)
        self.baseline.setflags(write=False)
        self.anchor_xy = np.array(anchor_xy, dtype=float, copy=True)
        self.frame = frame
        self.source_ns = int(source_ns)
        self.last_source_ns = int(source_ns)
        self.max_attempts = int(max_attempts)
        self.threshold = threshold
        self.views = []
        self.attempt = 0
        self.goal_id_ns = None
        self.state = 'RETRY'
        self.latest = None
        self.attempt_before = self.baseline.copy()
        self.reason = 'selected_live_deficit'
        self.deadline = None

    def begin(self, goal_id_ns, viewpoint):
        if (not isinstance(goal_id_ns, int) or goal_id_ns <= 0 or
                np.asarray(viewpoint).shape != (2,) or not np.isfinite(viewpoint).all()):
            raise ValueError('Invalid observation goal or viewpoint')
        if self.state != 'RETRY' or self.attempt >= self.max_attempts:
            raise ValueError('No structural observation attempt available')
        if not distinct_viewpoint(self.anchor_xy, viewpoint, self.views):
            raise ValueError('Observation retry must use a distinct safe angle')
        self.goal_id_ns = int(goal_id_ns)
        self.views.append(np.array(viewpoint, dtype=float, copy=True))
        self.attempt += 1
        self.attempt_before = (self.latest if self.latest is not None else self.baseline).copy()
        self.latest = None
        self.state = 'NAVIGATING'
        self.reason = 'travel_to_observation'

    def arrived(self, goal_id_ns, now_mono, arrival_source_ns, settle_seconds, wait_seconds):
        if self.state != 'NAVIGATING' or goal_id_ns != self.goal_id_ns:
            return False
        self.required_source_ns = max(self.last_source_ns, int(arrival_source_ns + settle_seconds * 1e9))
        self.settle_until = now_mono + settle_seconds
        self.deadline = self.settle_until + wait_seconds
        self.state = 'WAITING'
        self.reason = 'awaiting_post_arrival_native_score'
        return True

    def observe(self, distances, source_ns, frame, now_mono):
        if (self.state != 'WAITING' or frame != self.frame or now_mono < self.settle_until
                or now_mono > self.deadline or source_ns <= self.required_source_ns
                or source_ns <= self.last_source_ns):
            return False
        distances = np.asarray(distances, dtype=float)
        if distances.shape != self.baseline.shape or not np.isfinite(distances).all():
            return False
        self.last_source_ns = int(source_ns)
        self.latest = distances.copy()
        if np.all(distances <= self.threshold + 1e-6):
            self.state = 'RESOLVED'
            self.reason = 'fixed_targets_supported_by_fresh_native_score'
        else:
            self.reason = ('PARTIAL_IMPROVEMENT' if np.any(self.attempt_before - distances > .01)
                           else 'NO_IMPROVEMENT')
        return True

    def finish_attempt(self, reason=None):
        if self.state in ('RESOLVED', 'UNRESOLVED'):
            return
        self.reason = reason or (self.reason if self.latest is not None else 'NO_FRESH_SCORE')
        self.state = 'RETRY' if self.attempt < self.max_attempts else 'UNRESOLVED'

    def payload(self):
        return {
            'state': 'STRUCTURE_OBSERVATION', 'observation_state': self.state,
            'reason': self.reason, 'goal_id_ns': self.goal_id_ns,
            'attempt': self.attempt, 'maximum_attempts': self.max_attempts,
            'frame': self.frame, 'target_xyz': self.targets.tolist(),
            'target_anchor_xy': self.anchor_xy.tolist(),
            'viewpoints_xy': [view.tolist() for view in self.views],
            'baseline_distance_m': self.baseline.tolist(),
            'attempt_before_distance_m': self.attempt_before.tolist(),
            'after_distance_m': self.latest.tolist() if self.latest is not None else None,
            'baseline_source_ns': self.source_ns, 'last_source_ns': self.last_source_ns,
            'support_tolerance_m': self.threshold, 'geometric_acceptance': False,
        }
