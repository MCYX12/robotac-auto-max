"""A changed mixture must not hide an actual shift of the fixed main surface."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'validation'))
from surface_composition_review import classify, EXPLAINED


def visit(core=80, trim=20, shift=0., frames=4, width=.001):
    return [np.r_[np.linspace(-width, width, core), np.linspace(.079, .081, trim)]+shift
            for _ in range(frames)]


def test_static_surface_and_changing_facade_mix_is_corroborated_in_both_sources():
    data = [visit(), visit(core=30, trim=70)]
    result = classify(data, data)
    assert result['status'] == EXPLAINED
    assert abs(result['changes'][0]['filtered_broad_change_m']) > .01
    assert result['changes'][0]['maximum_core_change_m'] < .001


@pytest.mark.parametrize('shift', [.008, .03])
def test_rigid_main_surface_shift_cannot_be_explained_as_mix(shift):
    data = [visit(), visit(shift=shift)]
    assert classify(data, data)['status'] != EXPLAINED


def test_filter_only_distribution_bias_requires_review():
    raw = [visit(), visit()]
    filtered = [visit(), visit(core=30, trim=70)]
    assert classify(raw, filtered)['status'] == 'RAW_FILTERED_DISAGREEMENT'


def test_too_few_core_points_are_insufficient_not_agreement():
    data = [visit(), visit(core=7, trim=20)]
    assert classify(data, data)['status'] == 'INSUFFICIENT_PAIRED_COMPONENT_SUPPORT'


def test_two_core_frames_cannot_establish_repeat_support():
    data = [visit(), visit(core=30, trim=70, frames=2)]
    assert classify(data, data)['status'] == 'INSUFFICIENT_PAIRED_COMPONENT_SUPPORT'


def test_broad_core_cannot_be_made_stable_by_truncation():
    data = [visit(width=.012), visit(core=30, trim=70, width=.012)]
    assert classify(data, data)['status'] == 'MAIN_COMPONENT_TOO_BROAD'


def test_missing_source_visit_does_not_clear_review():
    assert classify([visit()], [visit(), visit(core=30, trim=70)])['status'] != EXPLAINED
