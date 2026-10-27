"""Acceptance rules must not turn wrong reads into oracle-selected successes."""

import pytest

from scripts.mobilenet_followup.core import choose, counts, empty


def obs(text, image, fmt="Code128"):
    return {"text": text, "format": fmt if text else None, "image_sha256": image}


def test_wrong_first_read_is_preserved_in_early_exit():
    d0, later = obs("wrong", "raw"), [obs("correct", "warp")]
    assert choose("early_exit", d0, later)["text"] == "wrong"
    assert choose("always_rectify", d0, later)["text"] == "correct"
    assert choose("agreement", d0, later)["reason"] == "conflicting_reads"


def test_first_wrong_warp_is_not_replaced_using_later_truth():
    warped = [obs("wrong", "w1"), obs("correct", "w2"), obs("correct", "w3")]
    assert choose("always_rectify", obs(None, "raw"), warped)["text"] == "wrong"
    assert choose("early_exit", obs(None, "raw"), warped)["text"] == "wrong"
    assert choose("agreement", obs(None, "raw"), warped)["text"] is None


@pytest.mark.parametrize("raw", [None, "same"])
def test_agreement_accepts_two_distinct_images(raw):
    assert (
        choose("agreement", obs(raw, "raw"), [obs("same", "w1"), obs("same", "w2")])["text"]
        == "same"
    )


def test_one_candidate_is_not_agreement():
    assert choose("agreement", obs(None, "raw"), [obs("one", "w1")])["text"] is None


def test_passthrough_duplicate_pixels_cannot_vote_twice():
    assert choose("agreement", obs("one", "identical"), [obs("one", "identical")])["text"] is None


def test_different_symbologies_conflict_even_if_text_matches():
    assert choose("agreement", obs("123", "raw"), [obs("123", "warp", "EAN13")])["text"] is None


def test_two_correlated_wrong_reads_can_still_be_accepted():
    # Agreement is a tradeoff experiment, never a guarantee of correct ground truth.
    assert choose("agreement", obs("wrong", "raw"), [obs("wrong", "warp")])["text"] == "wrong"


@pytest.mark.parametrize("policy", ["early_exit", "always_rectify", "agreement"])
def test_no_return_remains_no_return(policy):
    assert choose(policy, obs(None, "raw"), [obs(None, "w")])["text"] is None


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        choose("oracle", empty(), [])


def test_exact_counts_do_not_count_wrong_returns_as_success():
    assert counts(["C", "C", "W", "U"]) == {
        "n": 4,
        "C": 2,
        "W": 1,
        "U": 1,
        "EDR": 50.0,
        "FDR": 25.0,
        "NRR": 25.0,
    }


def test_paired_report_preserves_wrong_vs_no_return():
    from scripts.mobilenet_followup.report import paired

    left = [
        {"id": "a", "gt_text": "p1", "statuses": {"p": "W"}},
        {"id": "b", "gt_text": "p2", "statuses": {"p": "C"}},
    ]
    right = [
        {"id": "a", "gt_text": "p1", "statuses": {"q": "C"}},
        {"id": "b", "gt_text": "p2", "statuses": {"q": "U"}},
    ]
    result = paired(left, right, "p", "q")
    assert result["EDR_delta_pp"] == 0
    assert result["FDR_delta_pp"] == -50
    assert result["correct_gains"] == result["correct_losses"] == 1


def test_timing_audit_rejects_incorrect_quantile():
    import pytest

    from scripts.mobilenet_followup.report import verify_timing

    summary = {"paths": {"p": {"passes": [{"raw_ms": [1, 2, 3], "p50_ms": 1, "p95_ms": 3}]}}}
    with pytest.raises(AssertionError):
        verify_timing(summary)
