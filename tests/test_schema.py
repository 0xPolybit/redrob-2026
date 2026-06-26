"""A rank-style top-100 CSV must pass the organizers' validate_submission.py."""
import csv

import validate_submission as vs


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)


def _valid_rows():
    rows = []
    score = 1.0
    for i in range(1, 101):
        rows.append([f"CAND_{i:07d}", i, f"{score:.6f}", f"reason {i}"])
        score -= 0.001
    return rows


def test_valid_submission_passes(tmp_path):
    p = tmp_path / "team_test.csv"
    _write(p, _valid_rows())
    assert vs.validate_submission(str(p)) == []


def test_ties_need_ascending_candidate_id(tmp_path):
    # equal scores with candidate_id ascending is allowed; keep the tie at the
    # top so the rest of the sequence stays non-increasing.
    rows = _valid_rows()
    rows[0][2] = rows[1][2] = "1.000000"
    rows[0][0], rows[1][0] = "CAND_0000001", "CAND_0000002"
    p = tmp_path / "team_ties.csv"
    _write(p, rows)
    assert vs.validate_submission(str(p)) == []


def test_wrong_row_count_fails(tmp_path):
    p = tmp_path / "team_bad.csv"
    _write(p, _valid_rows()[:99])
    assert vs.validate_submission(str(p))  # non-empty error list


def test_increasing_score_fails(tmp_path):
    rows = _valid_rows()
    rows[0][2], rows[1][2] = "0.100000", "0.900000"  # rank 1 < rank 2
    p = tmp_path / "team_inc.csv"
    _write(p, rows)
    assert vs.validate_submission(str(p))
