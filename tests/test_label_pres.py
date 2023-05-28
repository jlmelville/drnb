import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from drnb.eval.labelpres import count_integers, get_counts, label_pres, majority_vote


def test_majority_vote_all_ties():
    true_labels = np.array([0, 1, 2, 3, 4])
    predicted_labels = np.array(
        [
            [0, 1, 0, 1, 2],
            [1, 2, 1, 2, 3],
            [2, 3, 2, 3, 4],
            [3, 4, 3, 4, 0],
            [4, 1, 4, 1, 2],
        ]
    )

    assert_allclose(
        majority_vote(true_labels, predicted_labels),
        [0.5, 0.5, 0.5, 0.5, 0.5],
        rtol=1e-3,
        atol=1e-6,
    )


def test_majority_vote_ties_with_one_failure():
    true_labels = np.array([0, 1, 2, 3, 4])
    predicted_labels = np.array(
        [
            [0, 1, 0, 1, 2],
            [1, 2, 1, 2, 3],
            [2, 3, 2, 3, 4],
            [3, 4, 4, 4, 2],
            [4, 0, 4, 0, 1],
        ]
    )

    assert_allclose(
        majority_vote(true_labels, predicted_labels),
        [0.5, 0.5, 0.5, 0.0, 0.5],
        rtol=1e-3,
        atol=1e-6,
    )


def test_mixed_results():
    true_labels = np.array([0, 1, 2, 3, 4, 0])
    predicted_labels = np.array(
        [
            [0, 0, 1],  # Correct classification (Majority is 0)
            [2, 1, 2],  # Incorrect classification (Majority is 2)
            [2, 2, 3],  # Correct classification (Majority is 2)
            [3, 4, 4],  # Incorrect classification (Majority is 4)
            [
                4,
                1,
                2,
            ],  # Tie (No majority vote, multiple labels have the same highest count)
            [0, 4, 1],  # Tie
        ]
    )
    assert_allclose(
        majority_vote(true_labels, predicted_labels),
        [1.0, 0.0, 1.0, 0.0, 0.33333, 0.5],
        rtol=1e-3,
        atol=1e-6,
    )


def test_count_integers():
    numbers = np.array([1, 2, 3, 2, 1, 2, 3, 1, 1, 2])
    counts = count_integers(numbers)
    assert_array_equal(counts, np.array([0, 4, 4, 2]))


def test_get_counts():
    numbers = [1, 2, 3, 2, 1, 2, 3, 1, 1, 2]
    counts = get_counts(numbers)
    assert_array_equal(counts, np.array([4, 4, 2, 4, 4, 4, 2, 4, 4, 4]))


def test_label_pres_unbalanced():
    labels = np.array([0, 1, 2, 3, 0])
    nbr_idxs = np.array([[1, 2], [2, 3], [3, 4], [4, 0], [0, 1]])
    n_neighbors = [1, 2, 3]
    result = label_pres(labels, nbr_idxs, n_neighbors, balanced=False)
    expected_result = [0.2, 0.1333, np.nan]
    assert np.allclose(
        result,
        expected_result,
        equal_nan=True,
        rtol=1e-3,
        atol=1e-6,
    )


def test_label_pres_balanced():
    labels = np.array([0, 1, 2, 3, 0])
    nbr_idxs = np.array([[1, 2], [2, 3], [3, 4], [4, 0], [0, 1]])
    n_neighbors = [1, 2, 3]
    result = label_pres(labels, nbr_idxs, n_neighbors, balanced=True)
    expected_result = [0.125, 0.08333, np.nan]
    assert np.allclose(
        result,
        expected_result,
        equal_nan=True,
        rtol=1e-3,
        atol=1e-6,
    )


def test_label_pres_balanced_and_unbalanced_are_equal_for_equal_label_frequencies():
    labels = np.array([0, 1, 2, 3])
    nbr_idxs = np.array([[0, 0], [2, 1], [3, 0], [1, 0]])
    n_neighbors = [1, 2, 3]
    balanced_result = label_pres(labels, nbr_idxs, n_neighbors, balanced=True)
    unbalanced_result = label_pres(labels, nbr_idxs, n_neighbors, balanced=False)
    expected_result = [0.25, 0.375, np.nan]
    assert np.allclose(
        balanced_result,
        expected_result,
        equal_nan=True,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.allclose(
        balanced_result,
        unbalanced_result,
        equal_nan=True,
    )
