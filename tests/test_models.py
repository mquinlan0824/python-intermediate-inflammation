"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

def test_daily_mean_zeros():

    """Test that mean function works for an array of zeros."""

    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():

    """Test that mean function works for an array of positive integers."""

    from inflammation.models import daily_mean
    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_negatives():

    """Test that mean function works for an array of positive integers."""

    from inflammation.models import daily_mean

    test_input = np.array([[-1, -2],
                           [-3, -4],
                           [-5, -6]])
    test_result = np.array([-3, -4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [8, 2], [3, 4], [5, 6] ], [3, 2]),

        ([ [1, 7], [9, 8], [5, 6] ], [1, 6]),
    ])

def test_daily_max_min(test, expected):

    """Test mean function works for array of zeroes and positive integers."""

    from inflammation.models import daily_max, daily_min
    #npt.assert_array_equal(daily_max(np.array(test)), np.array(expected), err_msg='')
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected), err_msg='')

def test_daily_max():

    """Test that mean function works for an array of positive integers."""

    from inflammation.models import daily_max

    test_input = np.array([[8, 2, 3],
                           [3, 4, 9],
                           [5, 6, 2]])
    test_result = np.array([8, 6, 9])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min():

    """Test that mean function works for an array of positive integers."""

    from inflammation.models import daily_min

    test_input = np.array([[8, 2, 3],
                           [3, 4, 9],
                           [5, 6, 2]])
    test_result = np.array([3, 2, 2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_min_string():

    """Test for TypeError when passing strings"""

    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

