import pytest

from app import square


def test_square_positive():
    assert square(5) == 25


def test_square_zero():
    assert square(0) == 0


def test_square_negative():
    assert square(-4) == 16


def test_square_float():
    assert square(2.5) == 6.25


def test_square_large_number():
    assert square(1000) == 1000000

# I added the all the branches inci.yaml file so the test will occur when code is pushed to alll branches
