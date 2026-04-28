import cyten


def test_check():
    x = cyten.add(1, 1)
    assert x == 2
    assert cyten.add(1, -1) == 0
