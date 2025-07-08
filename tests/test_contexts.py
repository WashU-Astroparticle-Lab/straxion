import straxion

def test_qualiphide_runs_without_error():
    st = straxion.qualiphide()
    assert st is not None 