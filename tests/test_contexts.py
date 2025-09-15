import straxion


def test_qualiphide_thz_online_runs_without_error():
    st = straxion.qualiphide_thz_online()
    assert st is not None


def test_qualiphide_thz_offline_runs_without_error():
    st = straxion.qualiphide_thz_offline()
    assert st is not None
