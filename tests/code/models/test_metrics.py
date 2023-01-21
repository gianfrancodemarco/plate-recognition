from src.models.metrics import lev_dist

class TestMetrics():
    def test_lev_dist(self):
        assert lev_dist("book", "back") == 2
        assert lev_dist("passport", "") == 8
        assert lev_dist("", "passport") == 8
        assert lev_dist("passport", "airport") == 3
        assert lev_dist("water", "water") == 0
