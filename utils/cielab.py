from functools import partial
import matplotlib.pyplot as plt
import numpy as np

class ABGamut:
    RESOURCE_POINTS = "../../utils/gamut_pts.npy"
    RESOURCE_PRIOR = "../../utils/gamut_probs.npy"
    DTYPE = np.float32
    EXPECTED_SIZE = 313
    def __init__(self):
        self.points = np.load(self.RESOURCE_POINTS).astype(self.DTYPE)
        self.prior = np.load(self.RESOURCE_PRIOR).astype(self.DTYPE)
        assert self.points.shape == (self.EXPECTED_SIZE, 2)
        assert self.prior.shape == (self.EXPECTED_SIZE,)


class CIELAB:
    L_MEAN = 50
    AB_BINSIZE = 10
    AB_RANGE = [-110 - AB_BINSIZE // 2, 110 + AB_BINSIZE // 2, AB_BINSIZE]
    AB_DTYPE = np.float32
    Q_DTYPE = np.int64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, gamut=None):
        self.gamut = gamut if gamut is not None else ABGamut()
        a, b, self.ab = self._get_ab()
        self.ab_gamut_mask = self._get_ab_gamut_mask(
            a, b, self.ab, self.gamut)

        self.ab_to_q = self._get_ab_to_q(self.ab_gamut_mask)
        self.q_to_ab = self._get_q_to_ab(self.ab, self.ab_gamut_mask)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))
        return a, b, ab

    @classmethod
    def _get_ab_gamut_mask(cls, a, b, ab, gamut):
        ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)
        a = np.digitize(gamut.points[:, 0], a) - 1
        b = np.digitize(gamut.points[:, 1], b) - 1
        for a_, b_ in zip(a, b):
            ab_gamut_mask[a_, b_] = True

        return ab_gamut_mask

    @classmethod
    def _get_ab_to_q(cls, ab_gamut_mask):
        ab_to_q = np.full(ab_gamut_mask.shape, -1, dtype=cls.Q_DTYPE)
        ab_to_q[ab_gamut_mask] = np.arange(np.count_nonzero(ab_gamut_mask))

        return ab_to_q

    @classmethod
    def _get_q_to_ab(cls, ab, ab_gamut_mask):
        return ab[ab_gamut_mask] + cls.AB_BINSIZE / 2

    def bin_ab(self, ab):
        ab_discrete = ((ab + 110) / self.AB_RANGE[2]).astype(int)

        a, b = np.hsplit(ab_discrete.reshape(-1, 2), 2)

        return self.ab_to_q[a, b].reshape(*ab.shape[:2])

