from mathx.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow


class Config:
    def __init__(self, sliding_window:SlidingWindow):
        self._sliding_window = sliding_window

    def get_sliding_window(self) -> SlidingWindow:
        return self._sliding_window