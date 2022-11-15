import numpy as np


class NoiseHandler:
    def __init__(self, augmentations):
        self._augmentations = augmentations
        self.generate = augmentations.get('generate')
        self.whether_to_generate = self.generate

    @staticmethod
    def _generate(x_len, dists, do_fpi=False):
        fl = dists.get('fl')
        if np.isnan(fl):
            fl = 0.25
        fnum = dists.get('fnum')
        if np.isnan(fnum):
            fl = 1
        ipi = dists.get('ipi')
        if np.isnan(ipi):
            ipi = 1.0
        fpi = dists.get('fpi')
        if np.isnan(fpi):
            fpi = 1.0

        new_x = []
        frame_rate = 1 / 30
        for flash in range(fnum):
            # generate flash_duration 1s
            on_index = round(fl / frame_rate)
            while on_index > 0:
                try:
                    assert (x_len > len(new_x))
                except AssertionError:
                    break
                new_x.append(1)
                on_index -= 1
            off_index = round(ipi / frame_rate)
            while off_index > 0:
                try:
                    assert (x_len > len(new_x))
                except AssertionError:
                    break
                new_x.append(0)
                off_index -= 1
        if do_fpi:
            # fpi stuff
            for off in range(int((fpi / frame_rate) - (ipi / frame_rate))):
                try:
                    assert (x_len > len(new_x))
                except AssertionError:
                    break
                new_x.append(0)
            num_things_remaining = x_len - len(new_x)
            while num_things_remaining > 0:
                for flash in range(fnum):
                    # generate flash_duration 1s
                    on_index = round(fl / frame_rate)
                    while on_index > 0:
                        try:
                            assert (x_len > len(new_x))
                        except AssertionError:
                            break
                        new_x.append(1)
                        num_things_remaining -= 1
                        on_index -= 1
                    off_index = round(ipi / frame_rate)
                    while off_index > 0:
                        try:
                            assert (x_len > len(new_x))
                        except AssertionError:
                            break
                        new_x.append(0)
                        num_things_remaining -= 1
                        off_index -= 1
                for off in range(int((fpi / frame_rate) - (ipi / frame_rate))):
                    try:
                        assert (x_len > len(new_x))
                    except AssertionError:
                        break
                    new_x.append(0)
                    num_things_remaining -= 1
        else:
            padding_required = x_len - len(new_x)
            padding_i = padding_required
            while padding_i > 0:
                try:
                    assert (x_len > len(new_x))
                except AssertionError:
                    break
                new_x.append(2)
                padding_i -= 1

        return new_x
