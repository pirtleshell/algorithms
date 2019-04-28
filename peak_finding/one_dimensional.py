# find a peak (any peak) if it exists.

import matplotlib.pyplot as plt
import numpy as np

NUM_POINTS = 51


# make random data the same for all instances
np.random.seed(0)

def build_peak_in_middle_data():
    peak = np.floor(NUM_POINTS/2)
    right = 2*peak - NUM_POINTS
    uphill = np.arange(0, peak)
    downhill = np.arange(peak, right, -1)
    return np.concatenate((uphill, downhill))

plot_data = [
    ('Peak on left', np.arange(NUM_POINTS, 0, -1)),
    ('Peak on right', np.arange(NUM_POINTS)),
    ('Peak in middle', build_peak_in_middle_data()),
    ('Random peaks', np.random.randint(0, 6, NUM_POINTS))
]


class PeakFinder1D:
    def __init__(self, name, func):
        # name of algorithm
        self.name = name
        # func takes array of data, returns tuple of (peak_index, num_steps)
        # returns -1 as peak_index if no peak was found
        self.func = func
        self.has_run = False

    def run(self, plotting=False):
        if not self.has_run:
            self.results = {}
            if plotting:
                self.setup_plotting()
            for i, (title, data) in enumerate(plot_data):
                peak_index, num_steps = self.func(data)
                self.results[title] = (peak_index, num_steps)
                if plotting:
                    self.plot_subplot(i, title, data, peak_index)

            if plotting:
                self.fig.tight_layout()
            self.has_run = True

    def setup_plotting(self):
        fig, axes = plt.subplots(2, 2)
        plt.figtext(.5, .96, '1D Peak Finding', fontweight='bold', fontsize=16, ha='center')
        plt.figtext(.5, .93, self.name, fontsize=10, ha='center')
        self.fig = fig
        self.flat_axes = np.ndarray.flatten(axes)

    def plot_subplot(self, subplot_index, title, data, peak_index):
        ax = self.flat_axes[subplot_index]
        alignment = {0: 'left', 1: 'right'}.get(subplot_index, 'center')
        ax.set_title(title, loc=alignment)
        ax.plot(np.arange(NUM_POINTS), data)
        if peak_index != -1:
            ax.plot(peak_index, data[peak_index], 'ro')

    def describe(self):
        if not self.has_run:
            self.run()
        print('## ' + self.name + ':\n')
        print(' | '.join(['Dataset', 'peak_index', 'num_steps']))
        print('--|--|--')
        for key, (peak_index, num_steps) in self.results.items():
            print(' | '.join([key, str(peak_index), str(num_steps)]))



# straightfoward solution
# start on the left, walk to end, stop if you find a peak
def straightfoward(data):
    step = 0
    prev_val = -np.inf
    curr_val = data[0]
    while step < NUM_POINTS:
        next_val = -np.inf if step == NUM_POINTS-1 else data[step + 1]
        if prev_val <= curr_val and curr_val >= next_val:
            return (step, step+1)
        prev_val = curr_val
        curr_val = next_val
        step += 1

    return (-1, step+1)


def main():
    algorithms = [
        PeakFinder1D('Straightfoward', straightfoward)
    ]

    for algorithm in algorithms:
        algorithm.run(plotting=True)
        algorithm.describe()

    plt.show()


if __name__ == '__main__':
    main()
