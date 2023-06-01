from filtration_for_50 import *
from optimize_mapper import *
import random

eps = [1.52]

min_samples = [6]

overlaps = []

for e in eps:
    for min_samp in min_samples:
        print(e, min_samp)
        best_cube, best_overlap = find_best(e, min_samp)
        ticks = get_ticks()
        # numbers = [i for i in range(100)]
        # sample = random.sample(numbers, 50)
        # ticks_sample = []
        # for i in sample:
        #     ticks_sample.append(ticks[4*i])
        #     ticks_sample.append(ticks[4*i+1])
        #     ticks_sample.append(ticks[4*i+2])
        #     ticks_sample.append(ticks[4*i+3])

        get_p_norms(ticks, eps=e, ms=min_samp, cube=best_cube, overlap = best_overlap, c="all")

        all_norms = pd.read_csv("185_8_60_25_all.csv")

        tick_0 = all_norms.loc[all_norms["tick number"] == 0]
        tick_48 = all_norms.loc[all_norms["tick number"] == 48]
        tick_96 = all_norms.loc[all_norms["tick number"] == 96]
        tick_144 = all_norms.loc[all_norms["tick number"] == 144]

        overlap = 0

        if (tick_48.min()[1] - tick_0.max()[1]) < 0:
            overlap = overlap - (tick_48.min()[1] - tick_0.max()[1])

        if (tick_96.min()[1] - tick_48.max()[1]) < 0:
            overlap = overlap - (tick_96.min()[1] - tick_48.max()[1])

        if (tick_144.min()[1] - tick_96.max()[1]) < 0:
            overlap = overlap - (tick_144.min()[1] - tick_96.max()[1])

        overlaps.append(overlap)


        print(f'\n-----\nEpsilon: {e}\nMin Points: {min_samp}\nOverlap: {overlap}\n-----')


sorted_overlaps = np.sort(overlaps)
smallest_five = sorted_overlaps[:5]
first = overlaps.index(smallest_five[0])
second = overlaps.index(smallest_five[1])
third = overlaps.index(smallest_five[2])
fourth = overlaps.index(smallest_five[3])
fifth = overlaps.index(smallest_five[4])


num_of_ms = len(min_samples)
ms_min = first % num_of_ms
eps_min = int((first - ms_min) / num_of_ms)

print(f'Minimum params with norm of {np.min(overlaps)} - {eps[eps_min]} {min_samples[ms_min]}')

ms_min = second % num_of_ms
eps_min = int((second - ms_min) / num_of_ms)

print(f'Second minimum params with norm of {smallest_five[1]} - {eps[eps_min]} {min_samples[ms_min]}')

ms_min = third % num_of_ms
eps_min = int((third - ms_min) / num_of_ms)

print(f'Third minimum params with norm of {smallest_five[2]} - {eps[eps_min]} {min_samples[ms_min]}')

ms_min = fourth % num_of_ms
eps_min = int((fourth - ms_min) / num_of_ms)

print(f'Fourth minimum params with norm of {smallest_five[3]} - {eps[eps_min]} {min_samples[ms_min]}')

ms_min = fifth % num_of_ms
eps_min = int((fifth - ms_min) / num_of_ms)

print(f'Fifth minimum params with norm of {smallest_five[4]} - {eps[eps_min]} {min_samples[ms_min]}')

# 1.4 5 6
# 1.43 6 7
# 1.46 7
# 1.49 6 7
# 1.52 6 7
# 1.61 7
# 1.64 7
# 1.67 7

# 1.4 5 (53) 6 (3)
# 1.52 6 (.2)
# 1.61 7 (12)