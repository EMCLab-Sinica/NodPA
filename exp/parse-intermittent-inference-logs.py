import argparse
import contextlib
import datetime
import math
import time
import statistics

# If the active time of an inference is very short (ex: < 100ms), it's very likely the inference
# completion signal is a false positive. Specifically, when an inference completion signal is sent
# over GPIO, the inference engine may not record the sent signal on NVM before power fails, and thus
# upon power resumption, the inference completion signal is sent again.
VALID_ACTIVE_TIME_THRESHOLD_MILLISECONDS = 100

# For fixing recharging time
# If power on/off is detected by whether VBAT_OK is high or low (i.e., higher or lower than 1.5V),
# the device remains on until VBAT_OK is roughly 1V. The gap between 1.5V and 1V is roughly 45
# milliseconds.
# If power on/off is detected by whether 3V3 is high or low, set this value to 0.
# EXTRA_POWER_ON_TIME = 45
EXTRA_POWER_ON_TIME = 0

def find_close_subarrays(A: list[float], B: list[float], threshold: float, avg_diff_threshold: float, subarray_length: int = 5) -> list[tuple]:
    """
    Finds pairs of non-overlapping, consecutive subarrays in B (where all values
    are under a threshold) whose corresponding subarrays in A have close average values.

    This function first identifies all candidate subarrays in B that meet the
    `threshold` condition. It then iterates through these candidates to find
    non-overlapping pairs whose corresponding subarrays in A have average
    values with a difference less than `avg_diff_threshold`.

    Args:
        A (list[float]): The first array of numerical values.
        B (list[float]): The second array of numerical values, with the same length as A.
        threshold (float): The maximum value allowed for elements in a valid
            subarray of B.
        avg_diff_threshold (float): The maximum allowed absolute difference
            between the averages of the corresponding A subarrays.

    Returns:
        list[tuple]: A list of tuples. Each tuple contains information about
            a pair of valid subarrays. The format is:
            ((start_idx1, end_idx1), (start_idx2, end_idx2))
            where start and end indices are inclusive.
    """
    if len(A) != len(B):
        raise ValueError("Arrays A and B must have the same length.")

    n = len(A)
    candidate_subarrays = []

    # Step 1: Find all consecutive subarrays in B where all values are under the threshold
    i = 0
    while i < n:
        # If the current value is below the threshold, it could be a start of a new subarray
        if B[i] < threshold:
            j = i
            # Expand the subarray until a value is at or above the threshold
            while j < n and B[j] < threshold:
                j += 1

            # Found a valid consecutive subarray from index i to j-1
            subarray_b = B[i:j]
            subarray_a = A[i:j]

            # A subarray must have many elements to be considered
            if len(subarray_b) >= subarray_length and j < n:
                # Calculate the average of the corresponding A subarray
                avg_a = sum(subarray_a) / len(subarray_a)
                candidate_subarrays.append({
                    'start': i,
                    'end': j - 1,
                    'avg_a': avg_a
                })

            i = j
        else:
            i += 1

    # Step 2: Find pairs of non-overlapping candidate subarrays with close averages
    results = []
    num_candidates = len(candidate_subarrays)

    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            sub1 = candidate_subarrays[i]
            sub2 = candidate_subarrays[j]

            # Check for non-overlapping condition
            # Subarrays are non-overlapping if the end of the first is before
            # the start of the second, or vice-versa. Since we are iterating
            # with j > i, we only need to check if sub1 ends before sub2 starts.
            is_non_overlapping = sub1['end'] < sub2['start']

            if is_non_overlapping:
                # Check if the average values of the corresponding A subarrays are close
                avg_diff = max(sub1['avg_a'] / sub2['avg_a'], sub2['avg_a'] / sub1['avg_a'])
                if avg_diff <= avg_diff_threshold:
                    results.append((sub1, sub2))

    return results

def check_latencies(average_latencies, ratio_average_latencies):
    results = find_close_subarrays(average_latencies, ratio_average_latencies, threshold=1.03, avg_diff_threshold=1.03)
    if results:
        sub1, sub2 = results[-1]
        if max(ratio_average_latencies[sub1['start']:]) < 1.04:
            print('sub1', sub1['avg_a']) #, ratio_average_latencies[sub1['start']:sub1['end']])
            print('sub2', sub2['avg_a']) #, ratio_average_latencies[sub2['start']:sub2['end']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename', required=True)
    parser.add_argument('--num-inferences', type=int, required=True)
    parser.add_argument('--follow', action=argparse.BooleanOptionalAction)
    parser.add_argument('--continuous-power', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    initial_timestamp = None

    MOVING_AVERAGE_LEN = 25

    with open(args.log_filename, 'r') as log_file, open(args.log_filename.replace('-raw.log', '.log'), 'w') if args.log_filename.endswith('-raw.log') else contextlib.nullcontext() as filtered_log_file:
        active_times = []
        inference_latencies = []
        average_latencies = []
        recent_average_latency_ratios = []
        accumulated_recharging_time = 0
        short_recharging_cycles = [0]
        power_failures = []
        first_inference_dropped = False

        current_line = ''

        while True:
            line = log_file.readline()
            # collect incomplete lines, as readline may not read a full line
            # when the last line is not fully written to the log file
            if '\n' not in line:
                current_line += line
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
                continue
            line = (current_line + line).strip()
            current_line = ''

            if ']' not in line:
                print(f'WARNING: found corrupted output line {repr(line)}')
                continue

            original_line = line

            timestamp, line = line.split(']')
            timestamp = datetime.datetime.strptime(timestamp.removeprefix('['), '%Y-%m-%d %H:%M:%S.%f')

            if not initial_timestamp:
                initial_timestamp = timestamp
            # Ignore lines within the first second, as those may contain data before logging starts
            if (timestamp - initial_timestamp).total_seconds() <= 1.0:
                continue

            line = line.strip()
            if line.startswith('.'):
                line = line[1:]
                if ' ' not in line:
                    print(f'WARNING: found corrupted output line {repr(line)}')
                    continue

                counter, power_failure = line.split(' ')
                cur_inference_latency = int(counter)
                # print(f'Cur inference latency: {cur_inference_latency}')
                power_failure = int(power_failure.removeprefix('PF='))

                cur_active_time = cur_inference_latency - accumulated_recharging_time

                found_new_inference = False
                if not inference_latencies:
                    found_new_inference = True
                else:
                    if cur_active_time < VALID_ACTIVE_TIME_THRESHOLD_MILLISECONDS:
                        found_new_inference = False
                    else:
                        found_new_inference = True

                if not found_new_inference:
                    if inference_latencies:
                        print(f'WARNING: Skip invalid latency {cur_inference_latency}')

                if found_new_inference:
                    inference_latencies.append(cur_inference_latency)
                    active_times.append(cur_active_time)
                    power_failures.append(power_failure)
                else:
                    inference_latencies[-1] += cur_inference_latency
                    active_times[-1] += cur_active_time
                    power_failures[-1] += power_failure

                accumulated_recharging_time = 0
                short_recharging_cycles.append(0)

                if not first_inference_dropped:
                    # Exclude the first inference, whose latency includes experiment setup time
                    inference_latencies = inference_latencies[1:]
                    active_times = active_times[1:]
                    power_failures = power_failures[1:]
                    first_inference_dropped = True
                    continue

                if args.continuous_power and power_failures[-1] != 0:
                    inference_latencies = inference_latencies[:-1]
                    active_times = active_times[:-1]
                    power_failures = power_failures[:-1]
                    continue

                if filtered_log_file:
                    filtered_line = original_line.replace(' .', ' Latency=').replace(' PF=', ', PF=') + f', AT={active_times[-1]}'
                    filtered_log_file.write(filtered_line + '\n')

                if len(inference_latencies) >= args.num_inferences:
                    if False:
                        import numpy as np
                        from scipy.stats.mstats import winsorize
                        winsorized_inference_latencies = winsorize(np.array(inference_latencies[-args.num_inferences:]), limits=[0.05, 0.05]).tolist()
                    else:
                        winsorized_inference_latencies = inference_latencies[-args.num_inferences:]
                    average_latency = statistics.mean(winsorized_inference_latencies)
                    average_latencies.append(average_latency)
                    print(f'Found {len(inference_latencies)} inferences. Average latency for the last {args.num_inferences} = {average_latency:.4f}', end='')

                    if len(average_latencies) >= MOVING_AVERAGE_LEN:
                        recent_average_latencies = average_latencies[-MOVING_AVERAGE_LEN:]
                        ratio_average_latencies = max(recent_average_latencies) / min(recent_average_latencies)
                        recent_average_latency_ratios.append(ratio_average_latencies)
                        if False:
                            print(f', Last {MOVING_AVERAGE_LEN} average latencies: {recent_average_latencies}')
                        print(f', ratio = {ratio_average_latencies:.4f}')
                        check_latencies(average_latencies[MOVING_AVERAGE_LEN-1:], recent_average_latency_ratios)
                    else:
                        print()
                    if not args.follow:
                        break
                else:
                    print(f'Found {len(inference_latencies)} inferences. Needs {args.num_inferences - len(inference_latencies)}')
            elif line.startswith('R'):
                try:
                    cur_recharge_time = int(line[1:])
                except ValueError:
                    print(f'Invalid line for recharging time: {repr(line)}')
                    continue
                # print(cur_recharge_time)

                if cur_recharge_time > EXTRA_POWER_ON_TIME:
                    cur_recharge_time -= EXTRA_POWER_ON_TIME
                else:
                    # cur_recharge_time = 0
                    short_recharging_cycles[-1] += 1

                accumulated_recharging_time += cur_recharge_time

        # print(f'Short recharging cycles: {short_recharging_cycles}')
        print('Inference latencies: ' + ' '.join(map(str, inference_latencies)))
        print('Active times: ' + ' '.join(map(str, active_times)))
        print('Power failures: ' + ' '.join(map(str, power_failures)))
        final_average_latency = statistics.mean(inference_latencies[-args.num_inferences:]) / 1000
        final_average_active_time = statistics.mean(active_times[-args.num_inferences:]) / 1000
        print(f'Average latency={final_average_latency:.4f} seconds')
        print(f'Average active time={final_average_active_time:.4f} seconds')
        print()

if __name__ == '__main__':
    main()
