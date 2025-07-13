import argparse
import contextlib
import datetime
import time
import statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename', required=True)
    parser.add_argument('--num-inferences', type=int, required=True)
    parser.add_argument('--follow', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    EXTRA_POWER_ON_TIME = 45

    with open(args.log_filename, 'r') as log_file, open(args.log_filename.replace('-raw.log', '.log'), 'w') if args.log_filename.endswith('-raw.log') else contextlib.nullcontext() as filtered_log_file:
        active_times = []
        inference_latencies = []
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
            datetime.datetime.strptime(timestamp.removeprefix('['), '%Y-%m-%d %H:%M:%S.%f')
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

                # add false positive latencies (ex: < 1000 ms) to the previous inference
                if cur_inference_latency >= 1000:
                    inference_latencies.append(cur_inference_latency)
                    active_times.append(cur_inference_latency - accumulated_recharging_time)
                    power_failures.append(power_failure)
                else:
                    inference_latencies[-1] += cur_inference_latency
                    active_times[-1] += cur_inference_latency - accumulated_recharging_time
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

                if filtered_log_file:
                    filtered_line = original_line.replace(' .', ' Latency=').replace(' PF=', ', PF=') + f', AT={active_times[-1]}'
                    filtered_log_file.write(filtered_line + '\n')

                if len(inference_latencies) >= args.num_inferences:
                    average_latency = statistics.mean(inference_latencies[-args.num_inferences:])
                    deviation_latency = statistics.stdev(inference_latencies[-args.num_inferences:])
                    cv_latency = deviation_latency / average_latency  # coefficient of variation
                    print(f'Found {len(inference_latencies)} inferences. Average latency for the last {args.num_inferences} = {average_latency}, coefficient of variation = {cv_latency}')
                    if cv_latency < 0.7 and not args.follow:
                        break
                else:
                    print(f'Found {len(inference_latencies)} inferences. Needs {args.num_inferences - len(inference_latencies)}')
            elif line.startswith('R'):
                cur_recharge_time = int(line[1:])
                # print(cur_recharge_time)

                # Fixing recharging time
                # Power on/off is detected by whether VBAT_OK is high or low (i.e., higher or lower than 1.5V)
                # However, the device remains on until VBAT_OK is roughly 1V. The gap between 1.5V and 1V is roughly 45 milliseconds.
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
        print(f'Average latency={statistics.mean(inference_latencies[-args.num_inferences:])}')
        print()

if __name__ == '__main__':
    main()
