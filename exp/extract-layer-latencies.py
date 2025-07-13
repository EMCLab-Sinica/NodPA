import datetime
import re
import sys

def main() -> None:
    log_filename = sys.argv[1]
    timing_filename = log_filename.replace('log_', 'timing_')

    previous_timestamp = 0.0

    with open(log_filename) as log_file, open(timing_filename, 'w') as timing_file:
        last_layer_idx = -1

        for line in log_file:
            line = line.strip()
            mobj = re.match(r'\[(.*)\] L(\d+)', line)
            if mobj is None:
                continue
            timestamp, layer_idx = mobj.groups()
            timestamp = datetime.datetime.strptime(timestamp + '000', '%Y-%m-%d %H:%M:%S.%f').timestamp()
            layer_idx = int(layer_idx)

            if previous_timestamp:
                for idx in range(last_layer_idx + 1, layer_idx):
                    print(f'Layer {idx-1}, time=0', file=timing_file)

                delta = timestamp - previous_timestamp
                # The signal is sent after layer_idx is incremented; see cnn_common.cpp
                print(f'Layer {layer_idx-1}, time={delta}', file=timing_file)

                last_layer_idx = layer_idx

            previous_timestamp = timestamp

if __name__ == '__main__':
    main()
