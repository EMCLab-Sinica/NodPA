import argparse
import logging
import pathlib
import random
import signal
import tempfile
import sys
from subprocess import Popen, TimeoutExpired, check_call

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('run-intermittently')

CHUNK_SIZE = 2000
CHUNK_LINES = 20

def run_one_inference(program, interval, logfile, shutdown_after_writes: tuple[int, int], power_cycles_limit, n_samples: int) -> int:
    num_power_cycles = 0
    timeout_counter = 0
    shutdown_after_writes_lower, shutdown_after_writes_upper = shutdown_after_writes

    while True:
        shutdown_args = []
        if shutdown_after_writes_upper == 0:
            # Only shutdown once
            if num_power_cycles == 0 and shutdown_after_writes_lower:
                shutdown_args = ['-c', str(shutdown_after_writes_lower)]
        else:
            if shutdown_after_writes_lower:
                shutdown_args = ['-c', str(random.randint(shutdown_after_writes_lower, shutdown_after_writes_upper))]

        cmd = [program, str(n_samples)] + shutdown_args

        with Popen(cmd, stdout=logfile, stderr=logfile) as proc:
            try:
                kwargs = {}
                if not shutdown_args:
                    kwargs['timeout'] = interval
                outs, errs = proc.communicate(**kwargs)
            except TimeoutExpired:
                timeout_counter += 1
                if power_cycles_limit > 0 and timeout_counter >= power_cycles_limit:
                    logger.error('The program does not run intermittently')
                    return 3
                proc.send_signal(signal.SIGINT)
            proc.wait()
            num_power_cycles += 1
            if proc.returncode in (2, -signal.SIGINT):
                # simulated power failure
                continue

            logfile.seek(0, 2)
            file_size = logfile.tell()
            logfile.seek(-(CHUNK_SIZE if file_size > CHUNK_SIZE else file_size), 2)
            last_chunk = logfile.read()
            print('\n'.join(last_chunk.decode('ascii').split('\n')[-CHUNK_LINES:]))

            print(f'Number of power cycles: {num_power_cycles}')

            if proc.returncode in (1, -signal.SIGFPE, -signal.SIGSEGV, -signal.SIGBUS):
                logger.error('Program crashed!')
                return 2
            if proc.returncode == 0:
                return 0

def main():
    def parse_shutdown_after_writes(arg: str) -> tuple[int, int]:
        ret = arg.split(',', maxsplit=1)
        if len(ret) == 1:
            ret.append(0)
        return tuple(map(int, ret))

    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=0)
    parser.add_argument('--interval', type=float, default=0.02)
    parser.add_argument('--shutdown-after-writes', type=parse_shutdown_after_writes, default=[0, 0])
    parser.add_argument('--power-cycles-limit', type=int, default=200)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--compress', default=False, action='store_true')
    parser.add_argument('--n-samples', default=1, type=int)
    parser.add_argument('program')
    args = parser.parse_args()

    rounds = 0
    ret = 0
    logdir = pathlib.Path(tempfile.gettempdir())
    if args.compress:
        if args.suffix:
            log_archive = logdir / f'intermittent-cnn-{args.suffix}.tar'
        else:
            log_archive = logdir / 'intermittent-cnn.tar'
        check_call(['rm', '-vf', log_archive])

    while True:
        if args.suffix:
            logfile_path = logdir / f'intermittent-cnn-{args.suffix}-{rounds}'
        else:
            logfile_path = logdir / f'intermittent-cnn-{rounds}'
        compressed_logfile_path = logfile_path.with_suffix('.zst')
        with open(logfile_path, mode='w+b') as logfile:
            ret = run_one_inference(args.program, args.interval, logfile, args.shutdown_after_writes, args.power_cycles_limit, args.n_samples)

        if args.compress:
            check_call(['touch', log_archive])
            check_call(['zstd', logfile_path, '-o', compressed_logfile_path, '-f'])
            check_call(['tar', '-C', logdir, '-uvf', log_archive, compressed_logfile_path.name])
            check_call(['rm', '-v', logfile_path])
            check_call(['rm', '-v', compressed_logfile_path])

        if ret != 0:
            break
        rounds += 1
        if args.rounds and rounds >= args.rounds:
            break
    return ret

if __name__ == '__main__':
    sys.exit(main())
