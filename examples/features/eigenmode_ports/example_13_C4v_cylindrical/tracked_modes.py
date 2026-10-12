"""Undriven circular PEC/copper modes on a C4v Cartesian staircase."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from testing.validation.cylindrical_modal_profiles import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall', choices=('PEC', 'copper', 'both'), default='both')
    parser.add_argument('--cells', type=int, nargs='+', default=[96, 192],
                        help='cells across the 24-mm modal window')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    for wall in (('PEC', 'copper') if args.wall == 'both' else (args.wall,)):
        run(args.output, wall, tuple(args.cells))
