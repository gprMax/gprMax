"""Undriven C4v_square PEC/copper modal-profile validation; run from the repository."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from testing.validation.waveguide_modal_profiles import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall', choices=('PEC', 'copper', 'both'), default='both')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    for wall in (('PEC', 'copper') if args.wall == 'both' else (args.wall,)):
        run(args.output, 'C4v_square', wall)
