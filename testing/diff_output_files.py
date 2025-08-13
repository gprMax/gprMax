# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, 
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def diff_output_files(filename1, filename2):
    """Calculates differences between two output files.

    Args:
        filename1: string of filename (including path) of output file 1.
        filename2: string of filename (including path) of output file 2.

    Returns:
        time: numpy array containing time.
        datadiffs: numpy array containing power (dB) of differences.
    """

    file1 = h5py.File(Path(filename1), "r")
    file2 = h5py.File(Path(filename2), "r")
    # Path to receivers in files
    path = "rxs/rx1/"

    # Get available field output component names
    outputs1 = list(file1[path].keys())
    outputs2 = list(file2[path].keys())
    if outputs1 != outputs2:
        logger.exception("Field output components are not the same in each file")
        raise ValueError

    # Check that type of float used to store fields matches
    floattype1 = file1[path + outputs1[0]].dtype
    floattype2 = file2[path + outputs2[0]].dtype
    if floattype1 != floattype2:
        logger.warning(
            f"Type of floating point number in test model ({file1[path + outputs1[0]].dtype}) "
            f"does not match type in reference solution ({file2[path + outputs2[0]].dtype})\n"
        )

    # Arrays for storing time
    time1 = np.zeros((file1.attrs["Iterations"]), dtype=floattype1)
    time1 = np.linspace(0, (file1.attrs["Iterations"] - 1), num=file1.attrs["Iterations"])
    time2 = np.zeros((file2.attrs["Iterations"]), dtype=floattype2)
    time2 = np.linspace(0, (file2.attrs["Iterations"] - 1), num=file2.attrs["Iterations"])

    # Arrays for storing field data
    data1 = np.zeros((file1.attrs["Iterations"], len(outputs1)), dtype=floattype1)
    data2 = np.zeros((file2.attrs["Iterations"], len(outputs2)), dtype=floattype2)
    for ID, name in enumerate(outputs1):
        data1[:, ID] = file1[path + str(name)][:]
        data2[:, ID] = file2[path + str(name)][:]
        if np.any(np.isnan(data1[:, ID])) or np.any(np.isnan(data2[:, ID])):
            logger.exception("Data contains NaNs")
            raise ValueError

    file1.close()
    file2.close()

    # Diffs
    datadiffs = np.zeros(data1.shape, dtype=np.float64)
    for i in range(len(outputs2)):
        maxi = np.amax(np.abs(data1[:, i]))
        datadiffs[:, i] = np.divide(
            np.abs(data2[:, i] - data1[:, i]),
            maxi,
            out=np.zeros_like(data1[:, i]),
            where=maxi != 0,
        )  # Replace any division by zero with zero

        # Calculate power (ignore warning from taking a log of any zero values)
        with np.errstate(divide="ignore"):
            datadiffs[:, i] = 20 * np.log10(datadiffs[:, i])
        # Replace any NaNs or Infs from zero division
        datadiffs[:, i][np.invert(np.isfinite(datadiffs[:, i]))] = 0

    return time1, datadiffs


def analyze_differences(time, datadiffs, outputs, tolerance_db=-20):
    """Analyze differences between two output files and provide detailed summary.
    
    Args:
        time: numpy array containing time values.
        datadiffs: numpy array containing power (dB) of differences.
        outputs: list of field component names.
        tolerance_db: threshold in dB below which differences are considered acceptable.
    
    Returns:
        analysis_results: dictionary containing detailed analysis results.
    """
    
    analysis_results = {
        'summary': {},
        'components': {},
        'overall_status': 'PASS',
        'tolerance_db': tolerance_db
    }
    
    print("="*80)
    print("DETAILED COMPARISON ANALYSIS")
    print("="*80)
    
    total_points = len(time)
    components_passed = 0
    
    for i, component in enumerate(outputs):
        diff_data = datadiffs[:, i]
        
        # Filter out -inf values (perfect matches)
        finite_diffs = diff_data[np.isfinite(diff_data)]
        
        if len(finite_diffs) == 0:
            # All values are identical (perfect match)
            max_diff = -np.inf
            min_diff = -np.inf
            avg_diff = -np.inf
            rms_diff = -np.inf
            points_above_tolerance = 0
        else:
            max_diff = np.max(finite_diffs)
            min_diff = np.min(finite_diffs)
            avg_diff = np.mean(finite_diffs)
            rms_diff = np.sqrt(np.mean(finite_diffs**2))
            points_above_tolerance = np.sum(finite_diffs > tolerance_db)
        
        # Determine pass/fail for this component
        component_status = 'PASS' if max_diff <= tolerance_db else 'FAIL'
        if component_status == 'PASS':
            components_passed += 1
        else:
            analysis_results['overall_status'] = 'FAIL'
        
        # Store component analysis
        analysis_results['components'][component] = {
            'max_diff_db': max_diff,
            'min_diff_db': min_diff,
            'avg_diff_db': avg_diff,
            'rms_diff_db': rms_diff,
            'points_above_tolerance': points_above_tolerance,
            'total_points': total_points,
            'status': component_status
        }
        
        # Print component analysis
        print(f"\n{component} Component Analysis:")
        print(f"  Status: {'✅ PASS' if component_status == 'PASS' else '❌ FAIL'}")
        if max_diff == -np.inf:
            print(f"  Perfect match: All values identical")
        else:
            print(f"  Max difference: {max_diff:.2f} dB")
            print(f"  Min difference: {min_diff:.2f} dB")
            print(f"  Avg difference: {avg_diff:.2f} dB")
            print(f"  RMS difference: {rms_diff:.2f} dB")
            print(f"  Points above tolerance ({tolerance_db} dB): {points_above_tolerance}/{total_points}")
        
        # Show some sample time points where significant differences occur
        if len(finite_diffs) > 0 and max_diff > tolerance_db:
            worst_indices = np.where(diff_data > tolerance_db)[0]
            if len(worst_indices) > 0:
                print(f"  Worst difference time indices: {worst_indices[:5]}...")  # Show first 5
    
    # Overall summary
    analysis_results['summary'] = {
        'total_components': len(outputs),
        'components_passed': components_passed,
        'components_failed': len(outputs) - components_passed,
        'pass_rate': components_passed / len(outputs) * 100,
        'total_time_points': total_points
    }
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Overall Status: {'✅ PASS' if analysis_results['overall_status'] == 'PASS' else '❌ FAIL'}")
    print(f"Components passed: {components_passed}/{len(outputs)} ({analysis_results['summary']['pass_rate']:.1f}%)")
    print(f"Total time points analyzed: {total_points}")
    print(f"Tolerance threshold: {tolerance_db} dB")
    
    if analysis_results['overall_status'] == 'PASS':
        print("\n🎉 SUCCESS: All field components match within tolerance!")
        print("The two simulation outputs are equivalent.")
    else:
        print("\n⚠️  WARNING: Some field components exceed tolerance!")
        print("The simulation outputs have significant differences.")
    
    return analysis_results


def compare_file_metadata(file1, file2):
    """Compare metadata between two HDF5 files.
    
    Args:
        file1: h5py.File object for first file.
        file2: h5py.File object for second file.
    
    Returns:
        metadata_match: boolean indicating if metadata matches.
    """
    
    print("\n" + "="*50)
    print("METADATA COMPARISON")
    print("="*50)
    
    metadata_match = True
    
    # Compare key attributes that are commonly present
    attrs_to_check = ['Iterations', 'dt']
    
    for attr in attrs_to_check:
        if attr in file1.attrs and attr in file2.attrs:
            val1 = file1.attrs[attr]
            val2 = file2.attrs[attr]
            match = np.allclose(val1, val2) if isinstance(val1, (int, float)) else val1 == val2
            status = "✅" if match else "❌"
            print(f"{attr}: {val1} vs {val2} {status}")
            if not match:
                metadata_match = False
        else:
            print(f"{attr}: Missing in one or both files ❌")
            metadata_match = False
    
    return metadata_match


def main():
    """Main function for command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare two gprMax output files and provide detailed analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diff_output_files.py file1.h5 file2.h5
  python diff_output_files.py file1.h5 file2.h5 --tolerance -50
  python diff_output_files.py file1.h5 file2.h5 --verbose
        """
    )
    
    parser.add_argument('file1', help='First HDF5 output file')
    parser.add_argument('file2', help='Second HDF5 output file')
    parser.add_argument('--tolerance', type=float, default=-40, 
                       help='Tolerance threshold in dB (default: -40)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.file1).exists():
        print(f"❌ Error: File {args.file1} does not exist")
        return 1
    
    if not Path(args.file2).exists():
        print(f"❌ Error: File {args.file2} does not exist")
        return 1
    
    print("🔍 gprMax Output File Comparison Tool")
    print("="*80)
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"Tolerance: {args.tolerance} dB")
    
    try:
        # Compare metadata first
        with h5py.File(Path(args.file1), "r") as f1, h5py.File(Path(args.file2), "r") as f2:
            metadata_match = compare_file_metadata(f1, f2)
        
        # Calculate differences
        time, datadiffs = diff_output_files(args.file1, args.file2)
        
        # Get field component names
        with h5py.File(Path(args.file1), "r") as f1:
            outputs = list(f1["rxs/rx1/"].keys())
        
        # Analyze differences
        analysis_results = analyze_differences(time, datadiffs, outputs, args.tolerance)
        
        # Print final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        
        if metadata_match and analysis_results['overall_status'] == 'PASS':
            print("🎉 OVERALL RESULT: FILES MATCH ✅")
            print("Both metadata and field data are equivalent within tolerance.")
            return 0
        elif not metadata_match:
            print("⚠️  OVERALL RESULT: METADATA MISMATCH ❌")
            print("Files have different metadata parameters.")
            return 1
        else:
            print("⚠️  OVERALL RESULT: FIELD DATA MISMATCH ❌")
            print("Files have significant differences in field data.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
