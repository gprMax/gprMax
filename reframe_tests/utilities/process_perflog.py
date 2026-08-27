# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_parameter(row, parameter):
    value = re.search(f"\s%{parameter}=(?P<value>\S+)\s", row["info"])["value"]
    return value


def get_parameter_names(item):
    return re.findall(f"\s%(?P<name>\S+)=\S+", item)


columns_to_keep = [
    "num_tasks",
    "num_cpus_per_task",
    "num_tasks_per_node",
    "run_time_value",
    "simulation_time_value",
]

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        usage="cd gprMax/reframe_tests; python -m utilities.process_perflog inputfile [-o OUTPUT]",
        description="Extract perfvars from reframe perflog file.",
    )
    parser.add_argument("inputfile", help="name of input file including path")
    parser.add_argument("--output", "-o", help="name of output file including path", required=False)

    args = parser.parse_args()

    perflog = pd.read_csv(args.inputfile, index_col=False)

    # Extract recorded parameters and create a new column for them in the dataframe
    parameters = perflog["info"].agg(get_parameter_names).explode().unique()
    for parameter in parameters:
        perflog[parameter] = perflog.apply(get_parameter, args=[parameter], axis=1)

    # Organise dataframe
    columns_to_keep += parameters.tolist()
    columns_to_keep.sort()
    perflog = perflog[columns_to_keep].sort_values(columns_to_keep)
    perflog["simulation_time_value"] = perflog["simulation_time_value"].apply(round, args=[2])
    perflog = perflog.rename(
        columns={"simulation_time_value": "simulation_time", "run_time_value": "run_time"}
    )

    # Save output to file
    if args.output:
        outputfile = args.output
    else:
        stem = f"{Path(args.inputfile).stem}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
        outputfile = Path("benchmark_results", stem).with_suffix(".csv")
    perflog.to_csv(outputfile, index=False)
    print(f"Saved benchmark: '{outputfile}'")
