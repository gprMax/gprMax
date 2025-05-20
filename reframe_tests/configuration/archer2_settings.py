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

site_configuration = {
    "general": [
        {
            # Necessary if using the --restore-session flag
            "keep_stage_files": True
        }
    ],
    "systems": [
        {
            "name": "archer2",
            "descr": "ARCHER2",
            "hostnames": ["uan", "ln", "dvn"],
            "modules_system": "lmod",
            "partitions": [
                {
                    "name": "login",
                    "descr": "Login nodes",
                    "scheduler": "local",
                    "launcher": "local",
                    "environs": ["PrgEnv-gnu", "PrgEnv-cray", "PrgEnv-aocc"],
                },
                {
                    "name": "compute",
                    "descr": "Compute nodes",
                    "scheduler": "slurm",
                    "launcher": "srun",
                    "access": [
                        "--hint=nomultithread",
                        "--distribution=block:block",
                        "--partition=standard",
                        "--qos=standard",
                    ],
                    "environs": ["PrgEnv-gnu", "PrgEnv-cray", "PrgEnv-aocc"],
                    "max_jobs": 16,
                    "processor": {
                        "num_cpus": 128,
                        "num_cpus_per_socket": 64,
                        "num_sockets": 2,
                    },
                },
            ],
        }
    ],
    "environments": [
        {
            "name": "PrgEnv-gnu",
            "modules": ["PrgEnv-gnu"],
            "cc": "cc",
            "cxx": "CC",
            "ftn": "ftn",
            "target_systems": ["archer2"],
        },
        {
            "name": "PrgEnv-cray",
            "modules": ["PrgEnv-cray"],
            "cc": "cc",
            "cxx": "CC",
            "ftn": "ftn",
            "target_systems": ["archer2"],
        },
        {
            "name": "PrgEnv-aocc",
            "modules": ["PrgEnv-aocc"],
            "cc": "cc",
            "cxx": "CC",
            "ftn": "ftn",
            "target_systems": ["archer2"],
        },
    ],
    "logging": [
        {
            "level": "debug",
            "handlers": [
                {"type": "stream", "name": "stdout", "level": "info", "format": "%(message)s"},
                {
                    "type": "file",
                    "name": "reframe.out",
                    "level": "info",
                    "format": "[%(asctime)s] %(check_info)s: %(message)s",
                    "append": True,
                },
                {
                    "type": "file",
                    "name": "reframe.log",
                    "level": "debug",
                    "format": "[%(asctime)s] %(levelname)s %(levelno)s: %(check_info)s: %(message)s",  # noqa: E501
                    "append": False,
                },
            ],
            "handlers_perflog": [
                {
                    "type": "file",
                    "name": "reframe_perf.out",
                    "level": "info",
                    "format": "[%(asctime)s] %(check_info)s job_id=%(check_jobid)s %(check_perfvalues)s",
                    "format_perfvars": "| %(check_perf_var)s: %(check_perf_value)s %(check_perf_unit)s (r: %(check_perf_ref)s l: %(check_perf_lower_thres)s u: %(check_perf_upper_thres)s) ",
                    "append": True,
                },
                {
                    "type": "filelog",
                    "prefix": "%(check_system)s/%(check_partition)s",
                    "level": "info",
                    "format": (
                        "%(check_result)s,%(check_job_completion_time)s,"
                        "%(check_info)s,%(check_jobid)s,"
                        "%(check_num_tasks)s,%(check_num_cpus_per_task)s,%(check_num_tasks_per_node)s,"
                        "%(check_perfvalues)s"
                    ),
                    "format_perfvars": (
                        "%(check_perf_value)s,%(check_perf_unit)s,"
                        "%(check_perf_ref)s,%(check_perf_lower_thres)s,"
                        "%(check_perf_upper_thres)s,"
                    ),
                    "append": True,
                },
                {
                    "type": "filelog",
                    "prefix": "%(check_system)s/%(check_partition)s/latest",
                    "level": "info",
                    "format": (
                        "%(check_result)s,%(check_job_completion_time)s,"
                        "%(check_info)s,%(check_jobid)s,"
                        "%(check_num_tasks)s,%(check_num_cpus_per_task)s,%(check_num_tasks_per_node)s,"
                        "%(check_perfvalues)s"
                    ),
                    "format_perfvars": (
                        "%(check_perf_value)s,%(check_perf_unit)s,"
                        "%(check_perf_ref)s,%(check_perf_lower_thres)s,"
                        "%(check_perf_upper_thres)s,"
                    ),
                    "append": False,
                },
            ],
        }
    ],
}
