site_configuration = {
    'systems': [
        {
            'name': 'archer2',
            'descr': 'ARCHER2',
            'hostnames': ['uan','ln','dvn'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'login',
                    'descr': 'Login nodes',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['PrgEnv-gnu','PrgEnv-cray','PrgEnv-aocc'],
                },
                {
                    'name': 'compute',
                    'descr': 'Compute nodes',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['--hint=nomultithread','--distribution=block:block','--partition=standard','--qos=standard'],
                    'environs': ['PrgEnv-gnu','PrgEnv-cray','PrgEnv-aocc'],
                    'max_jobs': 16,
                }
            ]
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['archer2']
        },
        {
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['archer2']
        },
        {
            'name': 'PrgEnv-aocc',
            'modules': ['PrgEnv-aocc'],
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn',
            'target_systems': ['archer2']
        },
    ],
    'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.out',
                    'level': 'info',
                    'format': '[%(asctime)s] %(check_info)s: %(message)s',
                    'append': True
                },
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s %(levelno)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'file',
                    'name': 'reframe_perf.out',
                    'level': 'info',
                    'format': '[%(asctime)s] %(check_info)s %(check_perfvalues)s', 
                    'format_perfvars': '| %(check_perf_var)s: %(check_perf_value)s %(check_perf_unit)s (r: %(check_perf_ref)s l: %(check_perf_lower_thres)s u: %(check_perf_upper_thres)s) ',
                    'append': True
                },
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': (
                        '%(check_result)s, %(check_job_completion_time)s, '
                        '%(check_name)s, %(check_short_name)s, %(check_jobid)s, '
                        '%(check_num_tasks)s, %(check_num_cpus_per_task)s, %(check_num_tasks_per_node)s, '
                        '%(check_#ALL)s'  # Any remaining loggable test attributes should be test parameters
                    ),
                    'ignore_keys': [
                        'check_build_locally',
                        'check_build_time_limit',
                        'check_descr',
                        'check_display_name',
                        'check_env_vars',
                        'check_exclusive_access',
                        'check_executable',
                        'check_executable_opts',
                        'check_extra_resources',
                        'check_hashcode',
                        'check_job_completion_time_unix',
                        'check_job_exitcode',
                        'check_job_nodelist',
                        'check_job_submit_time',
                        'check_jobid',
                        'check_keep_files',
                        'check_local',
                        'check_maintainers',
                        'check_max_pending_time',
                        'check_modules',
                        'check_name',
                        'check_num_cpus_per_task',
                        'check_num_gpus_per_node',
                        'check_num_tasks',
                        'check_num_tasks_per_core',
                        'check_num_tasks_per_node',
                        'check_num_tasks_per_socket',
                        'check_outputdir',
                        'check_partition',
                        'check_prebuild_cmds',
                        'check_prefix',
                        'check_prerun_cmds',
                        'check_postbuild_cmds',
                        'check_postrun_cmds',
                        'check_readonly_files',
                        'check_short_name',
                        'check_sourcepath',
                        'check_sourcesdir',
                        'check_stagedir',
                        'check_strict_check',
                        'check_system',
                        'check_tags',
                        'check_time_limit',
                        'check_unique_name',
                        'check_use_multithreading',
                        'check_valid_prog_environs',
                        'check_valid_systems',
                        'check_variables'
                    ],
                    'format_perfvars': (
                        '%(check_perf_value)s|%(check_perf_unit)s|'
                        '%(check_perf_ref)s|%(check_perf_lower_thres)s|'
                        '%(check_perf_upper_thres)s|'
                    ),
                    'append': True
                }
            ]
        }
    ],
}