# Copyright (C) 2015-2019: The University of Edinburgh
#                 Authors: Tobias Schruff
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

from mpi4py import MPI
from enum import IntEnum
import logging
import time

if __name__ == '__main__':
    _log = logging.getLogger('gprMax.mpi')
else:
    _log = logging.getLogger(__name__)


"""
MPI communication tags.

READY
    Send by worker to master to signal that it is ready to receive a new job.
START
    Send by master to worker together with a job dict to initiate work.
DONE
    Send by worker to master together with the results of the current job.
EXIT
    Send by master to worker to initiate worker shutdown and then
    send back to master to signal shutdown has completed.
"""
Tags = IntEnum('Tags', 'READY START DONE EXIT')


class MPIExecutor(object):
    """A generic parallel executor based on MPI.

    This executor can be used to run generic jobs on multiple
    processes based on a master/worker pattern with MPI being used for
    communication between the master and the workers.

    Examples
    --------

    A basic example of how to use the `MPIExecutor` to run
    `gprMax` models in parallel is given below.

    >>> from mpi4py import MPI
    >>> from gprMax.mpi import MPIExecutor
    >>> from gprMax.model_build_run import run_model
    >>> # choose an MPI.Intracomm for communication (MPI.COMM_WORLD by default)
    >>> comm = MPI.COMM_WORLD
    >>> # choose a target function
    >>> func = run_model
    >>> # define job parameters
    >>> inputfile = 'some_input_file.in'
    >>> n_traces = 10
    >>> jobs = []
    >>> # create jobs
    >>> for i in range(n_traces):
    >>>    jobs.append({
    >>>        'inputfile': inputfile,
    >>>        'currentmodelrun': i + 1,
    >>>        'modelend': n_traces,
    >>>        'numbermodelruns': n_traces
    >>>    })
    >>> gpr = MPIExecutor(func, comm=comm)
    >>> # send the workers to their work loop
    >>> gpr.start()
    >>> if gpr.is_master():
    >>>    results = gpr.submit(jobs)
    >>>    print('Results:', results)
    >>> # make the workers exit their work loop
    >>> # and join the main loop again
    >>> gpr.join()

    A slightly more concise way is to use the context manager
    interface of `MPIExecutor` that automatically takes care
    of calling `start()` and `join()` at the beginning and end
    of the execution, respectively.

    >>> with MPIExecutor(func, comm=comm) as executor:
    >>>    # executor will be None on all ranks except for the master
    >>>    if executor is not None:
    >>>        results = executor.submit(jobs)
    >>>        print('Results:', results)


    Limitations
    -----------
    Because some popular MPI implementations (especially on HPC machines) do not
    support concurrent MPI calls from multiple threads yet, the `MPIExecutor` does
    not use a separate thread in the master to do the communication between the
    master and the workers. Hence, the lowest thread level of MPI_THREAD_SINGLE
    (no multi-threading) is enough.

    However, this imposes some slight limitations on the usage since it is not
    possible to interact with the workers during a call to `submit()` until
    `submit()` returns.

    In particular, it is not possible to handle exceptions that occur on workers
    in the main loop. Instead all exceptions that occur on workers are caught and
    logged and the worker returns None instead of the actual result of the worker
    function. A second limitation is that it is not possible to terminate workers.

    If you need an MPI executor that supports custom exception handling, you should
    use a multi-threading implementation such as the `MPICommExecutor` in
    `mpi4py.futures`. Below is a brief example of how to use it with the example
    given above.

    >>> from mpi4py.futures import MPICommExecutor
    >>> from concurrent.futures import as_completed
    >>> # define comm, func, and jobs like above
    >>> with MPICommExecutor(comm, root=0) as executor:
    >>>     if executor is not None:
    >>>         futures = [executor.submit(func, **job) for job in jobs]
    >>>         for future in as_completed(futures):
    >>>             try:
    >>>                 print(future.result())
    >>>             except Exception as e:
    >>>                 # custom exception handling for exceptions
    >>>                 # raised in the worker
    >>>                 print(e)
    >>>                 comm.Abort()
    """

    def __init__(self, func, master=0, comm=None):
        """Initializes a new executor instance.

        Parameters
        ----------
        func: callable
            The worker function. Jobs will be passed as keyword arguments,
            so `func` must support this. This is usually the case, but
            can be a problem when builtin functions are used, e.g. `abs()`.
        master: int
            The rank of the master. Must be in `comm`. All other
            ranks in `comm` will be treated as workers.
        comm: MPI.Intracomm
            The MPI communicator used for communication between the
            master and workers.
        """
        if comm is None:
            self.comm = MPI.COMM_WORLD
        elif not comm.Is_intra():
            raise TypeError('MPI.Intracomm expected')
        else:
            self.comm = comm

        self.rank = self.comm.rank
        self.size = self.comm.size
        if self.size < 2:
            raise RuntimeError('MPIExecutor must run with at least 2 processes')

        self._up = False

        master = int(master)
        if master < 0:
            raise ValueError('master rank must be non-negative')
        elif master >= self.size:
            raise ValueError('master not in comm')
        else:
            self.master = master

        # the worker ranks
        self.workers = tuple(set(range(self.size)) - {self.master})
        # the worker function
        if not callable(func):
            raise TypeError('func must be a callable')
        self.func = func
        # holds the state of workers on the master
        self.busy = [False] * len(self.workers)

        _log.debug(f'MPIExecutor on comm: {self.comm.name}, Master: {self.master}, Workers: {self.workers}')
        if self.is_master():
            _log.debug('*** MASTER ***')
        else:
            _log.debug('*** WORKER ***')

    def __enter__(self):
        """Context manager enter.

        Only the master returns an executor,
        all other ranks return None.
        """
        self.start()
        if self.is_master():
            return self
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.
        """
        if exc_type is not None:
            _log.exception(exc_val)
            return False

        # no exception handling necessary
        # since we catch everything in __guarded_work
        # exc_type should always be None
        self.join()
        return True

    def is_idle(self):
        """Returns a bool indicating whether the executor is idle.

        The executor is considered to be not idle if *any* worker
        process is busy with a job. That means, it is idle only
        if *all* workers are idle.

        Note: This member must not be called on a worker.
        """
        assert self.is_master()
        return not any(self.busy)

    def is_master(self):
        """Returns a bool indicating whether `self` is the master.
        """
        return self.rank == self.master

    def is_worker(self):
        """Returns a bool indicating whether `self` is a worker.
        """
        return not self.is_master()

    def start(self):
        """Starts up workers.

        A check is performed on the master whether the executor
        has already been terminated, in which case a RuntimeError
        is raised on the master.
        """
        if self.is_master():
            if self._up:
                raise RuntimeError('start has already been called')
            self._up = True

        _log.info('Starting up MPIExecutor master/workers')
        if self.is_worker():
            self.__wait()

    def join(self):
        """Joins the workers.
        """
        if self.is_master():

            _log.debug('Terminating. Sending sentinel to all workers.')
            # send sentinel to all workers
            for worker in self.workers:
                self.comm.send(None, dest=worker, tag=Tags.EXIT)

            _log.debug('Waiting for all workers to terminate.')

            down = [False] * len(self.workers)
            while True:
                for i, worker in enumerate(self.workers):
                    if self.comm.Iprobe(source=worker, tag=Tags.EXIT):
                        self.comm.recv(source=worker, tag=Tags.EXIT)
                        down[i] = True
                if all(down):
                    break

            self._up = False
            _log.debug('All workers terminated.')

    def submit(self, jobs, sleep=0.0):
        """Submits a list of jobs to the workers and returns the results.

        Parameters
        ----------
        jobs: list
            A list of keyword argument dicts. Each dict describes
            a job and will be unpacked and supplied to the work function.
        sleep: float
            The number of seconds the master will sleep for when trying
            to find an idle worker. The default value is 0.0, which means
            the master will not sleep at all.

        Returns
        -------
        results: list
            A list of results, i.e. the return values of the work function,
            received from the workers. The order of results is identical to
            the order of `jobs`.
        """
        if not self._up:
            raise RuntimeError('cannot run jobs without a call to start()')

        _log.info('Running {:d} jobs.'.format(len(jobs)))
        assert self.is_master(), 'run() must not be called on a worker process'

        my_jobs = jobs.copy()
        num_jobs = len(my_jobs)
        results = [None] * num_jobs
        while len(my_jobs) or not self.is_idle():

            for i, worker in enumerate(self.workers):

                if self.comm.Iprobe(source=worker, tag=Tags.DONE):
                    job_idx, result = self.comm.recv(source=worker, tag=Tags.DONE)
                    _log.debug(f'Received finished job {job_idx} from worker {worker:d}.')
                    results[job_idx] = result
                    self.busy[i] = False
                elif self.comm.Iprobe(source=worker, tag=Tags.READY):
                    if len(my_jobs):
                        self.comm.recv(source=worker, tag=Tags.READY)
                        self.busy[i] = True
                        job_idx = num_jobs - len(my_jobs)
                        _log.debug(f'Sending job {job_idx} to worker {worker:d}.')
                        self.comm.send((job_idx, my_jobs.pop(0)), dest=worker, tag=Tags.START)
                elif self.comm.Iprobe(source=worker, tag=Tags.EXIT):
                    _log.debug(f'Worker on rank {worker:d} has terminated.')
                    self.comm.recv(source=worker, tag=Tags.EXIT)
                    self.busy[i] = False

            time.sleep(sleep)

        _log.info('Finished all jobs.')
        return results

    def __wait(self):
        """The worker main loop.

        The worker will enter the loop after `start()` has been called
        and stay here until it receives the sentinel, e.g. by calling
        `join()` on the master. In the mean time, the worker is
        accepting work.
        """
        assert self.is_worker()

        status = MPI.Status()

        _log.debug(f'Starting up worker.')

        while True:

            self.comm.send(None, dest=self.master, tag=Tags.READY)
            _log.debug(f'Worker on rank {self.rank} waiting for job.')

            data = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)
            tag = status.tag

            if tag == Tags.START:
                job_idx, work = data
                _log.debug(f'Received job {job_idx} (work={work}).')
                result = self.__guarded_work(work)
                _log.debug(f'Finished job. Sending results to master.')
                self.comm.send((job_idx, result), dest=self.master, tag=Tags.DONE)
            elif tag == Tags.EXIT:
                _log.debug(f'Received sentinel from master.')
                break

        _log.debug('Terminating worker.')
        self.comm.send(None, dest=self.master, tag=Tags.EXIT)

    def __guarded_work(self, work):
        """Executes work safely on the workers.

        Parameters
        ----------
        work: dict
            Keyword arguments that are unpacked and given to the
            work function.

        Notes
        -----
        All exceptions that occur in the work function `func` are caught
        and logged. The worker returns `None` to the master in that case
        instead of the actual result.
        """
        assert self.is_worker()
        try:
            return self.func(**work)
        except Exception as e:
            _log.exception(str(e))
            return None


def main(args=None):
    """CLI for gprMax in MPI mode.

    Example Usage:
        mpirun -np 4 python -m mpi -n 10 my_input_file.in
    """
    import argparse
    import os
    from gprMax.constants import c, e0, m0, z0
    from gprMax.model_build_run import run_model

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='gprMax', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'inputfile',
        help='relative or absolute path to inputfile.')
    parser.add_argument(
        '-n', '--num-traces', type=int, default=1,
        help='number of model runs (traces) to create a B-scan')
    parser.add_argument(
        '--geometry-only', action='store_true', default=False,
        help='flag to only build model and produce geometry file(s)')
    parser.add_argument(
        '--geometry-fixed', action='store_true', default=False,
        help='flag to not reprocess model geometry, e.g. for B-scans where the geometry is fixed')
    parser.add_argument(
        '--write-processed', action='store_true', default=False,
        help='flag to write an input file after any Python code and include commands '
             'in the original input file have been processed')
    parser.add_argument(
        '-r', '--restart', type=int, default=1,
        help='model number to restart from, e.g. when creating B-scan')
    parser.add_argument(
        '-l', '--logfile', action='store_true', default=False,
        help='flag to enable writing to a log file')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help="flag to increase output")
    parser.add_argument(
        '--gpu', type=int, action='append', nargs='*',
        help='flag to use Nvidia GPU or option to give list of device ID(s)')

    args = parser.parse_args(args)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # set-up logging
    logger = logging.getLogger('gprMax')
    level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(level)

    if args.logfile != "":
        mh = logging.FileHandler(f"log_{rank}.txt", mode='w')
        mh.setLevel(level)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
        mh.setFormatter(formatter)
        logger.addHandler(mh)

    namespace = {
        'c': c,
        'e0': e0,
        'm0': m0,
        'z0': z0,
        'number_model_runs': args.num_traces,
        'inputfile': os.path.abspath(args.inputfile)
    }

    model_args = argparse.Namespace(**{
        'geometry_only': args.geometry_only,
        'geometry_fixed': args.geometry_fixed,
        'write_processed': args.write_processed,
        'task': False,
        'restart': False,
        'gpu': args.gpu
    })

    # compile jobs
    jobs = []
    for i in range(args.num_traces):
        jobs.append({
            'args': model_args,
            'inputfile': args.inputfile,
            'currentmodelrun': i + 1,
            'modelend': args.num_traces,
            'numbermodelruns': args.num_traces,
            'usernamespace': namespace.copy()
        })

    # execute jobs
    logger.info(f'Starting execution of {args.num_traces} gprMax model runs.')
    with MPIExecutor(run_model, comm=comm) as gpr:
        if gpr is not None:
            results = gpr.submit(jobs)
            logger.info('Results: %s' % str(results))
    logger.info('Finished.')


if __name__ == '__main__':
    main()
