% outputfiles_merge.m
% Script to merge gprMax output files of A-scans (traces) into a single
% HDF5 file
%
% Craig Warren

clear all, clc

[filenames, pathname] = uigetfile('*.out', 'Select gprMax A-scan output files to merge', 'MultiSelect', 'on');

% Combine A-scans (traces) into a single HDF5 output file
if filenames{1} ~= 0
    [pathstr, basefilename, ext] = fileparts(filenames{1});
    outputfile = strcat(fullfile(pathname, basefilename(1:end-1)), '_all.out');
    modelruns = length(filenames);
    filename = fullfile(pathname, filenames{1});
    iterations = double(h5readatt(filename, '/', 'Iterations'));
    dt = h5readatt(filename, '/', 'dt');
    h5create(outputfile, '/rxs/rx1/Ex', [modelruns iterations]);
    h5create(outputfile, '/rxs/rx1/Ey', [modelruns iterations]);
    h5create(outputfile, '/rxs/rx1/Ez', [modelruns iterations]);
    h5create(outputfile, '/rxs/rx1/Hx', [modelruns iterations]);
    h5create(outputfile, '/rxs/rx1/Hy', [modelruns iterations]);
    h5create(outputfile, '/rxs/rx1/Hz', [modelruns iterations]);
    h5writeatt(outputfile, '/', 'Iterations', iterations);
    h5writeatt(outputfile, '/', 'dt', dt);
    Ex = zeros(iterations, modelruns);
    Ey = zeros(iterations, modelruns);
    Ez = zeros(iterations, modelruns);
    Hx = zeros(iterations, modelruns);
    Hy = zeros(iterations, modelruns);
    Hz = zeros(iterations, modelruns);
    for rx=1:modelruns
        filename = fullfile(pathname, filenames{rx});
        Ex(:, rx) = h5read(filename, '/rxs/rx1/Ex');
        Ey(:, rx) = h5read(filename, '/rxs/rx1/Ey');
        Ez(:, rx) = h5read(filename, '/rxs/rx1/Ez');
        Hx(:, rx) = h5read(filename, '/rxs/rx1/Hx');
        Hy(:, rx) = h5read(filename, '/rxs/rx1/Hy');
        Hz(:, rx) = h5read(filename, '/rxs/rx1/Hz');
    end
    h5write(outputfile, '/rxs/rx1/Ex', Ex');
    h5write(outputfile, '/rxs/rx1/Ey', Ey');
    h5write(outputfile, '/rxs/rx1/Ez', Ez');
    h5write(outputfile, '/rxs/rx1/Hx', Hx');
    h5write(outputfile, '/rxs/rx1/Hy', Hy');
    h5write(outputfile, '/rxs/rx1/Hz', Hz');
    prompt = 'Do you want to remove the multiple individual output files? [y] or n: ';
    check = input(prompt,'s');
    if isempty(check) || check == 'y'
        for f=1:length(filenames)
            filename = fullfile(pathname, filenames{f});
            delete(filename);
        end
    end
end