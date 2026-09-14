function outputFiles = gprmax_h5_to_mat(inputFiles, varargin)
%GPRMAX_H5_TO_MAT Convert gprMax HDF5 output to MATLAB v7.3 MAT files.
%
%   GPRMAX_H5_TO_MAT(FILE) creates FILE.mat and stores the converted output
%   in a variable derived from FILE's base name. This permits several files
%   to be loaded into one MATLAB workspace without overwriting one another.
%
%   GPRMAX_H5_TO_MAT(FILES) converts each member of a string or cell array
%   to a separate MAT file.
%
%   GPRMAX_H5_TO_MAT(FILES, "OutputFile", "study.mat") combines several
%   inputs in one structure. The top-level variable is derived from the MAT
%   filename and its runs field contains one member per input file.
%
%   Name-value options:
%       OutputFile   Destination for a single output or combined batch.
%       VariableName Explicit top-level MATLAB variable name.
%       Paths        HDF5 datasets/groups passed to gprmax_read_h5.
%       Overwrite    Permit replacing existing MAT files (default false).

parser = inputParser;
parser.FunctionName = mfilename;
addRequired(parser, 'inputFiles', ...
    @(x) ischar(x) || isstring(x) || iscellstr(x));
addParameter(parser, 'OutputFile', "", ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'VariableName', "", ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'Paths', strings(0, 1), ...
    @(x) ischar(x) || isstring(x) || iscellstr(x));
addParameter(parser, 'Overwrite', false, ...
    @(x) islogical(x) && isscalar(x));
parse(parser, inputFiles, varargin{:});

inputFiles = string(parser.Results.inputFiles);
inputFiles = inputFiles(:);
if isempty(inputFiles) || any(ismissing(inputFiles) | strlength(inputFiles) == 0)
    error('gprMax:MATLAB:InvalidInputFiles', ...
        'At least one non-empty HDF5 input filename is required.');
end
for index = 1:numel(inputFiles)
    if ~isfile(inputFiles(index))
        error('gprMax:MATLAB:HDF5FileNotFound', ...
            'The HDF5 file does not exist: %s', inputFiles(index));
    end
end

outputFile = string(parser.Results.OutputFile);
variableName = string(parser.Results.VariableName);
overwrite = parser.Results.Overwrite;
paths = parser.Results.Paths;

if numel(inputFiles) > 1 && strlength(variableName) > 0 ...
        && strlength(outputFile) == 0
    error('gprMax:MATLAB:AmbiguousVariableName', ...
        ['VariableName applies to the top-level variable of one MAT file. ' ...
         'Supply OutputFile to combine multiple inputs, or omit VariableName.']);
end

if strlength(outputFile) > 0
    outputFiles = convert_to_one_file( ...
        inputFiles, outputFile, variableName, paths, overwrite);
else
    outputFiles = strings(numel(inputFiles), 1);
    destinations = strings(numel(inputFiles), 1);
    storedNames = cell(numel(inputFiles), 1);
    usedNames = {};
    for index = 1:numel(inputFiles)
        [folder, stem] = fileparts(inputFiles(index));
        destinations(index) = fullfile(folder, stem + ".mat");
        if strlength(variableName) > 0
            storedNames{index} = valid_variable_name(variableName);
        else
            storedNames{index} = valid_variable_name(stem, usedNames);
        end
        usedNames{end + 1} = storedNames{index}; %#ok<AGROW>
    end
    if numel(unique(lower(destinations))) ~= numel(destinations)
        error('gprMax:MATLAB:DuplicateOutputFile', ...
            'The input list would write the same MAT destination more than once.');
    end
    if ~overwrite
        existing = destinations(isfile(destinations));
        if ~isempty(existing)
            error('gprMax:MATLAB:OutputExists', ...
                ['One or more output files already exist, beginning with: %s. ' ...
                 'Use Overwrite=true to replace them.'], existing(1));
        end
    end
    for index = 1:numel(inputFiles)
        destination = destinations(index);
        storedName = storedNames{index};
        result = gprmax_read_h5(inputFiles(index), 'Paths', paths);
        result.info.matlab_variable_name = string(storedName);
        save_named_variable(destination, storedName, result, overwrite);
        outputFiles(index) = string(destination);
    end
end

if isscalar(outputFiles)
    outputFiles = char(outputFiles);
end
end


function outputFile = convert_to_one_file(inputFiles, outputFile, ...
        variableName, paths, overwrite)
outputFile = ensure_mat_extension(outputFile);
[~, outputStem] = fileparts(outputFile);
if strlength(variableName) > 0
    storedName = valid_variable_name(variableName);
elseif isscalar(inputFiles)
    [~, inputStem] = fileparts(inputFiles);
    storedName = valid_variable_name(inputStem);
else
    storedName = valid_variable_name(outputStem);
end

if isscalar(inputFiles)
    result = gprmax_read_h5(inputFiles, 'Paths', paths);
    result.info.matlab_variable_name = string(storedName);
    save_named_variable(outputFile, storedName, result, overwrite);
    return
end

container = struct;
container.runs = struct;
container.name_map = struct( ...
    'source_file', {}, 'original_name', {}, 'matlab_name', {});
container.info = struct( ...
    'source_files', inputFiles, ...
    'matlab_variable_name', string(storedName), ...
    'converter', "gprmax_h5_to_mat");

usedNames = {};
for index = 1:numel(inputFiles)
    [~, stem] = fileparts(inputFiles(index));
    caseName = valid_variable_name(stem, usedNames);
    usedNames{end + 1} = caseName; %#ok<AGROW>
    result = gprmax_read_h5(inputFiles(index), 'Paths', paths);
    result.info.matlab_variable_name = string(caseName);
    container.runs.(caseName) = result;
    container.name_map(end + 1) = struct( ...
        'source_file', inputFiles(index), ...
        'original_name', string(stem), ...
        'matlab_name', string(caseName));
end

save_named_variable(outputFile, storedName, container, overwrite);
end


function save_named_variable(outputFile, variableName, value, overwrite)
outputFile = ensure_mat_extension(outputFile);
if isfile(outputFile) && ~overwrite
    error('gprMax:MATLAB:OutputExists', ...
        'The output file already exists: %s. Use Overwrite=true to replace it.', ...
        outputFile);
end

parent = fileparts(outputFile);
if ~isempty(parent) && ~isfolder(parent)
    error('gprMax:MATLAB:OutputFolderNotFound', ...
        'The output folder does not exist: %s', parent);
end

variables = struct;
variables.(variableName) = value;
save(char(outputFile), '-struct', 'variables', '-v7.3');
end


function outputFile = ensure_mat_extension(outputFile)
outputFile = string(outputFile);
[folder, stem, extension] = fileparts(outputFile);
if strlength(extension) == 0
    outputFile = fullfile(folder, stem + ".mat");
elseif ~strcmpi(extension, '.mat')
    error('gprMax:MATLAB:InvalidOutputExtension', ...
        'OutputFile must have a .mat extension: %s', outputFile);
end
end


function name = valid_variable_name(rawName, usedNames)
if nargin < 2
    usedNames = {};
end
if strlength(string(rawName)) == 0
    error('gprMax:MATLAB:InvalidVariableName', ...
        'VariableName cannot be empty when explicitly supplied.');
end
name = matlab.lang.makeValidName( ...
    char(string(rawName)), 'ReplacementStyle', 'hex');
name = matlab.lang.makeUniqueStrings(name, usedNames, namelengthmax);
end
