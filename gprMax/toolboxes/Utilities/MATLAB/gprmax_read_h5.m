function result = gprmax_read_h5(filename, varargin)
%GPRMAX_READ_H5 Read a gprMax HDF5 output into a MATLAB structure.
%
%   G = GPRMAX_READ_H5(FILENAME) recursively reads the complete HDF5
%   hierarchy. Numeric precision, integer arrays, strings, attributes, and
%   complex datasets are retained. Multidimensional datasets are returned
%   in the dimension order documented by gprMax rather than the reversed
%   order used by MATLAB's HDF5 interface.
%
%   G = GPRMAX_READ_H5(FILENAME, "Paths", PATHS) reads only the selected
%   HDF5 datasets or groups. PATHS may be a string array, character vector,
%   or cell array of character vectors, for example ["/rxs", "/ports"].
%
%   HDF5 names are converted to valid, locally unique MATLAB field names.
%   G.name_map records the reversible mapping between every original HDF5
%   path and its MATLAB path. G.metadata retains attributes for every group
%   and dataset that was read.

parser = inputParser;
parser.FunctionName = mfilename;
addRequired(parser, 'filename', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'Paths', strings(0, 1), ...
    @(x) ischar(x) || isstring(x) || iscellstr(x));
parse(parser, filename, varargin{:});

filename = char(string(parser.Results.filename));
if ~isfile(filename)
    error('gprMax:MATLAB:HDF5FileNotFound', ...
        'The HDF5 file does not exist: %s', filename);
end

selectedPaths = normalise_paths(parser.Results.Paths);
for index = 1:numel(selectedPaths)
    try
        h5info(filename, char(selectedPaths(index)));
    catch exception
        error('gprMax:MATLAB:HDF5PathNotFound', ...
            'The selected HDF5 path %s does not exist in %s.\n%s', ...
            selectedPaths(index), filename, exception.message);
    end
end

rootInfo = h5info(filename, '/');
[header, headerNameMap] = attributes_as_header(filename, '/', rootInfo.Attributes);
[data, metadata, nameMap] = read_group( ...
    filename, rootInfo, 'data', selectedPaths);

result = struct;
result.header = header;
result.data = data;
result.metadata = metadata;
result.name_map = nameMap;
result.header_name_map = headerNameMap;
result.info = struct( ...
    'source_file', string(filename), ...
    'selected_paths', selectedPaths(:), ...
    'dimension_order', "gprMax/HDF5", ...
    'reader', "gprmax_read_h5");
end


function [data, metadata, nameMap] = read_group(filename, groupInfo, ...
        matlabPath, selectedPaths)
data = struct;
metadata = metadata_entry(filename, groupInfo.Name, matlabPath, ...
    'group', groupInfo.Attributes, [], []);
nameMap = name_map_entry(groupInfo.Name, matlabPath, 'group');
usedNames = {};

for index = 1:numel(groupInfo.Datasets)
    datasetInfo = groupInfo.Datasets(index);
    hdf5Path = join_hdf5_path(groupInfo.Name, datasetInfo.Name);
    if ~dataset_selected(hdf5Path, selectedPaths)
        continue
    end

    matlabName = unique_matlab_name(datasetInfo.Name, usedNames);
    usedNames{end + 1} = matlabName; %#ok<AGROW>
    datasetMatlabPath = [matlabPath '.' matlabName];
    value = h5read(filename, hdf5Path);
    value = convert_complex(value);
    value = convert_enum(value, datasetInfo.Datatype);
    value = restore_dimension_order(value, datasetInfo.Dataspace);
    data.(matlabName) = value;

    metadata(end + 1) = metadata_entry( ...
        filename, hdf5Path, datasetMatlabPath, 'dataset', ...
        datasetInfo.Attributes, datasetInfo.Dataspace, value); %#ok<AGROW>
    nameMap(end + 1) = name_map_entry( ...
        hdf5Path, datasetMatlabPath, 'dataset'); %#ok<AGROW>
end

for index = 1:numel(groupInfo.Groups)
    childInfo = groupInfo.Groups(index);
    hdf5Path = childInfo.Name;
    if ~group_selected(hdf5Path, selectedPaths)
        continue
    end

    rawName = hdf5_leaf_name(hdf5Path);
    matlabName = unique_matlab_name(rawName, usedNames);
    usedNames{end + 1} = matlabName; %#ok<AGROW>
    childMatlabPath = [matlabPath '.' matlabName];
    [childData, childMetadata, childNameMap] = read_group( ...
        filename, childInfo, childMatlabPath, selectedPaths);
    data.(matlabName) = childData;
    metadata = [metadata childMetadata]; %#ok<AGROW>
    nameMap = [nameMap childNameMap]; %#ok<AGROW>
end
end


function entry = metadata_entry(filename, hdf5Path, matlabPath, kind, ...
        attributeInfo, dataspace, value)
entry = struct( ...
    'hdf5_path', string(hdf5Path), ...
    'matlab_path', string(matlabPath), ...
    'kind', string(kind), ...
    'attributes', read_attributes(filename, hdf5Path, attributeInfo), ...
    'hdf5_size', zeros(1, 0), ...
    'matlab_size', zeros(1, 0), ...
    'matlab_class', "", ...
    'is_complex', false);

if strcmp(kind, 'dataset')
    if isfield(dataspace, 'Size')
        entry.hdf5_size = double(dataspace.Size);
    end
    entry.matlab_size = double(size(value));
    entry.matlab_class = string(class(value));
    entry.is_complex = ~isreal(value);
end
end


function entry = name_map_entry(hdf5Path, matlabPath, kind)
entry = struct( ...
    'hdf5_path', string(hdf5Path), ...
    'matlab_path', string(matlabPath), ...
    'kind', string(kind));
end


function attributes = read_attributes(filename, objectPath, attributeInfo)
attributes = struct('name', {}, 'matlab_name', {}, 'value', {});
usedNames = {};
for index = 1:numel(attributeInfo)
    rawName = attributeInfo(index).Name;
    matlabName = unique_matlab_name(rawName, usedNames);
    usedNames{end + 1} = matlabName; %#ok<AGROW>
    attributes(end + 1) = struct( ...
        'name', string(rawName), ...
        'matlab_name', string(matlabName), ...
        'value', read_attribute_value( ...
            filename, objectPath, attributeInfo(index))); %#ok<AGROW>
end
end


function [header, nameMap] = attributes_as_header(filename, objectPath, ...
        attributeInfo)
header = struct;
nameMap = struct('hdf5_name', {}, 'matlab_name', {});
usedNames = {};
for index = 1:numel(attributeInfo)
    rawName = attributeInfo(index).Name;
    matlabName = unique_matlab_name(rawName, usedNames);
    usedNames{end + 1} = matlabName; %#ok<AGROW>
    header.(matlabName) = read_attribute_value( ...
        filename, objectPath, attributeInfo(index));
    nameMap(end + 1) = struct( ...
        'hdf5_name', string(rawName), ...
        'matlab_name', string(matlabName)); %#ok<AGROW>
end
end


function value = read_attribute_value(filename, objectPath, attributeInfo)
% MATLAB's h5readatt raises an HDF5 error for zero-length attributes. The
% value reported by h5info is authoritative in that case; a typed empty is
% used where the HDF datatype provides enough information.
if isempty(attributeInfo.Value)
    value = typed_empty_attribute(attributeInfo.Datatype);
    return
end

value = h5readatt(filename, objectPath, attributeInfo.Name);
value = convert_complex(value);
value = convert_enum(value, attributeInfo.Datatype);
if isfield(attributeInfo, 'Dataspace')
    value = restore_dimension_order(value, attributeInfo.Dataspace);
end
end


function value = typed_empty_attribute(datatype)
switch datatype.Class
    case 'H5T_STRING'
        value = strings(0, 1);
    case 'H5T_FLOAT'
        if datatype.Size == 4
            value = zeros(0, 1, 'single');
        else
            value = zeros(0, 1, 'double');
        end
    case 'H5T_INTEGER'
        typeName = string(datatype.Type);
        isUnsigned = contains(typeName, "_U");
        if datatype.Size == 1
            if isUnsigned
                value = zeros(0, 1, 'uint8');
            else
                value = zeros(0, 1, 'int8');
            end
        elseif datatype.Size == 2
            if isUnsigned
                value = zeros(0, 1, 'uint16');
            else
                value = zeros(0, 1, 'int16');
            end
        elseif datatype.Size == 4
            if isUnsigned
                value = zeros(0, 1, 'uint32');
            else
                value = zeros(0, 1, 'int32');
            end
        else
            if isUnsigned
                value = zeros(0, 1, 'uint64');
            else
                value = zeros(0, 1, 'int64');
            end
        end
    case 'H5T_ENUM'
        value = false(0, 1);
    otherwise
        value = [];
end
end


function value = convert_enum(value, datatype)
if ~strcmp(datatype.Class, 'H5T_ENUM') || isempty(value)
    return
end

if iscell(value) || isstring(value) || ischar(value)
    labels = string(value);
    normalised = upper(strtrim(labels));
    if all(normalised(:) == "TRUE" | normalised(:) == "FALSE")
        value = normalised == "TRUE";
    end
end
end


function value = convert_complex(value)
if ~isstruct(value) || ~isscalar(value)
    return
end

names = sort(string(fieldnames(value)));
if isequal(names, ["i"; "r"]) && isnumeric(value.r) && isnumeric(value.i) ...
        && isequal(size(value.r), size(value.i))
    value = complex(value.r, value.i);
end
end


function value = restore_dimension_order(value, dataspace)
if ~isfield(dataspace, 'Size')
    return
end

rank = numel(dataspace.Size);
if rank >= 2 && ~isscalar(value)
    value = permute(value, rank:-1:1);
end
end


function selectedPaths = normalise_paths(paths)
if isempty(paths)
    selectedPaths = strings(0, 1);
    return
end

selectedPaths = string(paths);
selectedPaths = selectedPaths(:);
if any(ismissing(selectedPaths) | strlength(strtrim(selectedPaths)) == 0)
    error('gprMax:MATLAB:InvalidHDF5Path', ...
        'Paths cannot contain missing or empty values.');
end

selectedPaths = strtrim(selectedPaths);
for index = 1:numel(selectedPaths)
    path = replace(selectedPaths(index), "\\", "/");
    if ~startsWith(path, "/")
        path = "/" + path;
    end
    while strlength(path) > 1 && endsWith(path, "/")
        path = extractBefore(path, strlength(path));
    end
    selectedPaths(index) = path;
end
selectedPaths = unique(selectedPaths, 'stable');
end


function selected = dataset_selected(path, selectedPaths)
if isempty(selectedPaths)
    selected = true;
    return
end

path = string(path);
selected = any(selectedPaths == "/" | selectedPaths == path | ...
    startsWith(path, selectedPaths + "/"));
end


function selected = group_selected(path, selectedPaths)
if isempty(selectedPaths)
    selected = true;
    return
end

path = string(path);
selected = any(selectedPaths == "/" | selectedPaths == path | ...
    startsWith(path, selectedPaths + "/") | ...
    startsWith(selectedPaths, path + "/"));
end


function path = join_hdf5_path(groupPath, name)
if strcmp(groupPath, '/')
    path = ['/' name];
else
    path = [groupPath '/' name];
end
end


function name = hdf5_leaf_name(path)
parts = split(string(path), "/");
name = char(parts(end));
end


function name = unique_matlab_name(rawName, usedNames)
name = matlab.lang.makeValidName(char(rawName), 'ReplacementStyle', 'hex');
name = matlab.lang.makeUniqueStrings(name, usedNames, namelengthmax);
end
