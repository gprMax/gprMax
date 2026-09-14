function [value, metadata] = gprmax_h5_get(result, hdf5Path)
%GPRMAX_H5_GET Resolve an original HDF5 path in gprmax_read_h5 output.
%
%   VALUE = GPRMAX_H5_GET(RESULT, PATH) returns the group or dataset at the
%   original HDF5 PATH, including paths whose names had to be changed to
%   become valid MATLAB structure fields.
%
%   [VALUE, METADATA] also returns the corresponding metadata entry.

if ~isstruct(result) || ~isfield(result, 'data') || ~isfield(result, 'name_map')
    error('gprMax:MATLAB:InvalidReaderResult', ...
        'The first argument must be a structure returned by gprmax_read_h5.');
end

hdf5Path = normalise_path(hdf5Path);
paths = string({result.name_map.hdf5_path});
index = find(paths == hdf5Path, 1);
if isempty(index)
    error('gprMax:MATLAB:HDF5PathNotLoaded', ...
        ['The HDF5 path %s is not present. It may not have been included ' ...
         'in the Paths selection passed to gprmax_read_h5.'], hdf5Path);
end

parts = split(result.name_map(index).matlab_path, '.');
value = result;
for partIndex = 1:numel(parts)
    value = value.(char(parts(partIndex)));
end

metadata = struct([]);
if isfield(result, 'metadata') && ~isempty(result.metadata)
    metadataPaths = string({result.metadata.hdf5_path});
    metadataIndex = find(metadataPaths == hdf5Path, 1);
    if ~isempty(metadataIndex)
        metadata = result.metadata(metadataIndex);
    end
end
end


function path = normalise_path(path)
if ~(ischar(path) || (isstring(path) && isscalar(path)))
    error('gprMax:MATLAB:InvalidHDF5Path', ...
        'The HDF5 path must be a character vector or scalar string.');
end
path = strtrim(string(path));
if strlength(path) == 0
    error('gprMax:MATLAB:InvalidHDF5Path', 'The HDF5 path cannot be empty.');
end
path = replace(path, "\\", "/");
if ~startsWith(path, "/")
    path = "/" + path;
end
while strlength(path) > 1 && endsWith(path, "/")
    path = extractBefore(path, strlength(path));
end
end
