function [figureHandle, scan] = plot_Bscan(filename, output, varargin)
%PLOT_BSCAN Plot a real, merged gprMax time-domain trace matrix.
% Original plotting utility contributed by Craig Warren.
%
%   PLOT_BSCAN(FILE, OUTPUT) plots OUTPUT from /rxs/rx1. Calling the
%   function without FILE opens a file picker; OUTPUT then defaults to Ez.
%
%   PLOT_BSCAN(..., "Path", GROUP) selects another HDF5 group, for example
%   "/ports/receive", "/tls/tl1", or a subgrid receiver. This permits
%   merged Vtotal antenna-terminal responses to be plotted directly.
%
%   [FIGURE, SCAN] = PLOT_BSCAN(...) returns the figure handle and the raw
%   trace matrix, physical time vector, horizontal coordinate, units, and
%   source metadata used by the plot.
%
%   Name-value options:
%       Path             Dataset parent (default "/rxs/rx1").
%       UseDistance      Use cumulative receiver/port path distance when
%                        trace metadata exist (default true).
%       Visible          Display the figure (default true).
%       Save             Save the figure as PNG (default false).
%       OutputFile       PNG destination; implies Save.
%       ColourLimit      Positive symmetric colour limit (default data max).

if nargin == 0 || isempty(filename)
    [selectedFile, selectedPath] = uigetfile( ...
        '*.h5', 'Select a merged gprMax output file');
    if isequal(selectedFile, 0)
        figureHandle = gobjects(0);
        scan = struct;
        return
    end
    filename = fullfile(selectedPath, selectedFile);
end
if nargin < 2 || isempty(output)
    output = "Ez";
end

parser = inputParser;
parser.FunctionName = mfilename;
addRequired(parser, 'filename', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addRequired(parser, 'output', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'Path', "/rxs/rx1", ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'UseDistance', true, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'Visible', true, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'Save', false, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'OutputFile', "", ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'ColourLimit', [], ...
    @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isfinite(x) && x > 0));
parse(parser, filename, output, varargin{:});

filename = char(string(parser.Results.filename));
if ~isfile(filename)
    error('gprMax:MATLAB:HDF5FileNotFound', ...
        'The HDF5 file does not exist: %s', filename);
end
groupPath = normalise_group_path(parser.Results.Path);
output = strtrim(string(parser.Results.output));
if strlength(output) == 0
    error('gprMax:MATLAB:InvalidOutput', 'Output cannot be empty.');
end
datasetPath = join_path(groupPath, output);
metadataPath = trace_metadata_path(groupPath);

paths = datasetPath;
hasTracePositions = false;
if parser.Results.UseDistance
    try
        h5info(filename, char(metadataPath));
        paths(end + 1) = metadataPath;
        hasTracePositions = true;
    catch
        % Trace positions are optional; fall back to trace number.
    end
end
result = gprmax_read_h5(filename, 'Paths', paths);
[values, metadata] = gprmax_h5_get(result, datasetPath);
if ~isnumeric(values) || ~isreal(values) || ~ismatrix(values) ...
        || size(values, 2) < 2
    error('gprMax:MATLAB:NotABScan', ...
        '%s must be a real (time, trace) matrix with at least two traces.', ...
        datasetPath);
end

sampleInterval = metadata_attribute( ...
    metadata, 'SampleInterval', fallback_dt(result, groupPath));
timeOffset = metadata_attribute(metadata, 'TimeSampleOffset', 0);
time = timeOffset + (0:size(values, 1) - 1)' * sampleInterval;
[units, ~] = output_style(output, metadata);

if hasTracePositions
    positions = gprmax_h5_get(result, metadataPath);
    if isnumeric(positions) && size(positions, 1) == size(values, 2) ...
            && size(positions, 2) == 3
        coordinate = [0; cumsum(vecnorm(diff(double(positions), 1, 1), 2, 2))];
        horizontalLabel = 'Cumulative trace distance [m]';
    else
        coordinate = (0:size(values, 2) - 1)';
        horizontalLabel = 'Trace number';
        hasTracePositions = false;
    end
else
    coordinate = (0:size(values, 2) - 1)';
    horizontalLabel = 'Trace number';
end

colourLimit = parser.Results.ColourLimit;
if isempty(colourLimit)
    colourLimit = max(abs(double(values(:))));
    if colourLimit == 0, colourLimit = 1; end
end

figureHandle = figure( ...
    'Name', sprintf('%s - %s', filename, datasetPath), ...
    'Color', 'white', ...
    'Visible', visibility_value(parser.Results.Visible));
axisHandle = axes(figureHandle);
surface(axisHandle, coordinate, time, zeros(size(values)), double(values), ...
    'EdgeColor', 'none', 'FaceColor', 'texturemap');
view(axisHandle, 2);
axis(axisHandle, 'tight');
set(axisHandle, 'YDir', 'reverse', 'FontSize', 12);
colormap(axisHandle, red_white_blue(256));
clim(axisHandle, [-colourLimit, colourLimit]);
xlabel(axisHandle, horizontalLabel);
ylabel(axisHandle, 'Time [s]');
title(axisHandle, plot_title(result, datasetPath), 'Interpreter', 'none');
colourbarHandle = colorbar(axisHandle);
colourbarHandle.Label.String = sprintf('%s [%s]', output, units);

scan = struct( ...
    'values', values, ...
    'time', time, ...
    'coordinate', coordinate, ...
    'positions_used', hasTracePositions, ...
    'sample_interval', sampleInterval, ...
    'time_offset', timeOffset, ...
    'units', string(units), ...
    'hdf5_path', datasetPath, ...
    'source_file', string(filename));

outputFile = string(parser.Results.OutputFile);
saveEnabled = parser.Results.Save || strlength(outputFile) > 0;
if saveEnabled
    if strlength(outputFile) == 0
        [folder, stem] = fileparts(filename);
        if isempty(folder), folder = '.'; end
        safePath = regexprep(char(datasetPath), '[^A-Za-z0-9]+', '_');
        outputFile = fullfile(folder, stem + string(safePath) + ".png");
    elseif ~endsWith(outputFile, '.png', 'IgnoreCase', true)
        error('gprMax:MATLAB:InvalidPlotExtension', ...
            'OutputFile must have a .png extension.');
    end
    parent = fileparts(outputFile);
    if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
    exportgraphics(figureHandle, outputFile, 'Resolution', 180);
end
end


function path = trace_metadata_path(groupPath)
groupPath = string(groupPath);
parts = split(strip(groupPath, '/'), '/');
if numel(parts) >= 2 && parts(1) == "subgrids"
    prefix = "/subgrids/" + parts(2);
    remainder = "/" + strjoin(parts(3:end), '/');
    path = prefix + "/trace_metadata" + remainder + "/Position";
else
    path = "/trace_metadata" + groupPath + "/Position";
end
end


function titleText = plot_title(result, datasetPath)
titleText = datasetPath;
if isfield(result.header, 'Title')
    titleText = string(result.header.Title) + " - " + datasetPath;
end
end


function dt = fallback_dt(result, groupPath)
dt = NaN;
if startsWith(groupPath, '/subgrids/')
    parts = split(strip(string(groupPath), '/'), '/');
    gridPath = "/subgrids/" + parts(2);
    metadata = metadata_for_path(result, gridPath);
    dt = metadata_attribute(metadata, 'dt', NaN);
elseif isfield(result.header, 'dt')
    dt = double(result.header.dt);
end
if ~isscalar(dt) || ~isfinite(dt) || dt <= 0
    error('gprMax:MATLAB:MissingSampleInterval', ...
        'No valid SampleInterval or grid dt metadata is available.');
end
end


function metadata = metadata_for_path(result, path)
metadata = struct([]);
paths = string({result.metadata.hdf5_path});
index = find(paths == string(path), 1);
if ~isempty(index), metadata = result.metadata(index); end
end


function value = metadata_attribute(metadata, name, defaultValue)
value = defaultValue;
if isempty(metadata) || isempty(metadata.attributes), return; end
names = string({metadata.attributes.name});
index = find(names == string(name), 1);
if ~isempty(index), value = metadata.attributes(index).value; end
end


function [units, colour] = output_style(output, metadata)
units = string(metadata_attribute(metadata, 'Units', ""));
if startsWith(output, 'E')
    if strlength(units) == 0, units = "V/m"; end
    colour = [0.80, 0.10, 0.10];
elseif startsWith(output, 'H')
    if strlength(units) == 0, units = "A/m"; end
    colour = [0.10, 0.35, 0.75];
elseif startsWith(output, 'I')
    if strlength(units) == 0, units = "A"; end
    colour = [0.15, 0.15, 0.15];
elseif startsWith(output, 'V')
    if strlength(units) == 0, units = "V"; end
    colour = [0.45, 0.15, 0.65];
else
    if strlength(units) == 0, units = "SI"; end
    colour = [0.15, 0.15, 0.15];
end
units = char(units);
end


function colours = red_white_blue(count)
half = floor(count / 2);
blueToWhite = [linspace(0, 1, half)', linspace(0, 1, half)', ones(half, 1)];
whiteToRed = [ones(count - half, 1), ...
    linspace(1, 0, count - half)', linspace(1, 0, count - half)'];
colours = [blueToWhite; whiteToRed];
end


function value = visibility_value(visible)
if visible, value = 'on'; else, value = 'off'; end
end


function path = normalise_group_path(path)
path = string(path);
path = replace(strtrim(path), "\\", "/");
if strlength(path) == 0, path = "/"; end
if ~startsWith(path, "/"), path = "/" + path; end
while strlength(path) > 1 && endsWith(path, "/")
    path = extractBefore(path, strlength(path));
end
end


function path = join_path(parent, child)
parent = string(parent);
child = strip(string(child), '/');
if parent == "/"
    path = "/" + child;
else
    path = strip(parent, 'right', '/') + "/" + child;
end
end
