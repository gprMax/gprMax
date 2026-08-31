function [figures, traces] = plot_Ascan(filename, varargin)
%PLOT_ASCAN Plot real time-domain outputs from gprMax receivers.
% Original plotting utility contributed by Craig Warren.
%
%   PLOT_ASCAN(FILE) plots every available component from every receiver in
%   the main grid. Calling PLOT_ASCAN without a filename opens a file picker.
%
%   [FIGURES, TRACES] = PLOT_ASCAN(...) returns the figure handles and the
%   plotted values, physical sample times, units, and original HDF5 paths.
%
%   Name-value options:
%       Grid             Main grid "/" or a subgrid path such as
%                        "/subgrids/fine" (default "/").
%       Receiver         Numeric receiver indices (default all).
%       Path             One or more arbitrary HDF5 groups, such as
%                        "/ports/feed"; cannot be combined with Receiver.
%       Outputs          Components such as ["Ex", "Hz"] (default all).
%       FFT              Plot a relative-magnitude spectrum alongside one
%                        selected component (default false).
%       Visible          Display figures (default true).
%       Save             Save each figure as PNG (default false).
%       OutputDirectory  PNG destination (default beside the HDF5 file).

if nargin == 0 || isempty(filename)
    [selectedFile, selectedPath] = uigetfile( ...
        '*.h5', 'Select a gprMax A-scan output file');
    if isequal(selectedFile, 0)
        figures = gobjects(0);
        traces = struct;
        return
    end
    filename = fullfile(selectedPath, selectedFile);
end

parser = inputParser;
parser.FunctionName = mfilename;
addRequired(parser, 'filename', @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'Grid', "/", ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(parser, 'Receiver', [], ...
    @(x) isnumeric(x) && isvector(x) && all(isfinite(x)) ...
    && all(x >= 1) && all(mod(x, 1) == 0));
addParameter(parser, 'Path', strings(0, 1), ...
    @(x) ischar(x) || isstring(x) || iscellstr(x));
addParameter(parser, 'Outputs', strings(0, 1), ...
    @(x) ischar(x) || isstring(x) || iscellstr(x));
addParameter(parser, 'FFT', false, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'Visible', true, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'Save', false, @(x) islogical(x) && isscalar(x));
addParameter(parser, 'OutputDirectory', "", ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
parse(parser, filename, varargin{:});

filename = char(string(parser.Results.filename));
if ~isfile(filename)
    error('gprMax:MATLAB:HDF5FileNotFound', ...
        'The HDF5 file does not exist: %s', filename);
end
gridPath = normalise_group_path(parser.Results.Grid);
receiverIndices = parser.Results.Receiver;
requestedPaths = string(parser.Results.Path);
requestedPaths = requestedPaths(:);
if any(ismissing(requestedPaths) | strlength(strtrim(requestedPaths)) == 0)
    error('gprMax:MATLAB:InvalidHDF5Path', ...
        'Path cannot contain missing or empty values.');
end
if ~isempty(requestedPaths)
    if ~isempty(receiverIndices)
        error('gprMax:MATLAB:ConflictingSelection', ...
            'Path and Receiver cannot be supplied together.');
    end
    objectPaths = strings(numel(requestedPaths), 1);
    objectFields = strings(numel(requestedPaths), 1);
    usedFields = {};
    for index = 1:numel(requestedPaths)
        objectPaths(index) = normalise_group_path(requestedPaths(index));
        rawField = regexprep(char(objectPaths(index)), '[^A-Za-z0-9]+', '_');
        field = matlab.lang.makeValidName(rawField);
        field = matlab.lang.makeUniqueStrings(field, usedFields, namelengthmax);
        objectFields(index) = string(field);
        usedFields{end + 1} = field; %#ok<AGROW>
    end
else
    receiverRoot = join_path(gridPath, 'rxs');
    try
        receiverInfo = h5info(filename, receiverRoot);
    catch exception
        error('gprMax:MATLAB:NoReceivers', ...
            'No receiver group exists at %s in %s.\n%s', ...
            receiverRoot, filename, exception.message);
    end
    allReceivers = strings(numel(receiverInfo.Groups), 1);
    for index = 1:numel(receiverInfo.Groups)
        allReceivers(index) = leaf_name(receiverInfo.Groups(index).Name);
    end
    if isempty(allReceivers)
        error('gprMax:MATLAB:NoReceivers', ...
            'No receivers were found below %s in %s.', receiverRoot, filename);
    end
    receiverNumbers = str2double(extractAfter(allReceivers, 'rx'));
    [~, order] = sort(receiverNumbers);
    allReceivers = allReceivers(order);
    if isempty(receiverIndices)
        selectedReceivers = allReceivers;
    else
        selectedReceivers = "rx" + string(receiverIndices(:));
        missing = selectedReceivers(~ismember(selectedReceivers, allReceivers));
        if ~isempty(missing)
            error('gprMax:MATLAB:ReceiverNotFound', ...
                'Receiver %s was not found below %s.', missing(1), receiverRoot);
        end
    end
    objectPaths = strings(numel(selectedReceivers), 1);
    for index = 1:numel(selectedReceivers)
        objectPaths(index) = join_path(receiverRoot, selectedReceivers(index));
    end
    objectFields = selectedReceivers;
end
result = gprmax_read_h5(filename, 'Paths', objectPaths);

requestedOutputs = string(parser.Results.Outputs);
requestedOutputs = requestedOutputs(:);
if any(ismissing(requestedOutputs) | strlength(strtrim(requestedOutputs)) == 0)
    error('gprMax:MATLAB:InvalidOutput', ...
        'Outputs cannot contain missing or empty values.');
end
requestedOutputs = strtrim(requestedOutputs);
if parser.Results.FFT && numel(requestedOutputs) > 1
    error('gprMax:MATLAB:FFTRequiresOneOutput', ...
        'FFT plotting accepts only one selected output.');
end

visible = visibility_value(parser.Results.Visible);
outputDirectory = output_directory(filename, parser.Results.OutputDirectory, ...
    parser.Results.Save);
figures = gobjects(0);
traces = struct;

for objectIndex = 1:numel(objectPaths)
    objectPath = objectPaths(objectIndex);
    info = h5info(filename, char(objectPath));
    available = string({info.Datasets.Name});
    outputs = choose_outputs(requestedOutputs, available, objectPath);
    if parser.Results.FFT && numel(outputs) ~= 1
        error('gprMax:MATLAB:FFTRequiresOneOutput', ...
            'FFT plotting requires exactly one available output.');
    end

    figureHandle = figure( ...
        'Name', char(objectPath), ...
        'Color', 'white', ...
        'Visible', visible);
    figures(end + 1) = figureHandle; %#ok<AGROW>
    if parser.Results.FFT
        layout = tiledlayout(figureHandle, 1, 2, ...
            'TileSpacing', 'compact', 'Padding', 'compact');
    else
        columns = min(2, numel(outputs));
        rows = ceil(numel(outputs) / columns);
        layout = tiledlayout(figureHandle, rows, columns, ...
            'TileSpacing', 'compact', 'Padding', 'compact');
    end
    title(layout, object_title(result, objectPath));

    objectField = char(objectFields(objectIndex));
    traces.(objectField) = struct;
    for outputIndex = 1:numel(outputs)
        output = outputs(outputIndex);
        datasetPath = join_path(objectPath, output);
        [values, metadata] = gprmax_h5_get(result, datasetPath);
        validate_time_trace(values, datasetPath);
        values = values(:);
        sampleInterval = metadata_attribute( ...
            metadata, 'SampleInterval', fallback_dt(result, objectPath));
        timeOffset = metadata_attribute(metadata, 'TimeSampleOffset', 0);
        time = timeOffset + (0:numel(values) - 1)' * sampleInterval;
        [units, colour] = output_style(output, metadata);

        traceField = matlab.lang.makeValidName(char(output));
        traces.(objectField).(traceField) = struct( ...
            'time', time, ...
            'values', values, ...
            'sample_interval', sampleInterval, ...
            'time_offset', timeOffset, ...
            'units', string(units), ...
            'hdf5_path', datasetPath);

        axisHandle = nexttile(layout);
        plot(axisHandle, time, values, 'Color', colour, 'LineWidth', 1.5);
        grid(axisHandle, 'on');
        xlabel(axisHandle, 'Time [s]');
        ylabel(axisHandle, sprintf('%s [%s]', output, units));
        title(axisHandle, output, 'Interpreter', 'none');
        if ~isempty(time)
            xlim(axisHandle, [time(1), time(end)]);
        end

        if parser.Results.FFT
            spectrumAxis = nexttile(layout);
            [frequency, relativeMagnitude] = relative_spectrum(values, sampleInterval);
            plot(spectrumAxis, frequency, relativeMagnitude, ...
                'Color', colour, 'LineWidth', 1.5);
            grid(spectrumAxis, 'on');
            xlabel(spectrumAxis, 'Frequency [Hz]');
            ylabel(spectrumAxis, 'Relative magnitude [dB]');
            title(spectrumAxis, output + " spectrum", 'Interpreter', 'none');
        end
    end

    if parser.Results.Save
        [~, stem] = fileparts(filename);
        safeReceiver = regexprep(char(objectPath), '[^A-Za-z0-9]+', '_');
        destination = fullfile(outputDirectory, ...
            sprintf('%s%s.png', stem, safeReceiver));
        exportgraphics(figureHandle, destination, 'Resolution', 180);
    end
end
end


function outputs = choose_outputs(requested, available, receiverPath)
if isempty(requested)
    preferred = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"];
    outputs = preferred(ismember(preferred, available));
    outputs = [outputs, available(~ismember(available, outputs))];
else
    outputs = requested(:)';
end
missing = outputs(~ismember(outputs, available));
if ~isempty(missing)
    error('gprMax:MATLAB:OutputNotFound', ...
        'Output %s was not found at %s. Available outputs: %s.', ...
        missing(1), receiverPath, strjoin(available, ', '));
end
if isempty(outputs)
    error('gprMax:MATLAB:NoTimeDomainOutputs', ...
        'No datasets were found at %s.', receiverPath);
end
end


function validate_time_trace(values, path)
if ~isnumeric(values) || ~isreal(values) || ~isvector(values)
    error('gprMax:MATLAB:NotAnAScan', ...
        '%s must be a real one-dimensional time history.', path);
end
end


function [frequency, magnitudeDb] = relative_spectrum(values, dt)
count = numel(values);
spectrum = fft(double(values));
last = floor(count / 2) + 1;
frequency = (0:last - 1)' / (count * dt);
magnitude = abs(spectrum(1:last)) / count;
peak = max(magnitude);
if peak == 0
    magnitudeDb = zeros(size(magnitude));
else
    magnitudeDb = 20 * log10(max(magnitude / peak, realmin('double')));
end
end


function titleText = object_title(result, objectPath)
titleText = string(objectPath);
metadata = metadata_for_path(result, objectPath);
name = metadata_attribute(metadata, 'Name', "");
if strlength(string(name)) > 0
    titleText = titleText + " - " + string(name);
end
end


function dt = fallback_dt(result, objectPath)
if startsWith(objectPath, '/subgrids/')
    parts = split(strip(string(objectPath), '/'), '/');
    gridPath = "/subgrids/" + parts(2);
    dt = metadata_attribute(metadata_for_path(result, gridPath), 'dt', NaN);
elseif isfield(result.header, 'dt')
    dt = double(result.header.dt);
else
    dt = NaN;
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
if ~isempty(index)
    metadata = result.metadata(index);
end
end


function value = metadata_attribute(metadata, name, defaultValue)
value = defaultValue;
if isempty(metadata) || isempty(metadata.attributes)
    return
end
names = string({metadata.attributes.name});
index = find(names == string(name), 1);
if ~isempty(index)
    value = metadata.attributes(index).value;
end
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


function directory = output_directory(filename, requested, saveEnabled)
directory = string(requested);
if strlength(directory) == 0
    [folder, ~] = fileparts(filename);
    if isempty(folder), folder = '.'; end
    directory = string(folder);
end
if saveEnabled && ~isfolder(directory)
    mkdir(directory);
end
directory = char(directory);
end


function value = visibility_value(visible)
if visible
    value = 'on';
else
    value = 'off';
end
end


function path = normalise_group_path(path)
path = string(path);
path = replace(strtrim(path), "\\", "/");
if strlength(path) == 0, path = "/"; end
if ~startsWith(path, "/"), path = "/" + path; end
while strlength(path) > 1 && endsWith(path, "/")
    path = extractBefore(path, strlength(path));
end
path = char(path);
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


function name = leaf_name(path)
parts = split(string(path), '/');
name = parts(end);
end
