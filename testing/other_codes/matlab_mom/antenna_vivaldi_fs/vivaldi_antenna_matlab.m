function vivaldi_antenna_matlab(max_edge, results_dir)
% Independent MoM reference for the default, free-space PEC Vivaldi.
% Run with no arguments for the retained reference; optional arguments allow
% mesh convergence runs in a separate directory without replacing it.
if nargin < 1
    max_edge = 0.008;
end
if nargin < 2
    results_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if ~isfolder(results_dir)
    mkdir(results_dir);
end
set(groot, 'defaultFigureVisible', 'off');
started = tic;
antenna = vivaldi(TaperLength=0.243, ApertureWidth=0.105, ...
    OpeningRate=25, SlotLineWidth=0.0005, CavityDiameter=0.024, ...
    CavityToTaperSpacing=0.023, GroundPlaneLength=0.300, ...
    GroundPlaneWidth=0.125, FeedOffset=-0.1045, Conductor=metal('PEC'));
reference_impedance = 50;
frequency_hz = (1:0.05:2).' * 1e9;
pattern_frequency_hz = [1, 1.5, 2] * 1e9;
angle_deg = (-180:2:180).';

% Export the actual CAD outline, including the separate cavity boundary.
% The comparison uses these coordinates, not a reimplementation of MATLAB's
% taper construction. Translation into the FDTD domain is recorded separately.
pcb = pcbStack(antenna);
vertices = pcb.Layers{1}.Vertices;
separators = [0; find(any(isnan(vertices), 2)); size(vertices, 1) + 1];
loops = cell(1, numel(separators)-1);
for k = 1:numel(loops)
    loops{k} = vertices(separators(k)+1:separators(k+1)-1, 1:2);
end
geometry = struct('coordinate_units', 'm', 'plane', 'xy', ...
    'conductor', 'pec', 'boundary_loops_xy_m', {loops}, ...
    'feed_x_m', antenna.FeedOffset, 'slot_width_m', antenna.SlotLineWidth, ...
    'reference_impedance_ohm', reference_impedance, ...
    'matlab_release', version('-release'));
write_json(fullfile(results_dir, 'vivaldi_geometry.json'), geometry);
figure; show(antenna); view(0, 90); axis equal;
exportgraphics(gcf, fullfile(results_dir, 'vivaldi_matlab_geometry.png'), Resolution=180);
close(gcf);

mesh_data = mesh(antenna, MaxEdgeLength=max_edge);
mesh_points = mesh_data.Points;
mesh_triangles = mesh_data.Triangles;
fprintf('MATLAB mesh: %d points, %d triangles\n', size(mesh_points,2), size(mesh_triangles,2));
input_impedance = complex(zeros(size(frequency_hz)));
for k = 1:numel(frequency_hz)
    input_impedance(k) = impedance(antenna, frequency_hz(k));
    fprintf('Zin %.3f GHz: %.6g %+.6gj ohm (%.1f s)\n', ...
        frequency_hz(k)/1e9, real(input_impedance(k)), imag(input_impedance(k)), toc(started));
end
s11 = (input_impedance-reference_impedance) ./ (input_impedance+reference_impedance);
port_table = table(frequency_hz, real(input_impedance), imag(input_impedance), ...
    real(s11), imag(s11), 'VariableNames', ...
    {'frequency_hz','Zin_real_ohm','Zin_imag_ohm','S11_real','S11_imag'});
writetable(port_table, fullfile(results_dir, 'vivaldi_matlab_port.csv'));

% Both full circles use angle zero along +x (the aperture/end-fire direction).
% XY: (cos(a), sin(a), 0); XZ: (cos(a), 0, sin(a)).
elevation = (-90:2:90).';
cut_el = asind(sind(angle_deg));
cut_az = zeros(size(angle_deg));
cut_az(cosd(angle_deg) < 0) = 180;
idx = sub2ind([numel(elevation), 2], round((cut_el+90)/2)+1, 1+cut_az/180);
directivity_xy_dbi = zeros(numel(angle_deg), numel(pattern_frequency_hz));
directivity_xz_dbi = directivity_xy_dbi;
peak_directivity_dbi = zeros(size(pattern_frequency_hz));
for k = 1:numel(pattern_frequency_hz)
    f = pattern_frequency_hz(k);
    directivity_xy_dbi(:,k) = reshape(pattern(antenna, f, angle_deg, 0, ...
        Type='directivity', Normalize=false), [], 1);
    xz = pattern(antenna, f, [0 180], elevation, Type='directivity', Normalize=false);
    assert(isequal(size(xz), [numel(elevation), 2]));
    directivity_xz_dbi(:,k) = xz(idx);
    full = pattern(antenna, f, -180:5:175, -90:5:90, Type='directivity', Normalize=false);
    peak_directivity_dbi(k) = max(full, [], 'all');
    pattern_table = table(angle_deg, directivity_xy_dbi(:,k), directivity_xz_dbi(:,k), ...
        'VariableNames', {'angle_deg','xy_directivity_dbi','xz_directivity_dbi'});
    writetable(pattern_table, fullfile(results_dir, sprintf('vivaldi_matlab_pattern_%gGHz.csv', f/1e9)));
    fprintf('Pattern %.3f GHz: peak %.4f dBi (%.1f s)\n', f/1e9, peak_directivity_dbi(k), toc(started));
end
% Lossless PEC in air: gain equals directivity. Realised gain also includes
% the mismatch to the stated 50-ohm reference; no artificial pattern scaling.
pattern_s11 = interp1(frequency_hz, s11, pattern_frequency_hz);
peak_realized_gain_dbi = peak_directivity_dbi + 10*log10(1-abs(pattern_s11).^2);
elapsed_seconds = toc(started);
matlab_version = version;
save(fullfile(results_dir, 'vivaldi_antenna_matlab.mat'), 'geometry', 'vertices', ...
    'mesh_points', 'mesh_triangles', 'max_edge', 'frequency_hz', 'input_impedance', ...
    's11', 'reference_impedance', 'pattern_frequency_hz', 'angle_deg', ...
    'directivity_xy_dbi', 'directivity_xz_dbi', 'peak_directivity_dbi', ...
    'peak_realized_gain_dbi', 'elapsed_seconds', 'matlab_version', '-v7');
summary = struct('matlab_version', matlab_version, 'max_edge_m', max_edge, ...
    'mesh_points', size(mesh_points,2), 'mesh_triangles', size(mesh_triangles,2), ...
    'elapsed_seconds', elapsed_seconds, 'pattern_frequency_hz', pattern_frequency_hz, ...
    'peak_directivity_dbi', peak_directivity_dbi, 'peak_realized_gain_dbi', peak_realized_gain_dbi);
write_json(fullfile(results_dir, 'vivaldi_matlab_summary.json'), summary);
fprintf('Finished MATLAB reference in %.1f s: %s\n', elapsed_seconds, results_dir);
end

function write_json(filename, value)
fid = fopen(filename, 'w');
assert(fid ~= -1, 'Cannot open %s', filename);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, PrettyPrint=true));
end
