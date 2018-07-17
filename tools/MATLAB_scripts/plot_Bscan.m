% plot_Bscan.m
% Script to plot EM fields from a gprMax B-scan
%
% Craig Warren

clear all, clc

[filename, pathname] = uigetfile('*.out', 'Select gprMax output file to plot B-scan', 'MultiSelect', 'on');
filename = fullfile(pathname, filename);

% Open file and read fields
if filename ~= 0
    iterations = double(h5readatt(filename, '/', 'Iterations'));
    dt = h5readatt(filename, '/', 'dt');

    prompt = 'Which field do you want to view? Ex, Ey, or Ez: ';
    field = input(prompt,'s');
    fieldpath = strcat('/rxs/rx1/', field);
    field = h5read(filename, fieldpath)';
    time = linspace(0, (iterations - 1) * dt, iterations)';
    traces = 0:size(field, 2);

    fh1=figure('Name', filename);
    clims = [-max(max(abs(field))) max(max(abs(field)))];
    im = imagesc(traces, time, field, clims);
    xlabel('Trace number');
    ylabel('Time [s]');
    c = colorbar;
    c.Label.String = 'Field strength [V/m]';
    ax = gca;
    ax.FontSize = 16;
    xlim([0 traces(end)]);

    % Options to create a nice looking figure for display and printing
    set(fh1,'Color','white','Menubar','none');
    X = 60;   % Paper size
    Y = 30;   % Paper size
    xMargin = 0; % Left/right margins from page borders
    yMargin = 0;  % Bottom/top margins from page borders
    xSize = X - 2*xMargin;    % Figure size on paper (width & height)
    ySize = Y - 2*yMargin;    % Figure size on paper (width & height)

    % Figure size displayed on screen
    set(fh1, 'Units','centimeters', 'Position', [0 0 xSize ySize])
    movegui(fh1, 'center')

    % Figure size printed on paper
    set(fh1,'PaperUnits', 'centimeters')
    set(fh1,'PaperSize', [X Y])
    set(fh1,'PaperPosition', [xMargin yMargin xSize ySize])
    set(fh1,'PaperOrientation', 'portrait')

end
