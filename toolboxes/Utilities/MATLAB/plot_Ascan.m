% plot_Ascan.m
% Script to save and plot EM fields from a gprMax A-scan
%
% Craig Warren

clearvars; clc

[filename, pathname] = uigetfile('*.h5', 'Select gprMax A-scan output file to plot');
fullfilename = strcat(pathname, filename);

if ~isequal(filename, 0)
    header.title = h5readatt(fullfilename, '/', 'Title');
    header.iterations = double(h5readatt(fullfilename,'/', 'Iterations'));
    tmp = h5readatt(fullfilename, '/', 'dx_dy_dz');
    header.dx = tmp(1);
    header.dy = tmp(2);
    header.dz = tmp(3);
    header.dt = h5readatt(fullfilename, '/', 'dt');
    header.nsrc = h5readatt(fullfilename, '/', 'nsrc');
    header.nrx = h5readatt(fullfilename, '/', 'nrx');

    % Time vector for plotting
    time = linspace(0, (header.iterations - 1) * header.dt, header.iterations)';

    % Save and plot fields from each receiver
    for n=1:header.nrx
        path = strcat('/rxs/rx', num2str(n));
        tmp = h5readatt(fullfilename, path, 'Position');
        header.rx(n) = tmp(1);
        header.ry(n) = tmp(2);
        header.rz(n) = tmp(3);
        info = h5info(fullfilename, path);
        outputs = {info.Datasets.Name};
        if isempty(outputs)
            warning('No receiver datasets found at %s', path);
            continue
        end

        fh1=figure('Name', strcat('rx', num2str(n)));
        tiledlayout(ceil(numel(outputs) / 2), 2);
        ax = gobjects(1, numel(outputs));
        for outputindex=1:numel(outputs)
            output = outputs{outputindex};
            outputdata = h5read(fullfilename, strcat(path, '/', output));
            fields.(lower(output))(:,n) = outputdata;
            ax(outputindex) = nexttile;
            if startsWith(output, 'E')
                colour = 'r'; units = 'Field strength [V/m]';
            elseif startsWith(output, 'H')
                colour = 'b'; units = 'Field strength [A/m]';
            else
                colour = 'k'; units = 'Current [A]';
            end
            plot(time, outputdata, colour, 'LineWidth', 2), grid on
            xlabel('Time [s]'), ylabel(units), title(strrep(output, '_', '\_'))
        end
        set(ax,'FontSize', 16, 'xlim', [0 time(end)]);

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
end
