% outputfile_converter.m - converts gprMax merged output HDF5 file to RD3,
% DZT, DT1 and IPRB files.
%
% Author: Dimitrios Angelis
% Copyright: 2017-2018
% Last modified: 18/07/2018

clear, clc, close all;

% Select file =============================================================
[infile, path] = uigetfile('*.out', 'Select gprMax output file', ...
    'Multiselect', 'Off');
if isequal(infile, 0)
    infile = [];
    path   = [];
    HDR    = [];
    data   = [];
    return
end


% File name, path name and file extension =================================
HDR.fname = strrep(lower(infile), '.out', '');
HDR.pname = path;
HDR.fext  = 'out';


% Read data from HDF5 file ================================================
infile = [HDR.pname infile];
dataex = h5read(infile, '/rxs/rx1/Ex');
dataey = h5read(infile, '/rxs/rx1/Ey');
dataez = h5read(infile, '/rxs/rx1/Ez');


% Field check =============================================================
if dataey == 0 & dataez == 0
    data = dataex';
elseif dataex == 0 & dataez == 0
    data = dataey';
elseif dataex == 0 & dataey == 0
    data = dataez';
else
    maxex = max(max(dataex));
    maxey = max(max(dataey));
    maxez = max(max(dataez));
    if maxex > maxey & maxex > maxez
        data = dataex';
    elseif maxey > maxex & maxey > maxez
        data = dataey';
    elseif maxez > maxex & maxez > maxey
        data = dataez';
    end
end


% Sigle to double precision ===============================================
data = double(data);


% The HDF5 file does not contain information about the centre frequency of
% the waveform, the Tx-Rx separation distance and the trace step. The user
% needs to provide this information.
while 1
    prompt = {'Waveform Centre Frequency (MHz)', ...
        'Source-Receiver Distance (m)', 'Trace Step (m)'};
    dlg_title = 'Additional Information';
    answer = inputdlg(prompt, dlg_title, [1 50]);
    answer = str2double(answer);
    if isempty(answer)
        HDR  = [];
        data = [];
        return
    elseif isnan(answer(1)) || isnan(answer(2)) || isnan(answer(3))...
            || answer(1) <= 0 || answer(2) < 0 || answer(3) <=0
        continue
    else
        break
    end
end


% Create header with basic information ====================================
HDR.centre_freq = answer(1);                                    % Centre frequency (MHz)
HDR.ant_sep = answer(2);                                        % Antenna seperation / Tx-Rx distance (m)
HDR.trac_int = answer(3);                                       % Trace interval / step (m)
HDR.samp_int = h5readatt(infile, '/', 'dt') * 10^9;             % Sampling interval / step (ns)
HDR.samp_freq = (1 / HDR.samp_int) * 10^3;                      % Sampling frequency (MHz)
[HDR.num_samp, HDR.num_trac] = size(data);                      % Number of samples & traces
HDR.time_window = HDR.num_samp * HDR.samp_int;                  % Time window (ns)
HDR.antenna = ['gprMax ', num2str(HDR.centre_freq), 'MHz'];     % Antenna name


%**************************************************************************
%******************************** Optional ********************************

% Resample to 1024 samples ================================================
% I usually perform this step for either 512 or 1024 samples (line 100)
% because many GPR processing software cannot load files with more samples.
tx1  = 1 : HDR.num_samp;
fs1  = 1024 / HDR.num_samp;                                     % <------- 1024 samples
data = resample(data, tx1, fs1, 'spline');

[HDR.num_samp, ~] = size(data);                                 % New number of samples after resampling
HDR.samp_int      = HDR.time_window / HDR.num_samp;             % New sampling interval (ns) after resampling
HDR.samp_freq     = (1 / HDR.samp_int) * 10^3;                  % New sampling frequency (MHz) after resampling

%**************************************************************************
%**************************************************************************


%**************************************************************************
%******************************** Optional ********************************

% Data scale ==============================================================
data = data * 32767.5 ./ max(max(abs(data)));                   %signal * ((1 - 1 / 2^bitrate) * 32768) / max(signal)
data(data >= 32767) = 32767;
data(data <= -32768) = -32768;
data = round(data);

%**************************************************************************
%**************************************************************************


% Plots ===================================================================
pmin = min(data(:));                                            %Minimun plot scale
pmax = max(data(:));                                            %Maximum plot scale

x = 0 : HDR.trac_int : (HDR.num_trac - 1) * HDR.trac_int;       %Distance of each trace (m)
t = HDR.samp_int : HDR.samp_int : HDR.num_samp * HDR.samp_int;  %Time of each sample (ns)

% Bscan
f1 = figure('Name', 'Bscan', ...
    'NumberTitle', 'off', ...
    'Menubar', 'None', ...
    'Toolbar', 'Figure');

clims = [pmin pmax];
colormap (bone(256));                                           %Black(negative) to white(positive)
im1 = imagesc(x, t, data, clims);
set(im1, 'cdatamapping', 'scaled');
title(HDR.fname);
xlabel('Distance (m)');
ylabel('Time (ns)');
ax1 = gca;
ax1.XAxisLocation = 'Top';
ax1.FontSize = 12;
box off;
movegui(f1, 'northeast');


% Frequency Spectrum
m = 2.^nextpow2(HDR.num_samp);

amp = fft(data, m);
amp = (abs(amp(1 : m / 2, :)) / m) * 2;
amp = mean(amp.');

freq = HDR.samp_freq .* (0 : (m / 2) - 1) / m;

f2 = figure('Name', 'Frequency Spectrum', ...
    'NumberTitle', 'off', ...
    'Menubar', 'None', ...
    'Toolbar', 'Figure');

area(freq, amp, 'FaceColor', 'black');
title(HDR.fname);
xlabel('Frequency (MHz)');
ylabel('Amplitude');
ax2 = gca;
ax2.FontSize = 12;
box off;
movegui(f2, 'southeast');


% Export option: RD3/RAD or DZT or DT1/HD or IPRB/IPRH ====================
while 1
    prompt = {'1 = RD3,  2 = DZT,  3 = DT1,  4 = IPRB'};
    dlg_title = 'Choose GPR Format';
    answer = inputdlg(prompt, dlg_title, [1 45]);
    answer = str2double(answer);
    if isempty(answer)
        return
    elseif ~isnumeric(answer) || answer ~= 1 && answer ~= 2 ...
            && answer ~= 3 && answer ~= 4
        continue
    else
        gpr_format = answer;
        break
    end
end


wb = waitbar(0, 'Exporting...', 'Name', 'Exporting File');


% RAD / RD3, Mala GeoScience ==============================================
% Rad is the header file. In this file is all the important information
% such as the number of samples, traces, measurement intervals can be
% found.
% Rd3 is the data file. This file contains only the data (amplitude values)
% in a binary form.
if gpr_format == 1
    % Header structure
    HDR.fname                 = HDR.fname;                      % File name
    HDR.num_samp              = HDR.num_samp;                   % Number of samples
    HDR.samp_freq             = HDR.samp_freq;                  % Sampling frequency (MHz)
    HDR.frequency_steps       = 1;                              % Frequency steps
    HDR.signal_pos            = 0;                              % Signal position
    HDR.raw_signal_pos        = 0;                              % Raw signal position
    HDR.distance_flag         = 1;                              % Distance flag: 0 time interval, 1 distance interval
    HDR.time_flag             = 0;                              % Time flag    : 0 distance interval, 1 time interval
    HDR.program_flag          = 0;                              % Program flag
    HDR.external_flag         = 0;                              % External flag
    HDR.trac_int_sec          = 0;                              % Trace interval in seconds(only if Time flag = 1)
    HDR.trac_int              = HDR.trac_int;                   % Trace interval in meters (only if Distance flag = 1)
    HDR.operator              = 'Unknown';                      % Operator
    HDR.customer              = 'Unknown';                      % Customer
    HDR.site                  = 'gprMax';                       % Site
    HDR.antenna               = HDR.antenna;                    % Antenna name
    HDR.antenna_orientation   = 'NOT VALID FIELD';              % Antenna orientation
    HDR.ant_sep               = HDR.ant_sep;                    % Antenna seperation / Tx-Rx distance (m)
    HDR.comment               = '----';                         % Comment
    HDR.time_window           = HDR.time_window;                % Time window (ns)
    HDR.stacks                = 1;                              % Stacks
    HDR.stack_exponent        = 0;                              % Stack exponent
    HDR.stacking_time         = 0;                              % Stacking Time
    HDR.num_trac              = HDR.num_trac;                   % Number of traces
    HDR.stop_pos              = HDR.num_trac * HDR.trac_int;    % Stop position (m)
    HDR.system_calibration    = 0;
    HDR.start_pos             = 0;                              % Start position (m)
    HDR.short_flag            = 1;
    HDR.intermediate_flag     = 0;
    HDR.long_flag             = 0;
    HDR.preprocessing         = 0;
    HDR.high                  = 0;
    HDR.low                   = 0;
    HDR.fixed_increment       = 0;
    HDR.fixed_moves_up        = 0;
    HDR.fixed_moves_down      = 1;
    HDR.fixed_position        = 0;
    HDR.wheel_calibration     = 0;
    HDR.positive_direction    = 1;

    % RAD file
    fid = fopen([HDR.fname '.rad'], 'w');
    fprintf(fid, 'SAMPLES:%i\r\n', HDR.num_samp);
    fprintf(fid, 'FREQUENCY:%0.6f\r\n', HDR.samp_freq);
    fprintf(fid, 'FREQUENCY STEPS:%i\r\n', HDR.frequency_steps);
    fprintf(fid, 'SIGNAL POSITION:%0.6f\r\n', HDR.signal_pos);
    fprintf(fid, 'RAW SIGNAL POSITION:%i\r\n', HDR.raw_signal_pos);
    fprintf(fid, 'DISTANCE FLAG:%i\r\n', HDR.distance_flag);
    fprintf(fid, 'TIME FLAG:%i\r\n', HDR.time_flag);
    fprintf(fid, 'PROGRAM FLAG:%i\r\n', HDR.program_flag);
    fprintf(fid, 'EXTERNAL FLAG:%i\r\n', HDR.external_flag);
    fprintf(fid, 'TIME INTERVAL:%0.6f\r\n', HDR.trac_int_sec);
    fprintf(fid, 'DISTANCE INTERVAL:%0.6f\r\n', HDR.trac_int);
    fprintf(fid, 'OPERATOR:%s\r\n', HDR.operator);
    fprintf(fid, 'CUSTOMER:%s\r\n', HDR.customer);
    fprintf(fid, 'SITE:%s\r\n', HDR.site);
    fprintf(fid, 'ANTENNAS:%s\r\n', HDR.antenna);
    fprintf(fid, 'ANTENNA ORIENTATION:%s\r\n', HDR.antenna_orientation);
    fprintf(fid, 'ANTENNA SEPARATION:%0.6f\r\n', HDR.ant_sep);
    fprintf(fid, 'COMMENT:%s\r\n', HDR.comment);
    fprintf(fid, 'TIMEWINDOW:%0.6f\r\n', HDR.time_window);
    fprintf(fid, 'STACKS:%i\r\n', HDR.stacks);
    fprintf(fid, 'STACK EXPONENT:%i\r\n', HDR.stack_exponent);
    fprintf(fid, 'STACKING TIME:%0.6f\r\n', HDR.stacking_time);
    fprintf(fid, 'LAST TRACE:%i\r\n', HDR.num_trac);
    fprintf(fid, 'STOP POSITION:%0.6f\r\n', HDR.stop_pos);
    fprintf(fid, 'SYSTEM CALIBRATION:%0.6f\r\n', HDR.system_calibration);
    fprintf(fid, 'START POSITION:%0.6f\r\n', HDR.start_pos);
    fprintf(fid, 'SHORT FLAG:%i\r\n', HDR.short_flag);
    fprintf(fid, 'INTERMEDIATE FLAG:%i\r\n', HDR.intermediate_flag);
    fprintf(fid, 'LONG FLAG:%i\r\n', HDR.long_flag);
    fprintf(fid, 'PREPROCESSING:%i\r\n', HDR.preprocessing);
    fprintf(fid, 'HIGH:%i\r\n', HDR.high);
    fprintf(fid, 'LOW:%i\r\n', HDR.low);
    fprintf(fid, 'FIXED INCREMENT:%0.6f\r\n', HDR.fixed_increment);
    fprintf(fid, 'FIXED MOVES UP:%i\r\n', HDR.fixed_moves_up);
    fprintf(fid, 'FIXED MOVES DOWN:%i\r\n', 1);
    fprintf(fid, 'FIXED POSITION:%0.6f\r\n', HDR.fixed_moves_down);
    fprintf(fid, 'WHEEL CALIBRATION:%0.6f\r\n', HDR.wheel_calibration);
    fprintf(fid, 'POSITIVE DIRECTION:%i\r\n', HDR.positive_direction);
    fclose(fid);

    % RD3 file
    fid = fopen([HDR.fname '.rd3'], 'w');
    fwrite(fid, data, 'short');
    fclose(fid);


% DZT, Geophysical Survey Systems Inc. (GSSI) =============================
% Dzt is a binary file that consists of the file header with all the
% important information (number of samples, traces, channels, etc.)
% followed by the data section.
% All the information is contained in this file except the TxRx distance
% (antenna separation). There is a possibility that the official GSSI
% software has stored this information and by using the antenna name
% presents the correct one. All the other software does not detect the TxRx
% distance.
elseif gpr_format == 2
    % Header structure
    HDR.fname                 = HDR.fname;                      % File name
    HDR.tag                   = 255;                            % Header = 255
    HDR.data_offset           = 1024;                           % Offset to data from the beginning of file
    HDR.num_samp              = HDR.num_samp;                   % Number of samples
    HDR.data_format           = 16;                             % Bits per data word (8, 16, 32)
    HDR.binary_offset         = 32768;                          % Binary offset, 8 bit = 128, 16 bit = 32768
    HDR.scans_per_second      = 0;                              % Scans per second
    HDR.scans_per_meter       = 1 / HDR.trac_int;               % Scans per meter
    HDR.meters_per_mark       = 0;                              % Meters per mark
    HDR.zero_time_adjustment  = 0;                              % Time zero adjustment (ns)
    HDR.time_window           = HDR.time_window;                % Time window (with no corrections i.e zero time)
    HDR.scans_per_pass        = 0;                              % Scan per pass for 2D files

    HDR.createdate.sec        = 0 / 2;                          % Structure, date created
    HDR.createdate.min        = 0;
    HDR.createdate.hour       = 0;
    HDR.createdate.day        = 0;
    HDR.createdate.month      = 0;
    HDR.createdate.year       = 0 - 1980;

    date_time                 = clock;
    HDR.modifydate.sec        = date_time(6) / 2;               % Structure, date modified
    HDR.modifydate.min        = date_time(5);
    HDR.modifydate.hour       = date_time(4);
    HDR.modifydate.day        = date_time(3);
    HDR.modifydate.month      = date_time(2);
    HDR.modifydate.year       = date_time(1) - 1980;

    HDR.offset_to_range_gain  = 0;                              % Offset to range gain
    HDR.size_of_range_gain    = 0;                              % Size of range gain
    HDR.offset_to_text        = 0;                              % Offset to text
    HDR.size_of_text          = 0;                              % Size of text
    HDR.offset_to_proc_his    = 0;                              % Offset to processing history
    HDR.size_of_proc_his      = 0;                              % Size of processing history
    HDR.num_channels          = 1;                              % Number of channels
    HDR.dielectric_constant   = 8;                              % Dielectric constant (8 is a random number)
    HDR.top_position          = 0;                              % Top position

    c = 299792458;
    v = (c / sqrt(HDR.dielectric_constant)) * 10^-9;
    HDR.range_depth           = v * (HDR.time_window / 2);      % Range depth (m)

    HDR.reserved              = zeros(31, 1);                   % Reserved
    HDR.data_type             = 0;                              % Data type

    if length(HDR.antenna) == 14                                % Antenna name
        HDR.antenna           = HDR.antenna;
    elseif length(HDR.antenna) < 14
        if verLessThan('matlab', '9.1')
            HDR.antenna       = [HDR.antenna repmat(' ', ...
                                    1, 14 - length(HDR.antenna))];
        else
            HDR.antenna       = pad(HDR.antenna, 14, 'right');
        end
    elseif length(HDR.antenna) > 14
        HDR.antenna           = HDR.antenna(1 : 14);
    end

    HDR.channel_mask          = 0;                              % Channel mask

    if length(HDR.fname) == 12                                  % Raw file name (File name during survey)
        HDR.raw_file_name     = HDR.fname;
    elseif length(HDR.fname) < 12
        if verLessThan('matlab', '9.1')
            HDR.raw_file_name = [HDR.raw_file_name repmat(' ', ...
                                    1, 12 - length(HDR.raw_file_name))];
        else
            HDR.raw_file_name = pad(HDR.fname, 12, 'right');
        end
    elseif length(HDR.fname) > 12
        HDR.raw_file_name     = HDR.fname(1 : 12);
    end

    HDR.checksum              = 0;                              % Checksum
    HDR.num_gain_points       = 0;                              % Number of gain points
    HDR.range_gain_db         = [];                             % Range gain in db
    HDR.variable              = zeros(896, 1);

    % DZT file
    fid = fopen([HDR.fname '.dzt'], 'w');
    fwrite(fid, HDR.tag, 'ushort');
    fwrite(fid, HDR.data_offset, 'ushort');
    fwrite(fid, HDR.num_samp, 'ushort');
    fwrite(fid, HDR.data_format, 'ushort');
    fwrite(fid, HDR.binary_offset, 'ushort');
    fwrite(fid, HDR.scans_per_second, 'float');
    fwrite(fid, HDR.scans_per_meter, 'float');
    fwrite(fid, HDR.meters_per_mark, 'float');
    fwrite(fid, HDR.zero_time_adjustment, 'float');
    fwrite(fid, HDR.time_window, 'float');
    fwrite(fid, HDR.scans_per_pass, 'ushort');
    fwrite(fid, HDR.createdate.sec, 'ubit5');
    fwrite(fid, HDR.createdate.min, 'ubit6');
    fwrite(fid, HDR.createdate.hour, 'ubit5');
    fwrite(fid, HDR.createdate.day, 'ubit5');
    fwrite(fid, HDR.createdate.month, 'ubit4');
    fwrite(fid, HDR.createdate.year, 'ubit7');
    fwrite(fid, HDR.modifydate.sec, 'ubit5');
    fwrite(fid, HDR.modifydate.min, 'ubit6');
    fwrite(fid, HDR.modifydate.hour, 'ubit5');
    fwrite(fid, HDR.modifydate.day, 'ubit5');
    fwrite(fid, HDR.modifydate.month, 'ubit4');
    fwrite(fid, HDR.modifydate.year, 'ubit7');
    fwrite(fid, HDR.offset_to_range_gain, 'ushort');
    fwrite(fid, HDR.size_of_range_gain, 'ushort');
    fwrite(fid, HDR.offset_to_text, 'ushort');
    fwrite(fid, HDR.size_of_text, 'ushort');
    fwrite(fid, HDR.offset_to_proc_his, 'ushort');
    fwrite(fid, HDR.size_of_proc_his, 'ushort');
    fwrite(fid, HDR.num_channels, 'ushort');
    fwrite(fid, HDR.dielectric_constant, 'float');
    fwrite(fid, HDR.top_position, 'float');
    fwrite(fid, HDR.range_depth, 'float');
    fwrite(fid, HDR.reserved, 'char');
    fwrite(fid, HDR.data_type, 'char');
    fwrite(fid, HDR.antenna, 'char');
    fwrite(fid, HDR.channel_mask, 'ushort');
    fwrite(fid, HDR.raw_file_name, 'char');
    fwrite(fid, HDR.checksum, 'ushort');
    fwrite(fid, HDR.num_gain_points, 'ushort');
    fwrite(fid, HDR.range_gain_db, 'float');
    fwrite(fid, HDR.variable, 'char');

    fseek(fid, HDR.data_offset, 'bof');
    data = data + 2^15;
    fwrite(fid, data, 'ushort');
    fclose(fid);


% HD / DT1, Sensors & Software Inc. =======================================
% Hd is the header file. In this file all the important information such as
% the number of samples, traces, stacks, etc. can be found.
% Dt1 is the data file written in binary form. This file contains as many
% records as there are traces. Each record consists of a header and a data
% section. This means that also in this file there are stored information
% such as the number of samples, traces, etc.

elseif gpr_format == 3
    %Header structure of HD
    HDR.fname                 = HDR.fname;                      % File name
    HDR.file_tag              = 1234;                           % File tag = 1234
    HDR.system                = 'gprMax';                       % The system the data collected with

    date_time                 = clock;
    HDR.date                  = ([num2str(date_time(1)), '-' ...
                                    num2str(date_time(2)), '-' ...
                                        num2str(date_time(3))]);% Date

    HDR.num_trac              = HDR.num_trac;                   % Number of traces
    HDR.num_samp              = HDR.num_samp;                   % Number of samples
    HDR.time_zero_point       = 0;                              % Time zero point
    HDR.time_window           = HDR.time_window;                % Total time window (ns)
    HDR.start_position        = 0;                              % Start position (m)
    HDR.final_position        = (HDR.num_trac - 1) * HDR.trac_int;        % Stop position (m)
    HDR.trac_int              = HDR.trac_int;                   % Trace interval (m)
    HDR.pos_units             = 'm';                            % Position units
    HDR.nominal_freq          = HDR.centre_freq;                % Nominal freq. / Centre freq. (MHz)
    HDR.ant_sep               = HDR.ant_sep;                    % Antenna seperation / Tx-Rx distance (m)
    HDR.pulser_voltage        = 0;                              % Pulser voltage (V)
    HDR.stacks                = 1;                              % Number of stacks
    HDR.survey_mode           = 'Reflection';                   % Survey mode
    HDR.odometer              = 0;                              % Odometer Cal (t/m)
    HDR.stacking_type         = 'F1';                           % Stacking type
    HDR.dvl_serial            = '0000-0000-0000';               % DVL serial
    HDR.console_serial        = '000000000000';                 % Console serial
    HDR.tx_serial             = '0000-0000-0000';               % Transmitter serial
    HDR.rx_serial             = '0000-0000-0000';               % Receiver Serial

    % Header structure of DT1
    HDR.num_each_trac         = 1 : 1 : HDR.num_trac;           % Number of each trace 1, 2, 3, ... num_trac
    HDR.position              = 0 : HDR.trac_int : ...
                                    (HDR.num_trac - 1) * HDR.trac_int;    % Position of each trace (m)
    HDR.num_samp_each_trac    = zeros(1, HDR.num_trac) + HDR.num_samp;    % Number of samples of each trace
    HDR.elevation             = zeros(1, HDR.num_trac);         % Elevation / topography of each trace
    HDR.not_used1             = zeros(1, HDR.num_trac);         % Not used
    HDR.bytes                 = zeros(1, HDR.num_trac) + 2;     % Always 2 for Rev 3 firmware
    HDR.time_window_each_trac = zeros(1, HDR.num_trac) + HDR.time_window; % Time window of each trace (ns)
    HDR.stacks_each_trac      = ones(1, HDR.num_trac);          % Number of stacks each trace
    HDR.not_used2             = zeros(1, HDR.num_trac);         % Not used
    HDR.rsv_gps_x             = zeros(1, HDR.num_trac);         % Reserved for GPS X position (double*8 number)
    HDR.rsv_gps_y             = zeros(1, HDR.num_trac);         % Reserved for GPS Y position (double*8 number)
    HDR.rsv_gps_z             = zeros(1, HDR.num_trac);         % Reserved for GPS Z position (double*8 number)
    HDR.rsv_rx_x              = zeros(1, HDR.num_trac);         % Reserved for receiver x position
    HDR.rsv_rx_y              = zeros(1, HDR.num_trac);         % Reserved for receiver y position
    HDR.rsv_rx_z              = zeros(1, HDR.num_trac);         % Reserved for receiver z position
    HDR.rsv_tx_x              = zeros(1, HDR.num_trac);         % Reserved for transmitter x position
    HDR.rsv_tx_y              = zeros(1, HDR.num_trac);         % Reserved for transmitter y position
    HDR.rsv_tx_z              = zeros(1, HDR.num_trac);         % Reserved for transmitter z position
    HDR.time_zero             = zeros(1, HDR.num_trac);         % Time zero adjustment where: point(x) = point(x + adjustment)
    HDR.zero_flag             = zeros(1, HDR.num_trac);         % 0 = data ok, 1 = zero data
    HDR.num_channels          = zeros(1, HDR.num_trac);         % Number of channels
    HDR.time                  = zeros(1, HDR.num_trac);         % Time of day data collected in seconds past midnight
    HDR.comment_flag          = zeros(1, HDR.num_trac);         % Comment flag
    HDR.comment               = zeros(1, 24);                   % Comment

    % HD file
    fid = fopen([HDR.fname '.hd'], 'w');
    fprintf(fid, '%i\r\n', HDR.file_tag);
    fprintf(fid, 'Data Collected with %s\r\n', HDR.system);
    fprintf(fid, '%s\r\n', HDR.date);
    fprintf(fid, 'NUMBER OF TRACES   = %i\r\n', HDR.num_trac);
    fprintf(fid, 'NUMBER OF PTS/TRC  = %i\r\n', HDR.num_samp);
    fprintf(fid, 'TIMEZERO AT POINT  = %i\r\n', HDR.time_zero_point);
    fprintf(fid, 'TOTAL TIME WINDOW  = %0.6f\r\n', HDR.time_window);
    fprintf(fid, 'STARTING POSITION  = %0.6f\r\n', HDR.start_position);
    fprintf(fid, 'FINAL POSITION     = %0.6f\r\n', HDR.final_position);
    fprintf(fid, 'STEP SIZE USED     = %0.6f\r\n', HDR.trac_int);
    fprintf(fid, 'POSITION UNITS     = %s\r\n', HDR.pos_units);
    fprintf(fid, 'NOMINAL FREQUENCY  = %0.6f\r\n', HDR.nominal_freq);
    fprintf(fid, 'ANTENNA SEPARATION = %0.6f\r\n', HDR.ant_sep);
    fprintf(fid, 'PULSER VOLTAGE (V) = %0.6f\r\n', HDR.pulser_voltage);
    fprintf(fid, 'NUMBER OF STACKS   = %i\r\n', HDR.stacks);
    fprintf(fid, 'SURVEY MODE        = %s\r\n', HDR.survey_mode);
    fprintf(fid, 'ODOMETER CAL (t/m) = %0.6f\r\n', HDR.odometer);
    fprintf(fid, 'STACKING TYPE      = %s\r\n', HDR.stacking_type);
    fprintf(fid, 'DVL Serial#        = %s\r\n', HDR.dvl_serial);
    fprintf(fid, 'Console Serial#    = %s\r\n', HDR.console_serial);
    fprintf(fid, 'Transmitter Serial#= %s\r\n', HDR.tx_serial);
    fprintf(fid, 'Receiver Serial#   = %s\r\n', HDR.rx_serial);
    fclose(fid);

    % DT1 file
    fid = fopen([HDR.fname '.dt1'], 'w');
    for i = 1 : HDR.num_trac
        fwrite(fid, HDR.num_each_trac(i), 'float');
        fwrite(fid, HDR.position(i), 'float');
        fwrite(fid, HDR.num_samp_each_trac(i), 'float');
        fwrite(fid, HDR.elevation(i), 'float');
        fwrite(fid, HDR.not_used1(i), 'float');
        fwrite(fid, HDR.bytes(i), 'float');
        fwrite(fid, HDR.time_window_each_trac(i), 'float');
        fwrite(fid, HDR.stacks_each_trac(i), 'float');
        fwrite(fid, HDR.not_used2(i), 'float');
        fwrite(fid, HDR.rsv_gps_x(i), 'double');
        fwrite(fid, HDR.rsv_gps_y(i), 'double');
        fwrite(fid, HDR.rsv_gps_z(i), 'double');
        fwrite(fid, HDR.rsv_rx_x(i), 'float');
        fwrite(fid, HDR.rsv_rx_y(i), 'float');
        fwrite(fid, HDR.rsv_rx_z(i), 'float');
        fwrite(fid, HDR.rsv_tx_x(i), 'float');
        fwrite(fid, HDR.rsv_tx_y(i), 'float');
        fwrite(fid, HDR.rsv_tx_z(i), 'float');
        fwrite(fid, HDR.time_zero(i), 'float');
        fwrite(fid, HDR.zero_flag(i), 'float');
        fwrite(fid, HDR.num_channels(i), 'float');
        fwrite(fid, HDR.time(i), 'float');
        fwrite(fid, HDR.comment_flag(i), 'float');
        fwrite(fid, HDR.comment, 'char');

        fwrite(fid, data(:, i), 'short');
        if mod(i, 100) == 0
            waitbar(i / HDR.num_trac, wb, sprintf('Exporting... %.f%%', ...
                i / HDR.num_trac * 100))
        end
    end
    fclose(fid);


% IPRH / IPRB, Impulse Radar ==============================================
% IPRH is the header file. In this file is all the important information
% such as the number of samples, traces, measurement intervals can be
% found.
% IPRB is the data file. This file contains only the data (amplitude values)
% in a binary form.
elseif gpr_format == 4
% Header structure
    HDR.fname                 = HDR.fname;                      % File name
    HDR.hdr_version           = 20;                             % Header version
    HDR.data_format           = 16;                             % Data format 16 or 32 bit

    date_time                 = clock;
    HDR.date                  = ([num2str(date_time(1)), '-' ...
                                    num2str(date_time(2)), '-' ...
                                        num2str(date_time(3))]);% Date

    HDR.start_time            = '00:00:00';                     % Measurement start time
    HDR.stop_time             = '00:00:00';                     % Measurement end time
    HDR.antenna               = [num2str(HDR.centre_freq) ' MHz'];        % Antenna frequency (MHz)
    HDR.ant_sep               = HDR.ant_sep;                    % Antenna seperation / Tx-Rx distance (m)
    HDR.num_samp              = HDR.num_samp;                   % Number of samples
    HDR.signal_pos            = 0;                              % Signal position
    HDR.clipped_samps         = 0;                              % Clipped samples
    HDR.runs                  = 0;                              % Number of runs
    HDR.stacks                = 1;                              % Maximum number of stacks
    HDR.auto_stacks           = 1;                              % Autostacks (1 = On)
    HDR.samp_freq             = HDR.samp_freq;                  % Sampling frequency (MHz)
    HDR.time_window           = HDR.time_window;                % Total time window (ns)
    HDR.num_trac              = HDR.num_trac;                   % Number of traces
    HDR.trig_source           = 'wheel';                        % Trig source (wheel or time)
    HDR.trac_int_sec          = 0;                              % Trace interval if trig source is time (sec)
    HDR.trac_int_met          = HDR.trac_int;                   % Trace interval if trig source is wheel (m)
    HDR.user_trac_int         = HDR.trac_int;                   % User defined trace interval if trig source is wheel (m)
    HDR.stop_pos              = HDR.num_trac * HDR.trac_int;    % Stop position (meters or seconds) -> num_trac * trac_int
    HDR.wheel_name            = 'Cart';                         % Wheel name
    HDR.wheel_calibration     = 0;                              % Wheel calibration
    HDR.zero_lvl              = 0;                              % Zero level
    HDR.vel                   = 100;                            % The soil velocity (Selected in field m/usec). 100 is a random number
    HDR.preprocessing         = 'Unknown Preprocessing';        % Not in use
    HDR.comment               = '----';                         % Not in use
    HDR.antenna_FW            = '----';                         % Receiver firmware version
    HDR.antenna_HW            = '----';                         % Not in use
    HDR.antenna_FPGA          = '----';                         % Receiver FPGA version
    HDR.antenna_serial        = '----';                         % Receiver serial number
    HDR.software_version      = '----';                         % Software version
    HDR.positioning           = 0;                              % Positioning: (0 = no, 1 = TS, 2 = GPS)
    HDR.num_channel           = 1;                              % Number of channels
    HDR.channel_config        = 1;                              % This channel configuration
    HDR.ch_x_offset           = 0;                              % Channel position relative to ext.positioning
    HDR.ch_y_offset           = 0;                              % Channel position relative to ext.positioning
    HDR.meas_direction        = 1;                              % Meas. direction forward or backward
    HDR.relative_direction    = 0;                              % Direction to RL start(clockwise 360)
    HDR.relative_distance     = 0;                              % Distance from RL start to cross section
    HDR.relative_start        = 0;                              % DIstance from profile start to cross section

    % IPRH file
    fid = fopen([HDR.fname '.iprh'], 'w');
    fprintf(fid, 'HEADER VERSION: %i\r\n', HDR.hdr_version);
    fprintf(fid, 'DATA VERSION: %i\r\n', HDR.data_format);
    fprintf(fid, 'DATE: %s\r\n', HDR.date);
    fprintf(fid, 'START TIME: %s\r\n', HDR.start_time);
    fprintf(fid, 'STOP TIME: %s\r\n', HDR.stop_time);
    fprintf(fid, 'ANTENNA: %s\r\n', HDR.antenna);
    fprintf(fid, 'ANTENNA SEPARATION: %0.6f\r\n', HDR.ant_sep);
    fprintf(fid, 'SAMPLES: %i\r\n', HDR.num_samp);
    fprintf(fid, 'SIGNAL POSITION: %0.6f\r\n', HDR.signal_pos);
    fprintf(fid, 'CLIPPED SAMPLES: %i\r\n', HDR.clipped_samps);
    fprintf(fid, 'RUNS: %i\r\n', HDR.runs);
    fprintf(fid, 'MAX STACKS: %i\r\n', HDR.stacks);
    fprintf(fid, 'AUTOSTACKS: %i\r\n', HDR.auto_stacks);
    fprintf(fid, 'FREQUENCY: %0.6f\r\n', HDR.samp_freq);
    fprintf(fid, 'TIMEWINDOW: %0.6f\r\n', HDR.time_window);
    fprintf(fid, 'LAST TRACE: %i\r\n', HDR.num_trac);
    fprintf(fid, 'TRIG SOURCE: %s\r\n', HDR.trig_source);
    fprintf(fid, 'TIME INTERVAL: %0.6f\r\n', HDR.trac_int_sec);
    fprintf(fid, 'DISTANCE INTERVAL: %0.6f\r\n', HDR.trac_int_met);
    fprintf(fid, 'USER DISTANCE INTERVAL: %0.6f\r\n', HDR.user_trac_int);
    fprintf(fid, 'STOP POSITION: %0.6f\r\n', HDR.stop_pos);
    fprintf(fid, 'WHEEL NAME: %s\r\n', HDR.wheel_name);
    fprintf(fid, 'WHEEL CALIBRATION: %0.6f\r\n', HDR.wheel_calibration);
    fprintf(fid, 'ZERO LEVEL: %i\r\n', HDR.zero_lvl);
    fprintf(fid, 'SOIL VELOCITY: %i\r\n', HDR.vel);
    fprintf(fid, 'PREPROCESSING: %s\r\n', HDR.preprocessing);
    fprintf(fid, 'OPERATOR COMMENT: %s\r\n', HDR.comment);
    fprintf(fid, 'ANTENNA F/W: %s\r\n', HDR.antenna_FW);
    fprintf(fid, 'ANTENNA H/W: %s\r\n', HDR.antenna_HW);
    fprintf(fid, 'ANTENNA FPGA: %s\r\n', HDR.antenna_FPGA);
    fprintf(fid, 'ANTENNA SERIAL: %s\r\n', HDR.antenna_serial);
    fprintf(fid, 'SOFTWARE VERSION: %s\r\n', HDR.software_version);
    fprintf(fid, 'POSITIONING: %i\r\n', HDR.positioning);
    fprintf(fid, 'CHANNELS: %i\r\n', HDR.num_channel);
    fprintf(fid, 'CHANNEL CONFIGURATION: %i\r\n', HDR.channel_config);
    fprintf(fid, 'CH_X_OFFSET: %0.6f\r\n', HDR.ch_x_offset);
    fprintf(fid, 'CH_Y_OFFSET: %0.6f\r\n', HDR.ch_y_offset);
    fprintf(fid, 'MEASUREMENT DIRECTION: %i\r\n', HDR.meas_direction);
    fprintf(fid, 'RELATIVE DIRECTION: %i\r\n', HDR.relative_direction);
    fprintf(fid, 'RELATIVE DISTANCE: %0.6f\r\n', HDR.relative_distance);
    fprintf(fid, 'RELATIVE START: %0.6f\r\n', HDR.relative_start);
    fclose(fid);

    % IPRB file
    fid = fopen([HDR.fname '.iprb'], 'w');
    fwrite(fid, data, 'short');
    fclose(fid);
end
waitbar(1, wb, 'Done!!!');
pause(1);
close(wb);
