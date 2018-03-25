%gprMax_output_converter, Angelis Dimitrios 2017, 2018
%Converts the gprMax merged output HDF5 file to RD3, DZT, DT1 files.

%Last Modified 25/03/2018

clear;                           close all;                            clc;

%Select file. Keep file name, path name and file extension ================
[infile, path] = uigetfile('*.out', 'Select gprMax File', 'Multiselect', 'Off');
if isequal(infile, 0)
    erd = errordlg('No Input File');
    but = findobj(erd, 'Style', 'Pushbutton');
    delete(but);
    pause(1);   close(erd);
    HDR = [];   data = [];
    return
end
HDR.fname = strrep(lower(infile), '.out', '');
HDR.pname = path;
HDR.fext  = 'out';


%Read data from HDF5 file =================================================
dataex = h5read(infile, '/rxs/rx1/Ex');
dataey = h5read(infile, '/rxs/rx1/Ey');
dataez = h5read(infile, '/rxs/rx1/Ez');


%Field check ==============================================================
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


%Single to double precision ===============================================
data = double(data);


%The HDF5 file doesn't contain information about the Tx Rx distance and the 
%trace step. We need to provide this information ==========================
while 1
    prompt = {'Antenna Seperation (m)', 'Trace Step (m)'};
    dlg_title = 'gprMax Information';
    answer = inputdlg(prompt, dlg_title, [1 45]);
    answer = str2double(answer);
    if isempty(answer)
        HDR = [];   data = [];
        return
    elseif isnan(answer(1)) || isnan(answer(2)) || answer(2) == 0
        continue
    else
        break
    end
end


%Create header ===================================================== Basic information
info                         = h5info(infile);
attributes                   = info.Attributes;
if length(attributes) > 4
    gMaxV                    = cell2mat(h5readatt(infile, '/', 'gprMax'));
    HDR.antenna              = (['gprMax ', gMaxV]);                 %Antenna name
else
    HDR.antenna              = ('gprMax');                           %Antenna name
end

HDR.ant_sep                  = answer(1);                            %Tx Rx distance
[HDR.num_samp, HDR.num_trac] = size(data);                           %Samples & traces
HDR.trac_int                 = answer(2);                            %Trace interval
HDR.samp_int                 = h5readatt(infile, '/', 'dt') * 10^9;  %Sampling interval
HDR.samp_freq                = (1 / HDR.samp_int) * 10^3;            %Sampling Frequency
HDR.time_window              = HDR.num_samp * HDR.samp_int;          %Time window

x = 0 : HDR.trac_int : (HDR.num_trac - 1) * HDR.trac_int;
t = HDR.samp_int : HDR.samp_int : HDR.num_samp * HDR.samp_int;


%**************************************************************************
%******************************** Optional ********************************

%Resample to 1024 samples =================================================
%I usually perform this step for either 512 or 1024 samples (line 98)
%because many programs cant load files with many samples.
tx1 = 1 : HDR.num_samp;
fs1 = 1024 / HDR.num_samp;
data = resample(data, tx1, fs1);

[HDR.num_samp, ~] = size(data);                                      %New number of samples
HDR.samp_int      = HDR.time_window / HDR.num_samp;                  %New sampling interval
HDR.samp_freq     = (1 / HDR.samp_int) * 10^3;                       %New sampling frequency

%**************************************************************************
%**************************************************************************


%**************************************************************************
%******************************** Optional ********************************

%Data scale ===============================================================
data = data * 32767.5 ./ max(max(abs(data)));                        %signal * ((1 - 1 / 2^bitrate) * 32768) / max(signal)
data(data >= 32767.5) = 32767;
data = round(data);

%**************************************************************************
%**************************************************************************


%Plots ====================================================================
pmin = min(data(:));
pmax = max(data(:));

                                   %Bscan
f1 = figure('Name', 'Bscan', ...
    'NumberTitle', 'off', ...
    'Menubar', 'None', ...
    'Toolbar', 'Figure');

clims = [pmin pmax];
colormap (flipud(bone(256)));
im1 = imagesc(x, t, data, clims);
set(im1, 'cdatamapping', 'scaled');
title(HDR.fname);      xlabel('Distance (m)');         ylabel('Time (ns)');
ax1 = gca;             ax1.XAxisLocation = 'Top';      ax1.FontSize = 12;
box off;               movegui(f1, 'northeast');


                            %Frequency Spectrum
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
title(HDR.fname);      xlabel('Frequency (MHz)');      ylabel('Amplitude');
ax2 = gca;             ax2.FontSize = 12;
box off;               movegui(f2, 'southeast');


%Export option rd3 or dzt or dt1 ==========================================
while 1
    choice = questdlg('File Type', 'Export', ...
        'RD3', 'DZT', 'DT1', 'Default');
    switch choice
        case 'RD3'
            flg = 1;
            break
        case 'DZT'
            flg = 2;
            break
        case 'DT1'
            flg = 3;
            break
    end
end

wb = waitbar(0, 'Exporting...', 'Name', 'Exporting File');


%RAD / RD3, Mala GeoScience ===============================================
%Rad is the header file. In this file are contained all the important 
%information such as the number of traces, samples, measurement intervals...
%Rd3 is the data file. This file contain only the data (amplitudes) in a
%binary form.
if flg == 1
    %Header structure
    HDR.fname                 = HDR.fname;                     %File name
    HDR.num_samp              = HDR.num_samp;                  %Number of samples
    HDR.samp_freq             = HDR.samp_freq;                 %Sampling frequency
    HDR.frequency_steps       = 1;                             %Frequency steps
    HDR.signal_pos            = 0;
    HDR.raw_signal_pos        = 0;
    HDR.distance_flag         = 1;                             %Distance flag: 0 time interval, 1 distance interval
    HDR.time_flag             = 0;                             %Time flag    : 0 distance interval, 1 time interval
    HDR.program_flag          = 0;
    HDR.external_flag         = 0;
    HDR.trac_int_sec          = 0;                             %Trace interval in seconds(only if Time flag = 1)
    HDR.trac_int              = HDR.trac_int;                  %Trace interval in meters (only if Distance flag = 1)
    HDR.operator              = 'Unknown';
    HDR.customer              = 'Unknown';
    HDR.site                  = 'gprMax';
    HDR.antenna               = HDR.antenna;                   %Antenna name
    HDR.antenna_orientation   = 'NOT VALID FIELD';
    HDR.ant_sep               = HDR.ant_sep;                   %Antenna seperation (Tx-Rx distance)
    HDR.comment               = '-';
    HDR.time_window           = HDR.time_window;               %Time window
    HDR.stacks                = 0;                             %Stacks
    HDR.stack_exponent        = 0;                             %Stack exponent
    HDR.stacking_time         = 0;                             %Stacking Time
    HDR.num_trac              = HDR.num_trac;                  %Number of traces
    HDR.stop_pos              = HDR.num_trac * HDR.trac_int;   %Stop position (meters)
    HDR.system_calibration    = 0;
    HDR.start_pos             = 0;                             %Start position (meters)
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
    
    %RAD file
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

    %RD3 file
    fid = fopen([HDR.fname '.rd3'], 'w');
    fwrite(fid, data, 'short');
    fclose(fid);
    
    
%DZT, Geophysical Survey Systems Inc. (GSSI) ==============================
%Dzt is a binary file that consists of the file header with all the
%important information (number of traces, samples, channels, etc.) followed
%by the data section.
%Every information that is needed is contained in this file except the
%TxRx distance (antenna seperation). There is a possibility that the
%official GSSI software has stored this information and by using the
%antenna name presents the correct one. All the other software doesn't
%detect the TxRx distance.

elseif flg == 2
    %Header structure
    HDR.fname                 = HDR.fname;                     %File name
    HDR.tag                   = 255;                           %Header = 255
    HDR.data_offset           = 1024;                          %Offset to data from the beginning of file
    HDR.num_samp              = HDR.num_samp;                  %Number of samples
    HDR.bits_per_word         = 16;                            %Bits per data word (8, 16, 32)
    HDR.binary_offset         = 32768;                         %Binary offset, 8 bit = 128, 16 bit = 32768
    HDR.scans_per_second      = 0;                             %Scans per second
    HDR.scans_per_meter       = 1 / HDR.trac_int;              %Scans per meter
    HDR.meters_per_mark       = 0;                             %Meters per mark
    HDR.zero_time_adjustment  = 0;                             %Zero time adjustment (nanoseconds)
    HDR.time_window           = HDR.time_window;               %Time window (with no corrections i.e zero time)
    HDR.scans_per_pass        = 0;                             %Scan per pass for 2D files

    HDR.createdate.sec        = 0 / 2;                         %Structure, date created
    HDR.createdate.min        = 0;
    HDR.createdate.hour       = 0;
    HDR.createdate.day        = 0;
    HDR.createdate.month      = 0;
    HDR.createdate.year       = 0 - 1980;
        
    date_time                 = clock;
    HDR.modifydate.sec        = date_time(6) / 2;              %Structure, date modified
    HDR.modifydate.min        = date_time(5);
    HDR.modifydate.hour       = date_time(4);
    HDR.modifydate.day        = date_time(3);
    HDR.modifydate.month      = date_time(2);
    HDR.modifydate.year       = date_time(1) - 1980;
        
    HDR.offset_to_range_gain  = 0;                             %Offset to range gain
    HDR.size_of_range_gain    = 0;                             %Size of range gain
    HDR.offset_to_text        = 0;                             %Offset to text
    HDR.size_of_text          = 0;                             %Size of text
    HDR.offset_to_proc_his    = 0;                             %Offset to processing history
    HDR.size_of_proc_his      = 0;                             %Size of processing hisstory
    HDR.num_channels          = 1;                             %Number of channels
    HDR.dielectric_constant   = 8;                             %Dielectric constant (8 is a random number)
    HDR.top_position          = 0;                             %Top position
        
    c = 299792458;
    v = (c / sqrt(HDR.dielectric_constant)) * 10^-9;
    HDR.range_depth           = v * (HDR.time_window / 2);     %Range depth
        
    HDR.reserved              = zeros(31, 1);                  %Reserved
    HDR.data_type             = 0;                             
        
    if length(HDR.antenna) == 14                               %Antenna name
        HDR.antenna           = HDR.antenna;
    elseif length(HDR.antenna) < 14
        HDR.antenna           = pad(HDR.antenna, 14, 'right');
    elseif length(HDR.antenna) > 14
        HDR.antenna           = HDR.antenna(1 : 14);
    end
    
    HDR.channel_mask          = 0;                             %Channel mask

    if length(HDR.fname) == 12
        HDR.raw_file_name     = HDR.fname;                     %Raw file name (File name during survey)
    elseif length(HDR.fname) < 12
        HDR.raw_file_name     = pad(HDR.fname, 12, 'right');
    elseif length(HDR.fname) > 12
        HDR.raw_file_name     = HDR.fname(1:12);
    end
    
    HDR.checksum              = 0;                             %Checksum
    HDR.num_gain_points       = 0;                             %Number of gain points
    HDR.range_gain_db         = [];                            %Range gain in db
    HDR.variable              = zeros(896, 1);
    
    %DZT file
    fid = fopen([HDR.fname '.dzt'], 'w');
    fwrite(fid, HDR.tag, 'ushort');
    fwrite(fid, HDR.data_offset, 'ushort');
    fwrite(fid, HDR.num_samp, 'ushort');
    fwrite(fid, HDR.bits_per_word, 'ushort');
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
    
    
%HD / DT1, Sensors & Software Inc. ========================================
%Hd is the header file. In this file can be found all the important
%information such as the number of traces, samples, stacks, etc.
%Dt1 is the data file written in binary form. This file contains as many
%records as there are traces. Each record consists of a header section and
%a data section. That means that in this file there are also stored
%information such as the number of samples, traces, etc.
elseif flg == 3
    %Header structure of HD
    HDR.fname                 = HDR.fname;                     %File name
    HDR.file_tag              = 1234;                          %File tag = 1234
    HDR.system                = ['Data Collected with ' HDR.antenna];
    
    date_time                 = clock;
    HDR.date                  = ([num2str(date_time(3)), '/' ...
                                    num2str(date_time(2)), '/' ...
                                    num2str(date_time(1))]);   %Date
    
    HDR.num_trac              = HDR.num_trac;                  %Number of traces
    HDR.num_samp              = HDR.num_samp;                  %Number of samples
    HDR.time_zero_point       = 0;
    HDR.time_window           = HDR.time_window;               %Total time window
    HDR.start_position        = 0;
    HDR.final_position        = (HDR.num_trac - 1 ) * HDR.trac_int;
    HDR.trac_int              = HDR.trac_int;                  %Trace interval
    HDR.pos_units             = 'm';                           %Position units
    HDR.nominal_freq          = 'Unknown';                     %Nominal frequency
    HDR.ant_sep               = HDR.ant_sep;                   %Antenna seperation
    HDR.pulser_voltage        = 'Unknown';                     %Pulser voltage (V)
    HDR.stacks                = 0;                             %Number of stacks
    HDR.survey_mode           = 'Reflection';                  %Survey mode
    HDR.odometer              = 0;                             %Odometer Cal (t/m)
    HDR.stacking_type         = 'F1';                          %Stacking type (Not using 'Unknown' in case it causes any problems)
    HDR.dvl_serial            = '0000-0000-0000';              %DVL serial
    HDR.console_serial        = '000000000000';                %Console serial
    HDR.tx_serial             = '0000-0000-0000';              %Transmitter serial
    HDR.rx_serial             = '0000-0000-0000';              %Receiver Serial
    
    %Header structure of DT1
    HDR.num_each_trac         = 1 : 1 : HDR.num_trac;          %Number of each trace 1, 2, 3, ... num_trac
    HDR.position              = 0 : HDR.trac_int : ...
                                    (HDR.num_trac - 1) * HDR.trac_int;    %Position of each trace (Xaxis)
    HDR.num_samp_each_trac    = zeros(1, HDR.num_trac) + HDR.num_samp;    %Number of samples of each trace
    HDR.elevation             = zeros(1, HDR.num_trac);        %Elevation / topography of each trace;
    HDR.not_used1             = zeros(1, HDR.num_trac);        %Not used
    HDR.bytes                 = zeros(1, HDR.num_trac) + 2;    %Always 2 for Rev 3 firmware
    HDR.time_window_each_trac = zeros(1, HDR.num_trac) + HDR.time_window; % Time window of each trace
    HDR.stacks_each_trac      = zeros(1, HDR.num_trac);        %Number of stacks each trace
    HDR.not_used2             = zeros(1, HDR.num_trac);        %Not used
    HDR.rsv_gps_x             = zeros(1, HDR.num_trac);        %Reserved for GPS X position (double*8 number)
    HDR.rsv_gps_y             = zeros(1, HDR.num_trac);        %Reserved for GPS Y position (double*8 number)
    HDR.rsv_gps_z             = zeros(1, HDR.num_trac);        %Reserved for GPS Z position (double*8 number)
    HDR.rsv_rx_x              = zeros(1, HDR.num_trac);        %Reserved for receiver x position
    HDR.rsv_rx_y              = zeros(1, HDR.num_trac);        %Reserved for receiver y position
    HDR.rsv_rx_z              = zeros(1, HDR.num_trac);        %Reserved for receiver z position
    HDR.rsv_tx_x              = zeros(1, HDR.num_trac);        %Reserved for transmitter x position
    HDR.rsv_tx_y              = zeros(1, HDR.num_trac);        %Reserved for transmitter y position
    HDR.rsv_tx_z              = zeros(1, HDR.num_trac);        %Reserved for transmitter z position 
    HDR.time_zero             = zeros(1, HDR.num_trac);        %Time zero adjustment where: point(x) = point(x + adjustment)
    HDR.zero_flag             = zeros(1, HDR.num_trac);        %0 = data ok, 1 = zero data  
    HDR.num_channels          = zeros(1, HDR.num_trac);        %Number of channels
    HDR.time                  = zeros(1, HDR.num_trac);        %Time of day data collected in seconds past midnight  
    HDR.comment_flag          = zeros(1, HDR.num_trac);        %Comment flag
    HDR.comment               = zeros(1, 24);                  %Comment
    
    %HD file
    fid = fopen([HDR.fname '.hd'], 'w');
    fprintf(fid, '%i\r\n\n', HDR.file_tag);
    fprintf(fid, 'Data Collected with %s\r\n\n', HDR.system);
    fprintf(fid, '%s\r\n\n', HDR.date);
    fprintf(fid, 'NUMBER OF TRACES   = %i\r\n\n', HDR.num_trac);
    fprintf(fid, 'NUMBER OF PTS/TRC  = %i\r\n\n', HDR.num_samp);
    fprintf(fid, 'TIMEZERO AT POINT  = %i\r\n\n', HDR.time_zero_point);
    fprintf(fid, 'TOTAL TIME WINDOW  = %0.6f\r\n\n', HDR.time_window);
    fprintf(fid, 'STARTING POSITION  = %0.f\r\n\n', HDR.start_position);
    fprintf(fid, 'FINAL POSITION     = %0.6f\r\n\n', HDR.final_position);
    fprintf(fid, 'STEP SIZE USED     = %0.6f\r\n\n', HDR.trac_int);
    fprintf(fid, 'POSITION UNITS     = %s\r\n\n', HDR.pos_units);
    fprintf(fid, 'NOMINAL FREQUENCY  = %s\r\n\n', HDR.nominal_freq);
    fprintf(fid, 'ANTENNA SEPARATION = %0.6f\r\n\n', HDR.ant_sep);
    fprintf(fid, 'PULSER VOLTAGE (V) = %s\r\n\n', HDR.pulser_voltage);
    fprintf(fid, 'NUMBER OF STACKS   = %i\r\n\n', HDR.stacks);
    fprintf(fid, 'SURVEY MODE        = %s\r\n\n', HDR.survey_mode);
    fprintf(fid, 'ODOMETER CAL (t/m) = %0.6f\r\n\n', HDR.odometer);
    fprintf(fid, 'STACKING TYPE      = %s\r\n\n', HDR.stacking_type);
    fprintf(fid, 'DVL Serial#        = %s\r\n\n', HDR.dvl_serial);
    fprintf(fid, 'Console Serial#    = %s\r\n\n', HDR.console_serial);
    fprintf(fid, 'Transmitter Serial#= %s\r\n\n', HDR.tx_serial);
    fprintf(fid, 'Receiver Serial#   = %s\r\n', HDR.rx_serial);
    fclose(fid);
    
    %DT1 file
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
        
        fwrite(fid, data(:, i) , 'short');
        if mod(i, 10) == 0
            waitbar(i / HDR.num_trac, wb, sprintf('Exporting... %.f%%', i / HDR.num_trac * 100))
        end
    end
    fclose(fid);
end
waitbar(1, wb, 'Done!!!');    pause(1);    close(wb);