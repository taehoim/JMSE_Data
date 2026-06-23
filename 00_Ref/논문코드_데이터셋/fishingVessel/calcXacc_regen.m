% calcXacc_regen.m
% MATLAB/Octave parametrized port of calculateXacc_dataset.m.
% Globs an input dir of 6Dof_*.csv files, computes the equivalent heeling angle
% Xacc = sqrt(phi^2 + theta^2) and the righting arm GZ = GM_T * sin(Xacc), and
% writes *_withXacc.csv (preserving any _r%d realization suffix) to an output dir.
% Run via MATLAB R2024b (matlab.exe); kept engine-neutral so it runs under Octave too.
%
% This script does NOT modify calculateXacc_dataset.m. The GM hydrostatics are
% IDENTICAL (Cb=0.68, Cw=0.80, KG=0.60*T, Munro-Smith I_T, KB, BG, BM_T, GM_T).
% It uses no MSS functions (pure CSV/regex/arithmetic), so it needs no addpath.
% The ONLY parametrization is the GM floor:
%
%   GM_FLOOR = 0.30  -> clamped set (Task A; reproduces published behaviour)
%   GM_FLOOR = 0     -> unclamped set (Task B; removes the 0.30 m hydrostatic floor,
%                       relevant for the 10 t hull whose true GM = 0.258 m)
%
% Switches (may be pre-set by the caller, e.g. via matlab.exe -batch):
%   INPUT_DIR, OUTPUT_DIR, GM_FLOOR
%
% Run from the fishingVessel dir (clamped, over the multi-realization set):
%   cd "00_Ref/논문코드_데이터셋/fishingVessel" && matlab.exe -batch \
%     "INPUT_DIR='6Dof_dataset_multi'; OUTPUT_DIR='6Dof_dataset_multi_withXacc'; GM_FLOOR=0.30; calcXacc_regen" 2>&1 | tr -d '\r'
% Unclamped (Task B):
%   cd "00_Ref/논문코드_데이터셋/fishingVessel" && matlab.exe -batch \
%     "INPUT_DIR='6Dof_dataset_multi'; OUTPUT_DIR='6Dof_dataset_multi_unclamped_withXacc'; GM_FLOOR=0; calcXacc_regen" 2>&1 | tr -d '\r'

% ------------------------------------------------------------------------------------
% Switches (guarded with exist() so the script runs standalone with defaults).
% ------------------------------------------------------------------------------------
if ~exist('INPUT_DIR',  'var'); INPUT_DIR  = '6Dof_dataset_multi';            end
if ~exist('OUTPUT_DIR', 'var'); OUTPUT_DIR = '6Dof_dataset_multi_withXacc';   end
if ~exist('GM_FLOOR',   'var'); GM_FLOOR   = 0.30;                            end  % 0 -> unclamped

% Resolve dirs relative to this script's location if they are not absolute.
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir); this_dir = pwd; end
input_dir  = local_resolve(INPUT_DIR,  this_dir);
output_dir = local_resolve(OUTPUT_DIR, this_dir);

fprintf('========================================\n');
fprintf('Xacc Calculator (Octave port)\n');
fprintf('========================================\n');
fprintf('Input dir : %s\n', input_dir);
fprintf('Output dir: %s\n', output_dir);
fprintf('GM_FLOOR  : %.3f m  (%s)\n\n', GM_FLOOR, ...
    iif(GM_FLOOR > 0, 'clamped', 'UNCLAMPED'));

if ~exist(input_dir, 'dir')
    error('Input directory does not exist: %s', input_dir);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% ------------------------------------------------------------------------------------
% Find all 6Dof_*.csv files (skip any *_withXacc.csv that may already exist)
% ------------------------------------------------------------------------------------
csv_files = dir(fullfile(input_dir, '6Dof_*.csv'));
csv_files = csv_files(~cellfun(@(n) ~isempty(strfind(n, '_withXacc')), {csv_files.name}));
if isempty(csv_files)
    error('No 6Dof_*.csv files found in: %s', input_dir);
end
fprintf('Found %d CSV files\n\n', length(csv_files));

% ------------------------------------------------------------------------------------
% Process each file
% ------------------------------------------------------------------------------------
for file_idx = 1:length(csv_files)
    csv_filename = csv_files(file_idx).name;
    csv_filepath = fullfile(input_dir, csv_filename);

    fprintf('----------------------------------------\n');
    fprintf('File %d/%d: %s\n', file_idx, length(csv_files), csv_filename);

    % Read CSV (skip the header row; numeric columns).
    data = local_readcsv(csv_filepath);   % returns matrix [time u v w p q r phi theta]
    time  = data(:,1);
    u     = data(:,2);  v = data(:,3);  w = data(:,4);
    p     = data(:,5);  q = data(:,6);  r = data(:,7);
    phi   = data(:,8);  theta = data(:,9);
    n_samples = length(time);

    % --- Parse vessel parameters from filename --------------------------------
    % Accept both the published name (..._Hs%.1f.csv) and the realization name
    % (..._Hs%.1f_r%d.csv). Capture the optional _r%d so we can preserve it.
    tokens = regexp(csv_filename, ...
        '6Dof_(\d+)ton_L([\d.]+)_B([\d.]+)_T([\d.]+)_Hs([\d.]+)(_r\d+)?\.csv', 'tokens');
    if ~isempty(tokens)
        tonnage = str2double(tokens{1}{1});
        L  = str2double(tokens{1}{2});
        B  = str2double(tokens{1}{3});
        T  = str2double(tokens{1}{4});
        Hs = str2double(tokens{1}{5});
        r_suffix = tokens{1}{6};   % '' or '_r<idx>'
        fprintf('  Vessel: %dton L=%.2f B=%.2f T=%.2f Hs=%.1f  suffix=%s\n', ...
            tonnage, L, B, T, Hs, iif(isempty(r_suffix), '(none)', r_suffix));
    else
        tonnage = NaN; L = NaN; B = NaN; T = NaN; Hs = NaN; r_suffix = '';
        warning('Could not parse vessel parameters from filename: %s', csv_filename);
    end

    % --- GM_T hydrostatics (identical to calculateXacc_dataset.m) --------------
    if ~isnan(tonnage)
        g = 9.81; rho = 1025;
        m = tonnage * 1000;
        Cb = 0.68;  nabla = Cb * L * B * T;
        Cw = 0.80;  Awp = Cw * B * L;

        KG = 0.60 * T;
        r_bg = [-0.05*L, 0, -(KG - T)]';

        KB = (1/3) * (5*T/2 - nabla/Awp);
        r_bb = [-0.05*L, 0, T - KB]';
        BG = r_bb(3) - r_bg(3);

        k_munro_smith = (6 * Cw^3) / ((1+Cw) * (1+2*Cw));
        I_T = k_munro_smith * (B^3 * L) / 12;

        BM_T = I_T / nabla;
        GM_T = BM_T - BG;

        % Parametrized floor: GM_FLOOR=0.30 -> published clamp; GM_FLOOR=0 -> off.
        if GM_T < GM_FLOOR
            GM_T = GM_FLOOR;
        end
        fprintf('  GM_T = %.4f m\n', GM_T);
    else
        GM_T = NaN;
    end

    % --- Xacc and GZ ----------------------------------------------------------
    Xacc = sqrt(phi.^2 + theta.^2);
    if ~isnan(GM_T)
        GZ = GM_T * sin(Xacc);     % exact formula (matches original)
    else
        GZ = NaN(size(Xacc));
    end

    % --- Output filename (preserve _r%d suffix) -------------------------------
    if ~isnan(tonnage)
        output_filename = sprintf('6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f%s_withXacc.csv', ...
            tonnage, L, B, T, Hs, r_suffix);
    else
        [~, basename, ~] = fileparts(csv_filename);
        output_filename = [basename, '_withXacc.csv'];
    end
    output_filepath = fullfile(output_dir, output_filename);

    % --- Write CSV ------------------------------------------------------------
    fid = fopen(output_filepath, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', output_filepath);
    end
    if ~isnan(GM_T)
        fprintf(fid, 'time,u,v,w,p,q,r,phi,theta,Xacc,GZ\n');
        for i = 1:n_samples
            fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                time(i), u(i), v(i), w(i), p(i), q(i), r(i), phi(i), theta(i), Xacc(i), GZ(i));
        end
    else
        fprintf(fid, 'time,u,v,w,p,q,r,phi,theta,Xacc\n');
        for i = 1:n_samples
            fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                time(i), u(i), v(i), w(i), p(i), q(i), r(i), phi(i), theta(i), Xacc(i));
        end
    end
    fclose(fid);
    fprintf('  Wrote: %s (%d rows)\n\n', output_filepath, n_samples);
end

fprintf('========================================\n');
fprintf('Done. Output in: %s\n', output_dir);
fprintf('========================================\n');

% ====================================================================================
% Local helper functions (Octave allows functions at the end of a script file).
% ====================================================================================
function out = iif(cond, a, b)
    if cond; out = a; else; out = b; end
end

function p = local_resolve(d, base)
    % Return d as-is if absolute (leading / or Windows drive X:\ or X:/), else
    % join it onto base (the script directory).
    is_abs = ~isempty(d) && ( d(1) == '/' || d(1) == '\' || ...
        (length(d) >= 2 && isletter(d(1)) && d(2) == ':') );
    if is_abs
        p = d;
    else
        p = fullfile(base, d);
    end
end

function M = local_readcsv(path)
    % Read a numeric CSV that has a one-line text header. Uses dlmread to skip
    % the header (row offset 1). Works in both Octave and MATLAB.
    M = dlmread(path, ',', 1, 0);
end
