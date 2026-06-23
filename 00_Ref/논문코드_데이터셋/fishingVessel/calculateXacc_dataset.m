% calculateXacc_dataset.m
% This script calculates equivalent heeling angle Xacc from existing 6-DOF CSV files
% and creates new CSV files with Xacc column added
%
% Formula: Xacc = sqrt(phi^2 + theta^2)  [rad]
% GZ = GM * Xacc  (for small angles, GZ ≈ GM * sin(Xacc))
%
% Author: Based on MSS toolbox
% Date: 2025

clearvars;
close all;

fprintf('========================================\n');
fprintf('Xacc Dataset Calculator\n');
fprintf('========================================\n\n');

%% Input/Output directory setup
input_dir = 'D:\MSS_6\fishingVessel\6Dof_dataset_ton';
output_dir = 'D:\MSS_6\fishingVessel\6Dof_dataset_withXacc_ton';

if ~exist(input_dir, 'dir')
    error('Input directory does not exist: %s', input_dir);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

fprintf('Input directory: %s\n', input_dir);
fprintf('Output directory: %s\n\n', output_dir);

%% Find all CSV files
csv_files = dir(fullfile(input_dir, '6Dof_*.csv'));
if isempty(csv_files)
    error('No CSV files found in: %s', input_dir);
end

fprintf('Found %d CSV files\n\n', length(csv_files));

%% Process each CSV file
for file_idx = 1:length(csv_files)
    csv_filename = csv_files(file_idx).name;
    csv_filepath = fullfile(input_dir, csv_filename);
    
    fprintf('----------------------------------------\n');
    fprintf('Processing file %d/%d: %s\n', file_idx, length(csv_files), csv_filename);
    fprintf('----------------------------------------\n');
    
    % Read CSV file
    fprintf('Reading CSV file...\n');
    try
        data = readtable(csv_filepath);
    catch ME
        warning('Failed to read %s: %s', csv_filename, ME.message);
        continue;
    end
    
    % Extract data
    time = data.time;
    u = data.u;
    v = data.v;
    w = data.w;
    p = data.p;
    q = data.q;
    r = data.r;
    phi = data.phi;
    theta = data.theta;
    
    n_samples = length(time);
    fprintf('  Total samples: %d\n', n_samples);
    
    % Extract vessel parameters from filename
    % Format: 6Dof_{tonnage}ton_L{L}_B{B}_T{T}_Hs{Hs}.csv
    tokens = regexp(csv_filename, '6Dof_(\d+)ton_L([\d.]+)_B([\d.]+)_T([\d.]+)_Hs([\d.]+)\.csv', 'tokens');
    if ~isempty(tokens)
        tonnage = str2double(tokens{1}{1});
        L = str2double(tokens{1}{2});
        B = str2double(tokens{1}{3});
        T = str2double(tokens{1}{4});
        Hs = str2double(tokens{1}{5});
        
        fprintf('  Vessel: %d ton, L=%.2fm, B=%.2fm, T=%.2fm, Hs=%.1fm\n', ...
            tonnage, L, B, T, Hs);
    else
        tonnage = NaN;
        L = NaN;
        B = NaN;
        T = NaN;
        Hs = NaN;
        warning('Could not parse vessel parameters from filename: %s', csv_filename);
    end
    
    % Calculate GM_T (Transverse Metacentric Height)
    % Using the same calculation as in fishingVessel.m
    if ~isnan(tonnage)
        % Constants
        g = 9.81;
        rho = 1025;
        
        % Ship parameters
        m = tonnage * 1000;  % Mass in kg
        Cb = 0.68;  % Block coefficient
        nabla = Cb * L * B * T;  % Volume displacement
        Cw = 0.80;  % Waterplane area coefficient
        Awp = Cw * B * L;  % Waterplane area
        
        % Center of gravity
        KG = 0.60 * T;  % Vertical position of CG above keel
        r_bg = [-0.05*L, 0, -(KG - T)]';  % CG relative to CO
        
        % Center of buoyancy
        KB = (1/3) * (5*T/2 - nabla/Awp);
        r_bb = [-0.05*L, 0, T - KB]';
        BG = r_bb(3) - r_bg(3);
        
        % Moments of inertia of waterplane area
        k_munro_smith = (6 * Cw^3) / ((1+Cw) * (1+2*Cw));
        I_T = k_munro_smith * (B^3 * L) / 12;
        
        % Metacentric heights
        BM_T = I_T / nabla;
        GM_T = BM_T - BG;
        
        % Ensure positive GM
        if GM_T < 0.3
            GM_T = 0.3;
        end
        
        fprintf('  GM_T (Transverse Metacentric Height): %.4f m\n', GM_T);
    else
        GM_T = NaN;
    end
    
    % Calculate Xacc (Equivalent Heeling Angle)
    % Formula: Xacc = sqrt(phi^2 + theta^2)  [rad]
    fprintf('Calculating Xacc...\n');
    Xacc = sqrt(phi.^2 + theta.^2);
    
    % Calculate GZ (Righting Arm) if GM_T is available
    % Formula options:
    %   1. GZ = GM_T * Xacc  (small angle approximation, linear)
    %   2. GZ = GM_T * sin(Xacc)  (exact formula, nonlinear)
    % 
    % Using sin(Xacc) for more accurate calculation (works for all angles)
    % For small angles: sin(Xacc) ≈ Xacc, so both formulas are similar
    if ~isnan(GM_T)
        % Exact formula using sin (more accurate for larger angles)
        GZ_sin = GM_T * sin(Xacc);
        
        % Linear approximation (for comparison)
        GZ_linear = GM_T * Xacc;
        
        % Use exact formula
        GZ = GZ_sin;
        
        fprintf('  Calculating GZ using GM_T = %.4f m\n', GM_T);
        fprintf('  Formula: GZ = GM_T * sin(Xacc) (exact)\n');
        fprintf('  Alternative: GZ = GM_T * Xacc (linear, for small angles)\n');
    else
        GZ = NaN(size(Xacc));
        fprintf('  Warning: GM_T not available, GZ will be NaN\n');
    end
    
    % Statistics
    fprintf('\nStatistics:\n');
    fprintf('  phi: mean=%.4f rad, std=%.4f rad, max=%.4f rad (%.1f deg)\n', ...
        mean(phi), std(phi), max(abs(phi)), rad2deg(max(abs(phi))));
    fprintf('  theta: mean=%.4f rad, std=%.4f rad, max=%.4f rad (%.1f deg)\n', ...
        mean(theta), std(theta), max(abs(theta)), rad2deg(max(abs(theta))));
    fprintf('  Xacc: mean=%.4f rad, std=%.4f rad, max=%.4f rad (%.1f deg)\n', ...
        mean(Xacc), std(Xacc), max(Xacc), rad2deg(max(Xacc)));
    if ~isnan(GM_T)
        fprintf('  GZ: mean=%.4f m, std=%.4f m, max=%.4f m\n', ...
            mean(GZ), std(GZ), max(GZ));
    end
    
    % Create output filename
    if ~isnan(tonnage)
        output_filename = sprintf('6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f_withXacc.csv', ...
            tonnage, L, B, T, Hs);
    else
        [~, basename, ~] = fileparts(csv_filename);
        output_filename = [basename, '_withXacc.csv'];
    end
    output_filepath = fullfile(output_dir, output_filename);
    
    % Write CSV file with Xacc and GZ columns
    fprintf('\nWriting CSV file: %s\n', output_filepath);
    
    fid = fopen(output_filepath, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', output_filepath);
    end
    
    % Write header
    if ~isnan(GM_T)
        fprintf(fid, 'time,u,v,w,p,q,r,phi,theta,Xacc,GZ\n');
    else
        fprintf(fid, 'time,u,v,w,p,q,r,phi,theta,Xacc\n');
    end
    
    % Write data (write in chunks to avoid memory issues)
    chunk_size = 100000;
    
    for chunk_start = 1:chunk_size:n_samples
        chunk_end = min(chunk_start + chunk_size - 1, n_samples);
        chunk_idx = chunk_start:chunk_end;
        
        for i = chunk_idx
            if ~isnan(GM_T)
                fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                    time(i), u(i), v(i), w(i), p(i), q(i), r(i), phi(i), theta(i), Xacc(i), GZ(i));
            else
                fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                    time(i), u(i), v(i), w(i), p(i), q(i), r(i), phi(i), theta(i), Xacc(i));
            end
        end
        
        if mod(chunk_start, 500000) == 1 && chunk_start > 1
            fprintf('  Progress: %.1f%%\n', 100*chunk_start/n_samples);
        end
    end
    
    fclose(fid);
    
    fprintf('CSV file saved: %s\n', output_filepath);
    file_info = dir(output_filepath);
    fprintf('File size: %.2f MB\n', file_info.bytes / 1024 / 1024);
    fprintf('Total rows: %d\n\n', n_samples);
    
end

fprintf('========================================\n');
fprintf('Xacc Dataset calculation completed!\n');
fprintf('========================================\n');
fprintf('Total files processed: %d\n', length(csv_files));
fprintf('Output directory: %s\n\n', output_dir);

