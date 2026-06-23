% generate6Dof_dataset.m
% This script generates a 6-DOF dataset for fishing vessels with different tonnages
% (10, 20, 30, 40, 50 tons) with 5000 seconds of data at 1Hz
%
% Output: CSV files containing 6-DOF simulation data (u, v, w, p, q, r, phi, theta)
% WITHOUT Xacc calculation
%
% Author: Based on MSS toolbox
% Date: 2025

clearvars;
close all;

fprintf('========================================\n');
fprintf('6-DOF Dataset Generator (No Xacc)\n');
fprintf('========================================\n\n');

%% Output directory setup
output_dir = 'D:\MSS_6\fishingVessel\6Dof_dataset_ton';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end
fprintf('Output directory: %s\n\n', output_dir);

%% Configuration
% Tonnage range (10톤 단위로 총 5개: 10, 20, 30, 40, 50톤)
tonnages = [10, 20, 30, 40, 50];  % 10톤 단위로 5개

% Wave parameters (PM spectrum) - Multiple Hs values
Hs_values = [5.0];  % Significant wave heights (m)
T0 = 8.0;              % Modal period (s) - 평범한 바다
beta_wave = deg2rad(140);  % Wave direction (rad)

% Simulation parameters
h = 1;            % Sampling time [s] (1Hz)
T_final = 20000;   % Final simulation time [s] (5000초)

fprintf('Configuration:\n');
fprintf('  Tonnages: %s\n', mat2str(tonnages));
fprintf('  Wave heights (Hs): %s m\n', mat2str(Hs_values));
fprintf('  Wave: T0=%.2fs, Direction=%.1f deg\n', T0, rad2deg(beta_wave));
fprintf('  Sampling: %.4fs (%.1f Hz), Duration: %.1fs\n', h, 1/h, T_final); 
fprintf('  Note: Using h=1s. Angle wrapping and divergence checks are enabled for stability.\n');
fprintf('  Total samples per vessel: %d\n', round(T_final/h));
fprintf('  Total combinations: %d (Hs) × %d (tonnages) = %d files\n\n', ...
    length(Hs_values), length(tonnages), length(Hs_values) * length(tonnages));

%% Generate dataset for each Hs and tonnage combination
total_combinations = length(Hs_values) * length(tonnages);
current_combination = 0;

for hs_idx = 1:length(Hs_values)
    Hs = Hs_values(hs_idx);
    
    fprintf('========================================\n');
    fprintf('Processing Hs = %.1f m\n', Hs);
    fprintf('========================================\n\n');
    
    for t_idx = 1:length(tonnages)
        tonnage = tonnages(t_idx);
        current_combination = current_combination + 1;
        
        fprintf('----------------------------------------\n');
        fprintf('Processing combination %d/%d: Hs=%.1fm, %d ton\n', ...
            current_combination, total_combinations, Hs, tonnage);
        fprintf('----------------------------------------\n');
        
        % Calculate ship dimensions (actual dimensions for fishing vessels)
        if tonnage == 10
            L = 7.93; B = 1.90; T = 0.95;
        elseif tonnage == 20
            L = 9.99; B = 2.40; T = 1.20;
        elseif tonnage == 30
            L = 11.43; B = 2.74; T = 1.37;
        elseif tonnage == 40
            L = 12.58; B = 3.02; T = 1.51;
        elseif tonnage == 50
            L = 13.56; B = 3.25; T = 1.63;
        else
            % Empirical formulas for other tonnages
            L = 3.5 * (tonnage^0.4);
            B = 0.24 * L;  % B ≈ 0.24·L
            T = 0.12 * L;  % T ≈ 0.12·L
        end
        
        fprintf('Dimensions: L=%.2fm, B=%.2fm, T=%.2fm\n', L, B, T);
        
        % Clear persistent variables
        clear fishingVessel;
        clear waveForceRAO;
        
        % Time vector
        t = 0:h:T_final;
        nTimeSteps = length(t);
        
        % Wave spectrum setup
        maxFreq = 3.0;
        numFreqIntervals = 100;
        spreadingFlag = false;
        numDirections = 1;
        
        % Convert T0 to w0 (peak frequency in rad/s)
        w0 = 2*pi/T0;
        [S_M, Omega, Amp, ~, ~, mu] = waveDirectionalSpectrum('Modified PM', ...
            [Hs, w0], numFreqIntervals, maxFreq, spreadingFlag, numDirections);
        
        % Initialize
        x = zeros(12,1);  % Initial state
        simdata = zeros(nTimeSteps, 9);  % [t, u, v, w, p, q, r, phi, theta]
        wave_elevation = zeros(nTimeSteps, 1);
        
        % Initialize random phases for wave
        rng(12345 + hs_idx * 100 + t_idx);  % Different seed for each combination
        if size(S_M, 2) == 1
            randomPhases = 2*pi*rand(numFreqIntervals, 1);
        else
            randomPhases = 2*pi*rand(numFreqIntervals, numDirections);
        end
        
        % Extract initial states
        nu = x(1:6);
        eta = x(7:12);
        
        % Simulation loop
        fprintf('Simulating...\n');
        tic;
        
        for i = 1:nTimeSteps
            
            % Compute wave elevation
            if size(S_M, 2) == 1
                wave_elevation(i) = sum(Amp .* cos(Omega * t(i) + randomPhases));
            else
                wave_elevation(i) = sum(sum(Amp .* cos(Omega * t(i) + randomPhases)));
            end
            
            % Improved wave force model (Froude-Krylov approximation)
            rho = 1025;
            g = 9.81;
            nabla = 0.68 * L * B * T;  % Block coefficient 0.68
            Awp = 0.80 * B * L;  % Waterplane area coefficient 0.80
            
            % Wave number (deep water approximation)
            w0_wave = 2*pi/T0;
            k = w0_wave^2 / g;  % Wave number
            
            % Froude-Krylov forces (scaled for realistic motion)
            scale_surge = 0.005;  % Scale down surge force
            scale_sway = 0.005;   % Scale down sway force
            scale_heave = 0.05;   % Scale down heave force
            scale_roll = 0.02;    % Scale down roll moment
            scale_pitch = 0.02;   % Scale down pitch moment
            
            % Surge force: pressure gradient effect
            F_wave_surge = scale_surge * rho * g * nabla * k * wave_elevation(i) * cos(beta_wave);
            
            % Sway force: lateral wave effect
            F_wave_sway = scale_sway * rho * g * nabla * k * wave_elevation(i) * sin(beta_wave);
            
            % Heave force: direct buoyancy change
            F_wave_heave = scale_heave * rho * g * Awp * wave_elevation(i);
            
            % Roll moment: wave-induced heeling moment
            M_wave_roll = scale_roll * rho * g * nabla * B * wave_elevation(i) * sin(beta_wave);
            
            % Pitch moment: wave-induced trim moment
            M_wave_pitch = scale_pitch * rho * g * nabla * L * wave_elevation(i) * cos(beta_wave);
            
            % Yaw moment: small, can be neglected
            M_wave_yaw = 0;
            
            tau_wave = [F_wave_surge; F_wave_sway; F_wave_heave; M_wave_roll; M_wave_pitch; M_wave_yaw];
            
            % Store simulation data
            simdata(i, :) = [t(i), nu(1), nu(2), nu(3), nu(4), nu(5), nu(6), ...
                eta(4), eta(5)];
            
            % RK4 integration
            x = rk4(@fishingVessel, h, x, tau_wave, L, B, T, tonnage);
            
            % Extract states
            nu = x(1:6);
            eta = x(7:12);
            
            % Wrap Euler angles to [-pi, pi) to prevent numerical overflow
            eta(4) = ssa(eta(4));  % phi (roll)
            eta(5) = ssa(eta(5));  % theta (pitch)
            eta(6) = ssa(eta(6));  % psi (yaw)
            
            % Check for numerical instability (state divergence)
            % Reasonable limits for small fishing vessels
            max_vel = 20;      % Maximum velocity (m/s)
            max_ang_vel = 20;  % Maximum angular velocity (rad/s)
            max_angle = pi/2 * 1.1;  % Maximum angle (99 degrees, with margin for numerical error)
            
            % Check if any value exceeds limits
            vel_exceeded = any(abs(nu(1:3)) > max_vel);
            ang_vel_exceeded = any(abs(nu(4:6)) > max_ang_vel);
            % Check angles after wrapping - allow small numerical errors
            angle_exceeded = any(abs(eta(4:6)) > max_angle);
            
            if vel_exceeded || ang_vel_exceeded || angle_exceeded
                % Only warn for significant divergence, not small numerical errors
                % Check if values are truly diverging (not just slightly over threshold)
                significant_divergence = vel_exceeded || ang_vel_exceeded || ...
                    any(abs(eta(4:6)) > pi/2 * 1.2);  % Only warn if > 108 degrees
                
                if significant_divergence && mod(i, 100) == 0  % Only print every 100 steps to avoid spam
                    warning('Numerical instability detected at t=%.2fs. Clipping values.', t(i));
                    fprintf('  nu: [%.2e, %.2e, %.2e, %.2e, %.2e, %.2e]\n', nu);
                    fprintf('  eta(4:6): [%.2e, %.2e, %.2e]\n', eta(4), eta(5), eta(6));
                end
                % Clip values to prevent further divergence
                nu(1:3) = sign(nu(1:3)) .* min(abs(nu(1:3)), max_vel);
                nu(4:6) = sign(nu(4:6)) .* min(abs(nu(4:6)), max_ang_vel);
                eta(4:6) = sign(eta(4:6)) .* min(abs(eta(4:6)), max_angle);
                eta(4:6) = ssa(eta(4:6));  % Wrap again after clipping
                x = [nu; eta];
            end
            
            % Update x with wrapped angles
            x(7:12) = eta;
            
            % Progress indicator (every 10% or every 10000 steps, whichever is smaller)
            progress_interval = min(round(nTimeSteps/10), 10000);
            if mod(i, progress_interval) == 0 || i == nTimeSteps
                elapsed = toc;
                remaining = (nTimeSteps - i) / i * elapsed;
                fprintf('  Progress: %.1f%% (t=%.1fs, Elapsed: %.1fs, Remaining: %.1fs)\n', ...
                    100*i/nTimeSteps, t(i), elapsed, remaining);
            end
        end
        
        elapsed_time = toc;
        fprintf('Simulation completed in %.1f seconds\n', elapsed_time);
        
        % Extract data
        time = simdata(:, 1);
        u = simdata(:, 2);
        v = simdata(:, 3);
        w = simdata(:, 4);
        p = simdata(:, 5);
        q = simdata(:, 6);
        r = simdata(:, 7);
        phi = simdata(:, 8);
        theta = simdata(:, 9);
        
        % Final check for outliers and remove them
        fprintf('Checking for outliers...\n');
        reasonable_ranges = struct();
        reasonable_ranges.u = [-10, 10];
        reasonable_ranges.v = [-10, 10];
        reasonable_ranges.w = [-10, 10];
        reasonable_ranges.p = [-10, 10];
        reasonable_ranges.q = [-10, 10];
        reasonable_ranges.r = [-10, 10];
        reasonable_ranges.phi = [-pi/2, pi/2];
        reasonable_ranges.theta = [-pi/2, pi/2];
        
        valid_idx = true(size(time));
        if any(abs(u) > reasonable_ranges.u(2)) || any(abs(u) < -abs(reasonable_ranges.u(1)))
            invalid = abs(u) > reasonable_ranges.u(2);
            fprintf('  Removing %d outliers from u\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(v) > reasonable_ranges.v(2)) || any(abs(v) < -abs(reasonable_ranges.v(1)))
            invalid = abs(v) > reasonable_ranges.v(2);
            fprintf('  Removing %d outliers from v\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(w) > reasonable_ranges.w(2)) || any(abs(w) < -abs(reasonable_ranges.w(1)))
            invalid = abs(w) > reasonable_ranges.w(2);
            fprintf('  Removing %d outliers from w\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(p) > reasonable_ranges.p(2)) || any(abs(p) < -abs(reasonable_ranges.p(1)))
            invalid = abs(p) > reasonable_ranges.p(2);
            fprintf('  Removing %d outliers from p\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(q) > reasonable_ranges.q(2)) || any(abs(q) < -abs(reasonable_ranges.q(1)))
            invalid = abs(q) > reasonable_ranges.q(2);
            fprintf('  Removing %d outliers from q\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(r) > reasonable_ranges.r(2)) || any(abs(r) < -abs(reasonable_ranges.r(1)))
            invalid = abs(r) > reasonable_ranges.r(2);
            fprintf('  Removing %d outliers from r\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(phi) > reasonable_ranges.phi(2)) || any(abs(phi) < -abs(reasonable_ranges.phi(1)))
            invalid = abs(phi) > reasonable_ranges.phi(2);
            fprintf('  Removing %d outliers from phi\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        if any(abs(theta) > reasonable_ranges.theta(2)) || any(abs(theta) < -abs(reasonable_ranges.theta(1)))
            invalid = abs(theta) > reasonable_ranges.theta(2);
            fprintf('  Removing %d outliers from theta\n', sum(invalid));
            valid_idx = valid_idx & ~invalid;
        end
        
        % Apply filtering
        time = time(valid_idx);
        u = u(valid_idx);
        v = v(valid_idx);
        w = w(valid_idx);
        p = p(valid_idx);
        q = q(valid_idx);
        r = r(valid_idx);
        phi = phi(valid_idx);
        theta = theta(valid_idx);
        
        fprintf('  After outlier removal: %d valid samples (removed %d)\n', ...
            length(time), sum(~valid_idx));
        
        % Ensure all vectors are column vectors and have same length
        time = time(:);
        u = u(:);
        v = v(:);
        w = w(:);
        p = p(:);
        q = q(:);
        r = r(:);
        phi = phi(:);
        theta = theta(:);
        
        % Verify all have same length
        n_samples = length(time);
        if ~(length(u) == n_samples && length(v) == n_samples && length(w) == n_samples && ...
             length(p) == n_samples && length(q) == n_samples && length(r) == n_samples && ...
             length(phi) == n_samples && length(theta) == n_samples)
            error('Vector length mismatch after outlier removal. time=%d, u=%d, v=%d, w=%d, p=%d, q=%d, r=%d, phi=%d, theta=%d', ...
                n_samples, length(u), length(v), length(w), length(p), length(q), length(r), length(phi), length(theta));
        end
        
        % Save to CSV file (include Hs in filename, NO Xacc)
        csv_filename = sprintf('6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f.csv', ...
            tonnage, L, B, T, Hs);
        csv_filepath = fullfile(output_dir, csv_filename);
        
        fprintf('Writing CSV file: %s\n', csv_filepath);
        
        % Write CSV file with header (NO Xacc column)
        fid = fopen(csv_filepath, 'w');
        if fid == -1
            error('Cannot open file for writing: %s', csv_filepath);
        end
        
        % Write header (6-DOF only, no Xacc)
        fprintf(fid, 'time,u,v,w,p,q,r,phi,theta\n');
        
        % Write data (write in chunks to avoid memory issues)
        chunk_size = 100000;
        n_samples = length(time);
        
        for chunk_start = 1:chunk_size:n_samples
            chunk_end = min(chunk_start + chunk_size - 1, n_samples);
            chunk_idx = chunk_start:chunk_end;
            
            for i = chunk_idx
                fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                    time(i), u(i), v(i), w(i), p(i), q(i), r(i), phi(i), theta(i));
            end
            
            if mod(chunk_start, 500000) == 1 && chunk_start > 1
                fprintf('  Progress: %.1f%%\n', 100*chunk_start/n_samples);
            end
        end
        
        fclose(fid);
        
        fprintf('CSV file saved: %s\n', csv_filepath);
        file_info = dir(csv_filepath);
        fprintf('File size: %.2f MB\n', file_info.bytes / 1024 / 1024);
        fprintf('Total rows: %d\n\n', n_samples);
        
    end  % end for tonnages
    
end  % end for Hs_values

fprintf('========================================\n');
fprintf('6-DOF Dataset generation completed!\n');
fprintf('========================================\n');
fprintf('Generated CSV files:\n');
file_count = 0;
for hs_idx = 1:length(Hs_values)
    Hs = Hs_values(hs_idx);
    fprintf('\nHs = %.1f m:\n', Hs);
    for t_idx = 1:length(tonnages)
        tonnage = tonnages(t_idx);
        % Use actual dimensions
        if tonnage == 10
            L = 7.93; B = 1.90; T = 0.95;
        elseif tonnage == 20
            L = 9.99; B = 2.40; T = 1.20;
        elseif tonnage == 30
            L = 11.43; B = 2.74; T = 1.37;
        elseif tonnage == 40
            L = 12.58; B = 3.02; T = 1.51;
        elseif tonnage == 50
            L = 13.56; B = 3.25; T = 1.63;
        else
            L = 3.5 * (tonnage^0.4);
            B = 0.24 * L;
            T = 0.12 * L;
        end
        csv_filename = sprintf('6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f.csv', ...
            tonnage, L, B, T, Hs);
        csv_filepath = fullfile(output_dir, csv_filename);
        if exist(csv_filepath, 'file')
            fprintf('  - %s\n', csv_filepath);
            file_count = file_count + 1;
        end
    end
end
fprintf('\nTotal files generated: %d\n', file_count);
fprintf('\n');

