% regen_6dof.m
% MATLAB/Octave wrapper around the ORIGINAL generate6Dof_dataset.m loop, extended to
% emit MULTIPLE independent wave-phase realizations per (tonnage, Hs) condition.
% Run via MATLAB R2024b (matlab.exe) -- the SAME engine that produced the published
% data -- but kept engine-neutral so it runs unchanged under Octave too.
%
% This script does NOT modify generate6Dof_dataset.m. It re-implements the SAME
% simulation loop (Modified PM spectrum, Froude-Krylov wave forcing, RK4 @fishingVessel,
% ssa angle wrapping, divergence clipping, outlier removal, %.6f CSV writing) with the
% PUBLISHED configuration locked in:
%
%   Hs_values = [3.0 5.0 7.0]   (published; original committed .m had only [5.0])
%   T_final   = 5000            (published; original committed .m had 20000)
%                               -> 5001 samples (t = 0,1,...,5000)
%   T0 = 8.0 s, beta = deg2rad(140), zero forward speed,
%   100 freq components, maxFreq = 3.0 rad/s, no spreading, h = 1 s.
%
% Realizations: loop  for r_idx = 0:R  (R = 5 -> 6 realizations incl. the published r0).
% Seed scheme (preserves published record as r_idx = 0):
%   r_idx == 0 :  rng(12345 + hs_idx*100 + t_idx)              % ORIGINAL seed
%   r_idx >= 1 :  rng(12345 + hs_idx*100 + t_idx + 1000*r_idx) % independent phase sets
%
% Output filenames (regex-compatible with the Python pipeline, plus _r%d suffix):
%   6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f_r%d.csv     (cols: time,u,v,w,p,q,r,phi,theta)
% written to  6Dof_dataset_multi/  (or 6Dof_dataset_parity/ when PARITY_ONLY=true).
%
% This script adds MSS_library to the path internally (see below), so it is robust
% regardless of how it is launched. Run from the fishingVessel dir, e.g.:
%   cd "00_Ref/논문코드_데이터셋/fishingVessel" && \
%     matlab.exe -batch "addpath(genpath('../../MSS_library')); regen_6dof" 2>&1 | tr -d '\r'
% Parity-only single file (10t, Hs5, r0) for the parity gate -- use the driver:
%   cd "00_Ref/논문코드_데이터셋/fishingVessel" && \
%     matlab.exe -batch "addpath(genpath('../../MSS_library')); regen_6dof_parity" 2>&1 | tr -d '\r'

% ------------------------------------------------------------------------------------
% Switches (may be pre-set by a caller, e.g. via matlab.exe -batch "...; regen_6dof").
% Guard each with exist() so this script also runs standalone with defaults.
% ------------------------------------------------------------------------------------
if ~exist('PARITY_ONLY', 'var'); PARITY_ONLY = false; end  % true -> emit ONE file only
if ~exist('PARITY_TON',  'var'); PARITY_TON  = 10;    end  % tonnage for the parity file
if ~exist('PARITY_HS',   'var'); PARITY_HS   = 5;     end  % Hs (m) for the parity file
if ~exist('PARITY_R',    'var'); PARITY_R    = 0;     end  % realization for the parity file
if ~exist('R',           'var'); R           = 5;     end  % extra realizations (total = R+1)
% R3.4 GM-floor study: when true, integrate with @fishingVessel_unclamped (transverse
% GM_T floor removed inside the EOM) and write to 6Dof_dataset_multi_unclampedEOM/.
% Everything else -- config, seeds, RNG carry-over, wave phases -- is IDENTICAL to the
% clamped run, so the ONLY difference is the 10 t restoring dynamics (clean isolation).
if ~exist('USE_UNCLAMPED_EOM', 'var'); USE_UNCLAMPED_EOM = false; end

% NOTE: do NOT use `clearvars` here -- it would wipe the switches above. The original
% generate6Dof_dataset.m calls clearvars at the top; we intentionally diverge so the
% caller-supplied switches survive. `clear fishingVessel`/`clear waveForceRAO` are still
% issued per-combination below to reset the persistent vessel_params (as in the original).
close all;

% ------------------------------------------------------------------------------------
% Path setup: add MSS_library (and this folder) so fishingVessel, rk4, ssa,
% waveDirectionalSpectrum, Gmtrx, rbody, m2c, invQR, Dmtrx, eulerang,
% forceSurgeDamping, crossFlowDrag all resolve. Works regardless of cwd, by anchoring
% relative to THIS script's location.
% ------------------------------------------------------------------------------------
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir); this_dir = pwd; end
mss_dir = fullfile(this_dir, '..', '..', 'MSS_library');
addpath(genpath(mss_dir));
addpath(this_dir);

fprintf('========================================\n');
fprintf('6-DOF Multi-Realization Generator (MATLAB/Octave wrapper)\n');
fprintf('========================================\n\n');

% ------------------------------------------------------------------------------------
% EOM selection: clamped (default) vs unclamped-EOM (R3.4 GM-floor study).
% The function NAME must match its filename for @handle dispatch:
%   fishingVessel.m            -> @fishingVessel            (GM_T floor active)
%   fishingVessel_unclamped.m  -> @fishingVessel_unclamped  (GM_T floor removed)
% ------------------------------------------------------------------------------------
if USE_UNCLAMPED_EOM
    eom_handle = @fishingVessel_unclamped;
    eom_name   = 'fishingVessel_unclamped';
else
    eom_handle = @fishingVessel;
    eom_name   = 'fishingVessel';
end

% ------------------------------------------------------------------------------------
% Output directory (parity vs full multi; clamped vs unclamped-EOM)
% ------------------------------------------------------------------------------------
if PARITY_ONLY
    output_dir = fullfile(this_dir, '6Dof_dataset_parity');
elseif USE_UNCLAMPED_EOM
    output_dir = fullfile(this_dir, '6Dof_dataset_multi_unclampedEOM');
else
    output_dir = fullfile(this_dir, '6Dof_dataset_multi');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end
fprintf('Output directory: %s\n', output_dir);
fprintf('EOM: @%s\n\n', eom_name);

% ------------------------------------------------------------------------------------
% Configuration  (PUBLISHED values -- see header)
% ------------------------------------------------------------------------------------
tonnages   = [10, 20, 30, 40, 50];   % 10-ton steps, 5 hulls
Hs_values  = [3.0, 5.0, 7.0];        % significant wave heights (m) -- PUBLISHED
T0         = 8.0;                     % modal period (s)
beta_wave  = deg2rad(140);           % wave direction (rad)
h          = 1;                      % sampling time [s] (1 Hz)
T_final    = 5000;                   % final sim time [s] -> 5001 samples -- PUBLISHED

fprintf('Configuration:\n');
fprintf('  Tonnages: %s\n', mat2str(tonnages));
fprintf('  Wave heights (Hs): %s m\n', mat2str(Hs_values));
fprintf('  Wave: T0=%.2fs, Direction=%.1f deg\n', T0, rad2deg(beta_wave));
fprintf('  Sampling: %.4fs (%.1f Hz), Duration: %.1fs\n', h, 1/h, T_final);
fprintf('  Realizations per condition: %d (r_idx = 0..%d)\n', R+1, R);
if PARITY_ONLY
    fprintf('  PARITY_ONLY: ton=%d, Hs=%.1f, r=%d\n', PARITY_TON, PARITY_HS, PARITY_R);
end
fprintf('\n');

% ------------------------------------------------------------------------------------
% Main loops: Hs x tonnage x realization
% ------------------------------------------------------------------------------------
for hs_idx = 1:length(Hs_values)
    Hs = Hs_values(hs_idx);

    for t_idx = 1:length(tonnages)
        tonnage = tonnages(t_idx);

        % --- Ship dimensions (identical to original) -----------------------------
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

        % ----------------------------------------------------------------------------
        % CRITICAL RNG ORDERING (must match the ORIGINAL sweep to reproduce r0).
        %
        % waveDirectionalSpectrum() consumes rand() INTERNALLY to randomize Omega within
        % each frequency interval, and the ORIGINAL generate6Dof_dataset.m does NOT seed
        % before that call. So the published Omega for each (hs,ton) depends on the RNG
        % state carried over from all PRECEDING combinations in the sweep. To reproduce
        % the published record exactly, we must:
        %   (1) call the spectrum ONCE per (hs,ton) with the natural carry-over state
        %       (exactly as the original -- NOT once per realization, which would corrupt
        %        the carry-over chain);
        %   (2) rng(seed_r0) then draw the r0 phases (this is the original consumption
        %       that the NEXT combination's spectrum call inherits);
        %   (3) capture that post-r0-phase state as the carry-over;
        %   (4) draw the extra (r>=1) realization phases from independent seeds;
        %   (5) RESTORE the carry-over state so the next (hs,ton) spectrum matches the
        %       original sweep.
        % Omega/Amp are shared across realizations of a condition (correct: realizations
        % differ only in wave PHASE, not in the frequency discretization).
        % ----------------------------------------------------------------------------
        maxFreq          = 3.0;
        numFreqIntervals = 100;
        spreadingFlag    = false;
        numDirections    = 1;
        w0 = 2*pi/T0;   % peak frequency (rad/s)

        % Reset persistent state in the active EOM/waveForceRAO (as the original does,
        % once per combination -- has no RNG effect). clear(eom_name) resets the
        % persistent vessel_params of whichever EOM (clamped/unclamped) is in use.
        clear(eom_name);
        clear waveForceRAO;

        % (1) Spectrum with natural carry-over state.
        [S_M, Omega, Amp, ~, ~, mu] = waveDirectionalSpectrum('Modified PM', ...
            [Hs, w0], numFreqIntervals, maxFreq, spreadingFlag, numDirections);

        % (2) r0 phases from the ORIGINAL published seed.
        rng(12345 + hs_idx*100 + t_idx);
        if size(S_M, 2) == 1
            phases_all = cell(R+1, 1);
            phases_all{1} = 2*pi*rand(numFreqIntervals, 1);          % r_idx = 0
        else
            phases_all = cell(R+1, 1);
            phases_all{1} = 2*pi*rand(numFreqIntervals, numDirections);
        end

        % (3) Capture carry-over state (state AFTER r0 phases) -- this is exactly what
        %     the original sweep carries into the next combination's spectrum call.
        carry_state = rng;

        % (4) Independent phases for r_idx = 1..R from distinct seeds.
        for rr = 1:R
            rng(12345 + hs_idx*100 + t_idx + 1000*rr);
            if size(S_M, 2) == 1
                phases_all{rr+1} = 2*pi*rand(numFreqIntervals, 1);
            else
                phases_all{rr+1} = 2*pi*rand(numFreqIntervals, numDirections);
            end
        end

        % (5) Restore carry-over so the NEXT combination's spectrum matches the original.
        rng(carry_state);

        % Time vector (shared across realizations).
        t = 0:h:T_final;
        nTimeSteps = length(t);

        for r_idx = 0:R

            % --- Parity gate: skip only INTEGRATION+WRITE for non-target files. -----
            % NOTE: we do NOT skip the RNG replay above -- that already happened once per
            % combination and is required to reproduce the target's carry-over Omega.
            if PARITY_ONLY
                if ~(tonnage == PARITY_TON && abs(Hs - PARITY_HS) < 1e-9 && r_idx == PARITY_R)
                    continue;
                end
            end

            fprintf('----------------------------------------\n');
            fprintf('Hs=%.1fm, %dton, r=%d  (L=%.2f B=%.2f T=%.2f)\n', ...
                Hs, tonnage, r_idx, L, B, T);
            fprintf('----------------------------------------\n');

            % Per-realization EOM persistent reset (defensive; no RNG effect).
            clear(eom_name);

            % Select this realization's pre-computed phases (Omega/Amp shared).
            randomPhases = phases_all{r_idx + 1};

            % --- Initialize ------------------------------------------------------
            x = zeros(12,1);
            simdata = zeros(nTimeSteps, 9);          % [t,u,v,w,p,q,r,phi,theta]
            wave_elevation = zeros(nTimeSteps, 1);

            % --- Extract initial states ------------------------------------------
            nu = x(1:6);
            eta = x(7:12);

            % --- Simulation loop (identical to original) -------------------------
            for i = 1:nTimeSteps

                % Wave elevation
                if size(S_M, 2) == 1
                    wave_elevation(i) = sum(Amp .* cos(Omega * t(i) + randomPhases));
                else
                    wave_elevation(i) = sum(sum(Amp .* cos(Omega * t(i) + randomPhases)));
                end

                % Froude-Krylov approximation (identical constants)
                rho = 1025;
                g = 9.81;
                nabla = 0.68 * L * B * T;   % Cb = 0.68
                Awp = 0.80 * B * L;         % Cw = 0.80

                w0_wave = 2*pi/T0;
                k = w0_wave^2 / g;          % deep-water wave number

                scale_surge = 0.005;
                scale_sway  = 0.005;
                scale_heave = 0.05;
                scale_roll  = 0.02;
                scale_pitch = 0.02;

                F_wave_surge = scale_surge * rho * g * nabla * k * wave_elevation(i) * cos(beta_wave);
                F_wave_sway  = scale_sway  * rho * g * nabla * k * wave_elevation(i) * sin(beta_wave);
                F_wave_heave = scale_heave * rho * g * Awp   *     wave_elevation(i);
                M_wave_roll  = scale_roll  * rho * g * nabla * B * wave_elevation(i) * sin(beta_wave);
                M_wave_pitch = scale_pitch * rho * g * nabla * L * wave_elevation(i) * cos(beta_wave);
                M_wave_yaw   = 0;

                tau_wave = [F_wave_surge; F_wave_sway; F_wave_heave; ...
                            M_wave_roll; M_wave_pitch; M_wave_yaw];

                % Store BEFORE integration (matches original: row i = state at t(i))
                simdata(i, :) = [t(i), nu(1), nu(2), nu(3), nu(4), nu(5), nu(6), ...
                    eta(4), eta(5)];

                % RK4 integration (eom_handle = @fishingVessel or @fishingVessel_unclamped)
                x = rk4(eom_handle, h, x, tau_wave, L, B, T, tonnage);

                nu = x(1:6);
                eta = x(7:12);

                % Wrap Euler angles to [-pi, pi)
                eta(4) = ssa(eta(4));
                eta(5) = ssa(eta(5));
                eta(6) = ssa(eta(6));

                % Divergence checks / clipping (identical thresholds)
                max_vel = 20;
                max_ang_vel = 20;
                max_angle = pi/2 * 1.1;

                vel_exceeded     = any(abs(nu(1:3)) > max_vel);
                ang_vel_exceeded = any(abs(nu(4:6)) > max_ang_vel);
                angle_exceeded   = any(abs(eta(4:6)) > max_angle);

                if vel_exceeded || ang_vel_exceeded || angle_exceeded
                    nu(1:3) = sign(nu(1:3)) .* min(abs(nu(1:3)), max_vel);
                    nu(4:6) = sign(nu(4:6)) .* min(abs(nu(4:6)), max_ang_vel);
                    eta(4:6) = sign(eta(4:6)) .* min(abs(eta(4:6)), max_angle);
                    eta(4:6) = ssa(eta(4:6));
                    x = [nu; eta];
                end

                x(7:12) = eta;
            end

            % --- Extract columns -------------------------------------------------
            time  = simdata(:, 1);
            u     = simdata(:, 2);
            v     = simdata(:, 3);
            w     = simdata(:, 4);
            p     = simdata(:, 5);
            q     = simdata(:, 6);
            r     = simdata(:, 7);
            phi   = simdata(:, 8);
            theta = simdata(:, 9);

            % --- Outlier removal (identical ranges) ------------------------------
            valid_idx = true(size(time));
            valid_idx = valid_idx & ~(abs(u)     > 10);
            valid_idx = valid_idx & ~(abs(v)     > 10);
            valid_idx = valid_idx & ~(abs(w)     > 10);
            valid_idx = valid_idx & ~(abs(p)     > 10);
            valid_idx = valid_idx & ~(abs(q)     > 10);
            valid_idx = valid_idx & ~(abs(r)     > 10);
            valid_idx = valid_idx & ~(abs(phi)   > pi/2);
            valid_idx = valid_idx & ~(abs(theta) > pi/2);

            time = time(valid_idx);  u = u(valid_idx);  v = v(valid_idx);
            w = w(valid_idx);  p = p(valid_idx);  q = q(valid_idx);
            r = r(valid_idx);  phi = phi(valid_idx);  theta = theta(valid_idx);

            n_samples = length(time);
            fprintf('  Valid samples: %d (removed %d)\n', n_samples, sum(~valid_idx));

            % --- Write CSV (motion only; _r%d suffix) ----------------------------
            csv_filename = sprintf('6Dof_%dton_L%.1f_B%.1f_T%.1f_Hs%.1f_r%d.csv', ...
                tonnage, L, B, T, Hs, r_idx);
            csv_filepath = fullfile(output_dir, csv_filename);

            fid = fopen(csv_filepath, 'w');
            if fid == -1
                error('Cannot open file for writing: %s', csv_filepath);
            end
            fprintf(fid, 'time,u,v,w,p,q,r,phi,theta\n');
            for i = 1:n_samples
                fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                    time(i), u(i), v(i), w(i), p(i), q(i), r(i), phi(i), theta(i));
            end
            fclose(fid);

            fprintf('  Wrote: %s (%d rows)\n\n', csv_filepath, n_samples);

        end  % r_idx
    end  % tonnage
end  % Hs

fprintf('========================================\n');
fprintf('Done. Output in: %s\n', output_dir);
fprintf('========================================\n');
