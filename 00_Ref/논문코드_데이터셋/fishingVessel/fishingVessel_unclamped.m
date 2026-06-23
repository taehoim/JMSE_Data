function [xdot, U, M] = fishingVessel_unclamped(x, tau_wave, L, B, T, tonnage)
% UNCLAMPED-EOM variant of fishingVessel.m for the GM-floor study (R3.4).
% IDENTICAL to fishingVessel.m EXCEPT the transverse-metacentric-height floor
% (`if GM_T < 0.3, GM_T = 0.3`) is REMOVED, so the restoring matrix Gmtrx uses the
% TRUE computed GM_T. This matters only for the 10 t hull (true GM_T = 0.2582 m < 0.30),
% whose published roll MOTION was simulated at the clamped GM_T = 0.30. The GM_L<0.5
% floor is KEPT verbatim (it is inactive for all tonnages: GM_L >= 5.63 m), so this
% isolates the TRANSVERSE floor only. 20-50 t are unaffected (true GM_T >= 0.326 m).
% Do NOT mutate the original fishingVessel.m.
%
% Compatible with MATLAB and GNU Octave (www.octave.org)
% [xdot, U, M] = fishingVessel_unclamped(x, tau_wave, L, B, T, tonnage) returns the
% time derivative xdot of the state vector for a fishing vessel. The 6x6 mass
% matrix M is an optional output.
%
% Inputs:
%   x: state vector x = [ u v w p q r x y z phi theta psi ]'
%   tau_wave: 6x1 wave force vector (optional, default zeros)
%   L: Length (m)
%   B: Beam (m)
%   T: Draft (m)
%   tonnage: Ship tonnage (ton)
%
% Outputs:
%   xdot: time derivative of state vector
%   U: speed (m/s)
%   M: 6x6 mass matrix (optional)
%
% The 6-DOF equations of motion are based on the nonlinear model:
%   eta_dot = J(eta) * nu
%   nu_dot = Minv * ( tau_wave + tau_drag + tau_crossflow - (CRB + CA + D) * nu - G * eta )
%
% Author: Based on MSS toolbox models
% Date: 2025

persistent vessel_params;

% Handle case when called without arguments (display vessel info)
if nargin == 0
    x = zeros(12,1);
    tau_wave = zeros(6,1);
    L = 20;
    B = 4;
    T = 2;
    tonnage = 20;
    vessel_params = [];  % Force recomputation
end

% Initialize vessel parameters (computed once or when parameters change)
if isempty(vessel_params) || (nargin > 0 && nargin < 6)
    
    if nargin > 0 && nargin < 6
        error('fishingVessel requires 6 arguments: x, tau_wave, L, B, T, tonnage');
    end
    
    % Check if parameters have changed (only if vessel_params exists)
    if ~isempty(vessel_params) && nargin >= 6
        if vessel_params.L ~= L || vessel_params.B ~= B || ...
           vessel_params.T ~= T || vessel_params.tonnage ~= tonnage
            % Parameters changed, need to recompute
            vessel_params = [];
        end
    end
    
    % Constants
    g = 9.81;
    rho = 1025;  % Density of water (kg/m^3)
    
    % Ship parameters
    vessel_params.L = L;
    vessel_params.B = B;
    vessel_params.T = T;
    vessel_params.tonnage = tonnage;
    vessel_params.m = tonnage * 1000;  % Mass in kg
    vessel_params.rho = rho;
    
    % Block coefficient (small fishing vessel)
    vessel_params.Cb = 0.68;
    
    % Volume displacement
    vessel_params.nabla = vessel_params.Cb * L * B * T;
    
    % Waterplane area coefficient
    vessel_params.Cw = 0.80;
    vessel_params.Awp = vessel_params.Cw * B * L;
    
    % Center of gravity (KG ≈ 0.60·T for small fishing vessels with superstructure)
    KG = 0.60 * T;  % Vertical position of CG above keel
    vessel_params.r_bg = [-0.05*L, 0, -(KG - T)]';  % CG relative to CO (negative z is upward)
    
    % Center of buoyancy
    vessel_params.KB = (1/3) * (5*T/2 - vessel_params.nabla/vessel_params.Awp);
    vessel_params.r_bb = [-0.05*L, 0, T - vessel_params.KB]';
    vessel_params.BG = vessel_params.r_bb(3) - vessel_params.r_bg(3);
    
    % Moments of inertia of waterplane area
    vessel_params.k_munro_smith = (6 * vessel_params.Cw^3) / ...
        ((1+vessel_params.Cw) * (1+2*vessel_params.Cw));
    vessel_params.I_T = vessel_params.k_munro_smith * (B^3 * L) / 12;
    vessel_params.I_L = 0.7 * (L^3 * B) / 12;
    
    % Metacentric heights
    vessel_params.BM_T = vessel_params.I_T / vessel_params.nabla;
    vessel_params.BM_L = vessel_params.I_L / vessel_params.nabla;
    vessel_params.GM_T = vessel_params.BM_T - vessel_params.BG;
    vessel_params.GM_L = vessel_params.BM_L - vessel_params.BG;
    
    % Ensure positive GM
    % --- UNCLAMPED-EOM (R3.4): transverse GM_T floor REMOVED ---
    % Original fishingVessel.m clamps `if GM_T < 0.3, GM_T = 0.3`; here we keep the
    % TRUE computed GM_T so the 10 t restoring dynamics reflect its real GM (0.2582 m).
    % The longitudinal GM_L floor is preserved verbatim (inactive for all hulls).
    if vessel_params.GM_L < 0.5
        vessel_params.GM_L = 0.5;
    end
    
    % G matrix (restoring forces)
    vessel_params.LCF = -0.05*L;
    vessel_params.r_bp = [0, 0, 0]';
    vessel_params.G = Gmtrx(vessel_params.nabla, vessel_params.Awp, ...
        vessel_params.GM_T, vessel_params.GM_L, vessel_params.LCF, vessel_params.r_bp);
    
    % Rigid-body mass matrix MRB
    vessel_params.R44 = 0.35 * B;  % Radius of gyration in roll
    vessel_params.R55 = 0.25 * L;  % Radius of gyration in pitch
    vessel_params.R66 = 0.25 * L;  % Radius of gyration in yaw
    vessel_params.MRB = rbody(vessel_params.m, vessel_params.R44, ...
        vessel_params.R55, vessel_params.R66, [0,0,0]', vessel_params.r_bg);
    
    % Added mass matrix (simplified, based on typical values)
    % Using empirical formulas for fishing vessels
    Xudot = -0.05 * vessel_params.m;
    Yvdot = -0.8 * vessel_params.m;
    Zwdot = -0.9 * vessel_params.m;
    Kpdot = -0.2 * vessel_params.m * vessel_params.R44^2;
    Mqdot = -0.8 * vessel_params.m * vessel_params.R55^2;
    Nrdot = -0.8 * vessel_params.m * vessel_params.R66^2;
    
    vessel_params.MA = -diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot]);
    
    % Mass matrix
    vessel_params.M = vessel_params.MRB + vessel_params.MA;
    vessel_params.Minv = invQR(vessel_params.M);
    
    % Damping matrix
    vessel_params.T1 = 50;   % Time constant in surge (s)
    vessel_params.T2 = 50;   % Time constant in sway (s)
    vessel_params.T6 = 5;    % Time constant in yaw (s)
    vessel_params.zeta4 = 0.25;  % Relative damping ratio in roll (increased for stability)
    vessel_params.zeta5 = 0.4;   % Relative damping ratio in pitch (increased for stability)
    
    vessel_params.D = Dmtrx([vessel_params.T1, vessel_params.T2, vessel_params.T6], ...
        [vessel_params.zeta4, vessel_params.zeta5], vessel_params.MRB, ...
        vessel_params.MA, vessel_params.G);
    
    % Wetted surface area
    vessel_params.S = L * B + 2 * T * B;
    
end

if nargin < 2
    tau_wave = zeros(6,1);
end

% Ensure vessel_params is initialized
if isempty(vessel_params)
    if nargin > 0 && nargin < 6
        error('fishingVessel requires 6 arguments: x, tau_wave, L, B, T, tonnage');
    end
end

nu = x(1:6);      % Generalized velocity vector
eta = x(7:12);    % Generalized position vector

% Output arguments
M = vessel_params.M;
U = sqrt(nu(1)^2 + nu(2)^2);

% Coriolis matrices
[~, CRB] = rbody(vessel_params.m, vessel_params.R44, vessel_params.R55, ...
    vessel_params.R66, nu(4:6), vessel_params.r_bg');
CA = m2c(vessel_params.MA, nu);

% Linear and quadratic drag in surge
flag = 0;
U_max = 10;  % Maximum speed (m/s)
thrust_max = 100e3;  % Maximum thrust (N)
[X, ~, ~] = forceSurgeDamping(flag, nu(1), vessel_params.m, vessel_params.S, ...
    vessel_params.L, vessel_params.T1, vessel_params.rho, U_max, thrust_max);
tau_drag = [X; zeros(5,1)];

% Avoid double counting
vessel_params.D(1,1) = 0;

% Crossflow drag
tau_crossflow = crossFlowDrag(vessel_params.L, vessel_params.B, vessel_params.T, nu);

% Kinematics
J = eulerang(eta(4), eta(5), eta(6));

% Equations of motion
eta_dot = J * nu;
nu_dot = vessel_params.Minv * (tau_wave + tau_drag + tau_crossflow ...
    - (CRB + CA + vessel_params.D) * nu - vessel_params.G * eta);

xdot = [nu_dot; eta_dot];

% Restore D(1,1) for next call
if abs(nu(1)) > 1e-6
    vessel_params.D(1,1) = -X / nu(1);
else
    vessel_params.D(1,1) = vessel_params.M(1,1) / vessel_params.T1;
end

%% Print vessel data
if nargin == 0 && nargout == 0
    
    % Natural frequencies
    w3 = sqrt( vessel_params.G(3,3) / vessel_params.M(3,3) );
    w4 = sqrt( vessel_params.G(4,4) / vessel_params.M(4,4) );
    w5 = sqrt( vessel_params.G(5,5) / vessel_params.M(5,5) );
    T3 = 2 * pi / w3;
    T4 = 2 * pi / w4;    
    T5 = 2 * pi / w5;
    
    % Compute damping coefficients for display
    flag = 1;
    U_max = 10;
    thrust_max = 100e3;
    [~, Xuu, Xu] = forceSurgeDamping(flag, 0, vessel_params.m, vessel_params.S, ...
        vessel_params.L, vessel_params.T1, vessel_params.rho, U_max, thrust_max);
    
    fprintf('\n');
    fprintf('%s\n','-------------------------------------------------------------------------------------');
    fprintf('%s\n','FISHING VESSEL MAIN CHARACTERISTICS');
    fprintf('%s\n','-------------------------------------------------------------------------------------');
    fprintf('%-40s %8.2f ton \n', 'Tonnage:', vessel_params.tonnage);
    fprintf('%-40s %8.2f m \n', 'Length (L):', vessel_params.L);
    fprintf('%-40s %8.2f m \n', 'Beam (B):', vessel_params.B);
    fprintf('%-40s %8.2f m \n', 'Draft (T):', vessel_params.T);
    fprintf('%-40s %8.2f kg \n', 'Mass (m):', vessel_params.m);
    fprintf('%-40s %8.2f kg/m^3 \n', 'Density of water (rho):', vessel_params.rho);
    fprintf('%-40s %8.2f m^3 \n', 'Volume displacement (nabla):', vessel_params.nabla);
    fprintf('%-40s %8.2f \n', 'Block coefficient (C_b):', vessel_params.Cb);
    fprintf('%-40s %8.2f \n', 'Waterplane area coefficient (C_w):', vessel_params.Cw);
    fprintf('%-40s [%2.1f %2.1f %2.1f] m \n', 'Center of gravity (r_bg):',...
        vessel_params.r_bg(1), vessel_params.r_bg(2), vessel_params.r_bg(3));
    fprintf('%-40s %8.2f \n', 'Relative damping ratio in roll:', vessel_params.zeta4);
    fprintf('%-40s %8.2f \n', 'Relative damping ratio in pitch:', vessel_params.zeta5);
    fprintf('%-40s %8.2f m \n', 'Transverse metacentric height (GM_T):', vessel_params.GM_T);
    fprintf('%-40s %8.2f m \n', 'Longitudinal metacentric height (GM_L):', vessel_params.GM_L);
    fprintf('%-40s %8.2f s \n', 'Natural period in heave (T3):', T3);
    fprintf('%-40s %8.2f s \n', 'Natural period in roll (T4):', T4);
    fprintf('%-40s %8.2f s \n', 'Natural period in pitch (T5):', T5);   
    fprintf('%-40s %8.2e \n', 'Linear surge damping coefficient (Xu):', Xu); 
    fprintf('%-40s %8.2e \n', 'Quadratic drag coefficient (X|u|u):', Xuu); 
   
    vessel_params.D(1,1) = -Xu;
    matrices = {'Mass matrix: M = MRB + MA', vessel_params.M;...
        'Linear damping matrix: D', vessel_params.D; 'Restoring matrix: G', vessel_params.G};
    
    for k = 1:size(matrices, 1)
        fprintf('%s\n','-------------------------------------------------------------------------------------');
        fprintf('%-40s\n', matrices{k, 1});
        for i = 1:size(matrices{k, 2}, 1)
            for j = 1:size(matrices{k, 2}, 2)
                if abs(matrices{k, 2}(i,j)) < 1e-10
                    fprintf('         0 ');
                else
                    fprintf('%10.2e ', matrices{k, 2}(i,j));
                end
            end
            fprintf('\n');
        end
    end
    
end

end

