% regen_6dof_parity.m
% Thin driver: sets the parity switches then runs regen_6dof to emit the SINGLE
% file (10 t, Hs 5, r0) used by tests/test_sim_parity.py to validate that the
% MATLAB regeneration wrapper reproduces the published record.
%
% Run from the fishingVessel dir (MATLAB R2024b):
%   cd "00_Ref/논문코드_데이터셋/fishingVessel" && \
%     matlab.exe -batch "addpath(genpath('../../MSS_library')); regen_6dof_parity" 2>&1 | tr -d '\r'

PARITY_ONLY = true;
PARITY_TON  = 10;
PARITY_HS   = 5;
PARITY_R    = 0;
regen_6dof;
