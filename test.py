import scipy.io
import subprocess
import os
import numpy as np
from model import *
import argparse

parser = argparse.ArgumentParser(description="parser")
parser.add_argument("--mode", type=str, help="Mode: Constant / RL / P")
parser.add_argument("--lp_const", type=float, default=400.0, help="Constant LP value if mode is Constant")
parser.add_argument("--K", type=float, default=50, help="Proportional gain for P controller")
args = parser.parse_args()
# -----------------------------
# 1. Prepare MATLAB input paramsStruct
# -----------------------------
params_dict = {
    'k': 0.0067,
    'rho': 0.00433,
    'specificHeat': 0.526,
    'ic': 300.0,
    'thick': 2.0,
    'width': 12.0,
    'height': 3.0,
    'hmax': 0.4,
    'squareSideFraction': 0.4,
    'scan_pattern': np.linspace(0.0, 12.0, 48),
    'style': 'simultaneous',
    'params': {
        'SS': 600.0,
        'LP': 100.0,
        'eeta': 0.3,
        'r_b': 0.06,
        'H': 0.1
    },
    'heatTime': 0.05,
    'coolTime': 0.10,
    'nTimeStepsHeat': 50.0,
    'nTimeStepsCool': 50.0,
    'doPlot': False,
    'tempRange': np.array([2500.0, 3000.0])
}

# Load RL Agent model
MODE = args.mode

if MODE == 'Constant':
    LP_CONST = args.lp_const
    print(f'Testing constant LP: {LP_CONST}')
elif MODE == 'RL':
    n = 4000
    step_size = 50
    agent = torch.load(f"checkpoints/qnet_offline_{n}_{step_size}.pt")
    print("Pretrained agent loaded.")
elif MODE == 'P':
    print("Testing under Proportional control mode.")

nSteps = 8
# Define layer evolution
initialFraction = 0.4
finalFraction = 0.5
fractions = np.linspace(initialFraction, finalFraction, nSteps)


states = [torch.ones(1053, dtype=torch.float32).to(device) * params_dict['ic']]
for i in range(nSteps):

    params_dict['squareSideFraction'] = fractions[i]
    # Select action using RL agent
    if MODE == 'Constant':
        params_dict['params']['LP'] = LP_CONST
    elif MODE == 'RL':  
        if i == 0:
            params_dict['params']['LP'] = 300.0  # Initial action
        else:
            params_dict['params']['LP'] = select_action(agent, states[i])
    elif MODE == 'P':
        if i == 0:
            params_dict['params']['LP'] = 300.0
        else:
            # Proportional control update
            K = args.K  # proportional gain (tune this)
            delta = K * r_prev  # negative because we want to reduce the loss toward zero
            a_new = a_prev + delta

            # Clip and round to nearest 50 within 100-600
            a_new = float(max(100, min(600, round(a_new / 50) * 50)))
            params_dict['params']['LP'] = a_new

    print(f"Action chosen at step {i}: LP = {params_dict['params']['LP']}")
    # Save as MATLAB .mat file
    scipy.io.savemat('../LPBF-Simulation/test/params.mat', {'paramsStruct': params_dict})

    # -----------------------------
    # 2. Create MATLAB wrapper script
    # -----------------------------
    result_path = "../Offline-RL-Controller-in-AM/checkpoints/results/"
    matlab_folder = '../../LPBF-Simulation/'
    matlab_script = f"""
                    cd('{matlab_folder}');
                    paramsStruct = load('test/params.mat').paramsStruct;
                    [uFinal, tAll, uAll, resultAll, model, meanDeviation] = simulateHeatingCooling(paramsStruct);
                    save('test/results.mat','uFinal','tAll','uAll','meanDeviation');
                    i={i};
                    fig = figure('Visible','off');
                    pdeplot(model,'XYData',uFinal,'Mesh','on','ColorMap','jet');
                    colorbar; caxis([300 5000]);
                    title(sprintf('Step %d: Cooling Final Temperature',i));
                    saveas(fig, fullfile("{result_path}",sprintf('layer_%d_finalTemp.png',i)));
                    close(fig);
                    exit
    """
    with open('../LPBF-Simulation/test/runSim.m', 'w') as f:
        f.write(matlab_script)

    # -----------------------------
    # 3. Call MATLAB via subprocess
    # -----------------------------
    script_path = os.path.abspath("../LPBF-Simulation/test/runSim.m")

    subprocess.run([
        "matlab",
        "-nodisplay",
        "-nosplash",
        "-nodesktop",
        "-r", f"run('{script_path}'); exit;"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # -----------------------------
    # 4. Load MATLAB output
    # -----------------------------
    res = scipy.io.loadmat('../LPBF-Simulation/test/results.mat')
    # print(res.keys())
    uFinal = res['uFinal']
    tAll = res['tAll']
    uAll = res['uAll']
    meanDeviation = res['meanDeviation']
    states.append(torch.tensor(uFinal.flatten(), dtype=torch.float32).to(device))

    print("Reward:", -meanDeviation[0][0])
    r_prev = -meanDeviation[0][0] # Used for P controller
    a_prev = params_dict['params']['LP']  # Used for P controller
    # print("uFinal shape:", uFinal.shape)
    # print("tAll shape:", tAll.shape)
