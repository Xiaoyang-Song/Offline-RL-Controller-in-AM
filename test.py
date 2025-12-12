import scipy.io
import subprocess
import os
import numpy as np
from model import *

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
agent = torch.load('checkpoints/qnet_offline_4000.pt')
print("Pretrained agent loaded.")

nSteps = 8
# Define layer evolution
initialFraction = 0.4
finalFraction = 0.5
fractions = np.linspace(initialFraction, finalFraction, nSteps)


states = [torch.ones(1053, dtype=torch.float32).to(device) * params_dict['ic']]
for i in range(nSteps):

    params_dict['squareSideFraction'] = fractions[i]
    # Select action using RL agent
    params_dict['params']['LP'] = select_action(agent, states[i])
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
    # print("uFinal shape:", uFinal.shape)
    # print("tAll shape:", tAll.shape)
