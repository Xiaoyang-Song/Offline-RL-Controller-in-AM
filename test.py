import scipy.io
import subprocess
import os
import sys
import numpy as np
from model import *
import argparse

# ── make sure the repo root is on the path so sub-packages are importable ─────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def to_matlab_numeric(obj):
    if isinstance(obj, dict):
        return {k: to_matlab_numeric(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.bool_:
            return obj
        return obj.astype(np.float64)
    if isinstance(obj, (np.integer, int, np.floating, float)):
        return np.float64(obj)
    return obj

parser = argparse.ArgumentParser(description="parser")
parser.add_argument("--mode", type=str,
                    help="Mode: Constant / RL / OnlineRL / P / Random")
parser.add_argument("--lp_const", type=float, default=400.0,
                    help="Constant LP value if mode is Constant")
parser.add_argument("--K", type=float, default=50,
                    help="Proportional gain for P controller")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/qnet_offline_5000_10.pt",
    help="Checkpoint to load when mode is RL or OnlineRL",
)
parser.add_argument(
    "--surrogate",
    type=str,
    default="surrogate_model_latent/runs/20260601_162903/latent_best.pt",
    help="Latent surrogate checkpoint path — provides normalisation stats for OnlineRL mode.",
)
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
    # 'coolTime': 0.10,
    'coolTime': 0.10,
    'nTimeStepsHeat': 50.0,
    'nTimeStepsCool': 50.0,
    'doPlot': False,
    'tempRange': np.array([2000.0, 2800.0], dtype=np.float64)
}

# Load RL Agent model
MODE = args.mode

if MODE == 'Constant':
    LP_CONST = args.lp_const
    print(f'Testing constant LP: {LP_CONST}')
elif MODE == 'RL':
    agent = load_policy(args.checkpoint)
    print("Pretrained agent loaded successfully...")
elif MODE == 'OnlineRL':
    # ── Load the online Double-DQN agent ────────────────────────────────────
    from online_RL.agent import DQNAgent
    from online_RL.env   import ACTION_LIST as ONLINE_ACTION_LIST
    from surrogate_model_latent.train import load_latent_surrogate

    # The online RL Q-net was trained on states normalised with the latent
    # surrogate's stats.  Load them from the latent surrogate checkpoint
    # (same stats used during online RL training via LPBFEnv).
    # The surrogate model itself is not used here — MATLAB runs the real sim.
    _, _sm, _ss, _, _, _ = load_latent_surrogate(args.surrogate, device='cpu')
    _sm = _sm.to(device)
    _ss = _ss.to(device)

    online_agent = DQNAgent.load(args.checkpoint, device=device)
    online_agent.q_net.eval()

    _N_LAYERS_ONLINE = 12   # must match n_layers used during training

    def _select_action_online(state_raw_tensor: torch.Tensor, layer_idx: int) -> float:
        """
        state_raw_tensor : (D,) raw Kelvin temperature field (torch tensor)
        layer_idx        : 0-based layer number (0 = first layer)
        Returns laser power [W] chosen greedily by the online DQN policy.

        The observation matches LPBFEnv._make_obs used during training:
          [z-score-normalised temp field (D,) ‖ layer_token (1,)]
        where layer_token = layer_idx / (n_layers - 1) ∈ [0, 1].
        """
        state_norm  = ((state_raw_tensor.to(device) - _sm) / _ss).cpu().numpy()
        layer_token = np.array(
            [layer_idx / max(_N_LAYERS_ONLINE - 1, 1)], dtype=np.float32
        )
        obs = np.concatenate([state_norm, layer_token])   # (D+1,)
        action_idx = online_agent.select_action_greedy(obs)
        return float(ONLINE_ACTION_LIST[action_idx].item())

    print(f"Online DQN agent loaded: {online_agent.q_net}")
elif MODE == 'P':
    print("Testing under Proportional control mode.")
elif MODE == 'Random':
    print("Testing under Random action mode.")
else:
    raise ValueError("Invalid mode. Choose from Constant, RL, OnlineRL, P, or Random.")

nSteps = 12
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
        params_dict['params']['LP'] = select_action(agent, states[i])
    elif MODE == 'OnlineRL':
        params_dict['params']['LP'] = _select_action_online(states[i], i)
    elif MODE == 'Random':
        params_dict['params']['LP'] = float(np.random.choice(np.arange(150, 310, 10)))
    elif MODE == 'P':
        if i == 0:
            params_dict['params']['LP'] = 220
        else:
            # Proportional control update
            K = args.K  # proportional gain (tune this)
            delta = - K * r_prev  # negative because we want to reduce the loss toward zero
            a_new = a_prev + delta

            # Clip and round to nearest 10 within 150-300
            a_new = float(max(150, min(300, round(a_new / 10) * 10)))
            params_dict['params']['LP'] = a_new

    print(f"Action chosen at step {i+1}: LP = {params_dict['params']['LP']}")
    # Save as MATLAB .mat file
    matlab_params = to_matlab_numeric(params_dict)
    scipy.io.savemat('../LPBF-Simulation/test/params.mat', {'paramsStruct': matlab_params})

    # -----------------------------
    # 2. Create MATLAB wrapper script
    # -----------------------------
    result_path = "../Offline-RL-Controller-in-AM/checkpoints/results/"
    matlab_folder = '../../LPBF-Simulation/'
    matlab_script = f"""
                    cd('{matlab_folder}');
                    paramsStruct = load('test/params.mat').paramsStruct;
                    if {i} > 0
                        prevData = load('test/results.mat', 'resultCool');
                        paramsStruct.ic = prevData.resultCool;
                    end
                    [uFinal, tAll, uAll, resultAll, model, meanDeviation] = simulateHeatingCooling(paramsStruct);
                    resultCool = resultAll(2);
                    save('test/results.mat','uFinal','tAll','uAll','meanDeviation','resultCool');
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

    print(f"Running Layer {i+1} simulation in MATLAB...")
    result = subprocess.run(
        [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r", f"try, run('{script_path}'); catch ME, disp(getReport(ME)); exit(1); end; exit(0);"
        ],
        text=True,
        capture_output=True
    )

    # if result.stdout.strip():
    #     print("MATLAB stdout:", result.stdout[-2000:])  # last 2000 chars to avoid spam
    # if result.stderr.strip():
    #     print("MATLAB stderr:", result.stderr[-2000:])

    if result.returncode != 0:
        raise RuntimeError(f"MATLAB failed with return code {result.returncode}")
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
    # IC for next layer is handled inside the MATLAB script via resultCool

    r_prev = -meanDeviation[0][0] # Used for P controller
    a_prev = params_dict['params']['LP']  # Used for P controller
    # print("uFinal shape:", uFinal.shape)
    # print("tAll shape:", tAll.shape)
