% surrogate_model/extract_mesh.m
% -----------------------------------------------------------------------
% One-time script: extract PDE mesh node coordinates and element
% connectivity from a saved simulation result, then write them to
%   surrogate_model/mesh.mat
%
% Run once from the LPBF-Simulation directory, e.g.:
%   cd /path/to/LPBF-Simulation
%   matlab -nodisplay -r "run('../Offline-RL-Controller-in-AM/surrogate_model/extract_mesh.m'); exit"
%
% Output (saved to surrogate_model/mesh.mat):
%   nodes    – (2, N_nodes) float64  [X; Y] coordinates of each node
%   elements – (3, N_tri)  uint32   node indices for each triangle
% -----------------------------------------------------------------------

% Load any one simulation result that contains the PDE model object
result_file = '/nfs/turbo/coe-sunwbgt/xysong/LPBF-Simulation/RL_Dataset/trajectory_001/layer_1_data.mat';
data = load(result_file);

% Print all fields so we can see what's actually in the struct
fn = fieldnames(data);
fprintf('Fields in .mat file:\n');
for i = 1:numel(fn)
    fprintf('  [%d] %s\n', i, fn{i});
end

% Search all fields for the pde.TimeDependentResults object by type
results_obj = [];
for i = 1:numel(fn)
    val = data.(fn{i});
    if isa(val, 'pde.TimeDependentResults')
        results_obj = val;
        fprintf('Found pde.TimeDependentResults in field: %s\n', fn{i});
        break;
    end
end

if isempty(results_obj)
    error(['Could not find a pde.TimeDependentResults object in any field. ' ...
           'Check the field list printed above.']);
end

model = results_obj.Mesh;      % pde.FEMesh

% Extract geometry
nodes    = model.Nodes;        % (2, N_nodes)
elements = model.Elements;     % (3, N_tri)  — linear triangles

fprintf('Mesh: %d nodes, %d triangles\n', size(nodes,2), size(elements,2));
fprintf('X range: [%.3f, %.3f]\n', min(nodes(1,:)), max(nodes(1,:)));
fprintf('Y range: [%.3f, %.3f]\n', min(nodes(2,:)), max(nodes(2,:)));

% Save to Python-readable .mat file
out_path = '/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model/mesh.mat';
save(out_path, 'nodes', 'elements');
fprintf('Mesh saved to: %s\n', out_path);
