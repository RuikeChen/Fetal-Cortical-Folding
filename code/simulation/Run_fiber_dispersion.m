clear; clc;
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(fileparts(scriptDir));
outDir = fullfile(rootDir,'results','generated','simulation','fiber_dispersion');
if ~exist(outDir,'dir'), mkdir(outDir); end
mliPath = getenv('COMSOL_MLI_PATH');
if isempty(mliPath), error('Set COMSOL_MLI_PATH to the COMSOL Multiphysics mli directory.'); end
addpath(mliPath);
import com.comsol.model.*
import com.comsol.model.util.*
model = mphload(fullfile(scriptDir,'Model_main.mph'));
values = [0,0.0640,0.1321,0.2083,0.2790,0.3333];
for i = 1:numel(values)
    value = values(i);
    fprintf('Running fiber dispersion simulation: %.4f\n',value);
    model.physics('solid').feature('hmm1').feature('fib1').set('k3HGO',num2str(value));
    model.study('std1').run;
    timeSteps = model.sol('sol1').getPVals;
    model.result.numerical('av2').set('t',timeSteps); model.result.numerical('av2').setResult;
    curvature = mphtable(model,model.result.numerical('av2').getString('table'));
    model.result.numerical('meas2').set('t',timeSteps); model.result.numerical('meas2').setResult;
    diameter = mphtable(model,model.result.numerical('meas2').getString('table'));
    writematrix(curvature.data,fullfile(outDir,sprintf('curvature_dispersion_%g.csv',value)));
    writematrix(diameter.data,fullfile(outDir,sprintf('diameter_dispersion_%g.csv',value)));
    model.sol('sol1').clearSolution();
end
