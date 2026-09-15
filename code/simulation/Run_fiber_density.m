clear; clc;
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(fileparts(scriptDir));
outDir = fullfile(rootDir,'results','generated','simulation','fiber_density');
if ~exist(outDir,'dir'), mkdir(outDir); end
mliPath = getenv('COMSOL_MLI_PATH');
if isempty(mliPath), error('Set COMSOL_MLI_PATH to the COMSOL Multiphysics mli directory.'); end
addpath(mliPath);
import com.comsol.model.*
import com.comsol.model.util.*
model = mphload(fullfile(scriptDir,'Model_main.mph'));
values = [0,0.1,0.5,1.0];
for i = 1:numel(values)
    value = values(i);
    fprintf('Running fiber density simulation: %.2f\n',value);
    model.variable('var4').set('fiber_density',num2str(value));
    model.study('std1').run;
    timeSteps = model.sol('sol1').getPVals;
    model.result.numerical('av2').set('t',timeSteps); model.result.numerical('av2').setResult;
    curvature = mphtable(model,model.result.numerical('av2').getString('table'));
    model.result.numerical('meas2').set('t',timeSteps); model.result.numerical('meas2').setResult;
    diameter = mphtable(model,model.result.numerical('meas2').getString('table'));
    writematrix(curvature.data,fullfile(outDir,sprintf('curvature_density_%g.csv',value)));
    writematrix(diameter.data,fullfile(outDir,sprintf('diameter_density_%g.csv',value)));
    model.sol('sol1').clearSolution();
end
