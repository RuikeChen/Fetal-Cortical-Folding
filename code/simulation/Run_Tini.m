clear; clc;
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(fileparts(scriptDir));
outDir = fullfile(rootDir,'results','generated','simulation','fiber_timing');
if ~exist(outDir,'dir'), mkdir(outDir); end
mliPath = getenv('COMSOL_MLI_PATH');
if isempty(mliPath), error('Set COMSOL_MLI_PATH to the COMSOL Multiphysics mli directory.'); end
addpath(mliPath);
import com.comsol.model.*
import com.comsol.model.util.*
model = mphload(fullfile(scriptDir,'Model_Tini.mph'));
values = [0,0.2,0.4,0.8];
for i = 1:numel(values)
    value = values(i);
    fprintf('Running fiber timing simulation: %.2f\n',value);
    model.variable('var4').set('t_ini',num2str(value));
    model.study('std1').run;
    timeSteps = model.sol('sol1').getPVals;
    model.result.numerical('av2').set('t',timeSteps); model.result.numerical('av2').setResult;
    curvature = mphtable(model,model.result.numerical('av2').getString('table'));
    model.result.numerical('meas2').set('t',timeSteps); model.result.numerical('meas2').setResult;
    diameter = mphtable(model,model.result.numerical('meas2').getString('table'));
    writematrix(curvature.data,fullfile(outDir,sprintf('curvature_tini_%g.csv',value)));
    writematrix(diameter.data,fullfile(outDir,sprintf('diameter_tini_%g.csv',value)));
    model.sol('sol1').clearSolution();
end
