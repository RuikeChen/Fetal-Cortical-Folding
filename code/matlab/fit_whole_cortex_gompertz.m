clear; clc;

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataDir = fullfile(rootDir, 'data', 'derived');
outDir = fullfile(rootDir, 'results', 'generated', 'whole_cortex_gompertz');
if ~exist(outDir, 'dir'), mkdir(outDir); end

curvTbl = readtable(fullfile(dataDir, 'zju_whole_cortex_curvature.csv'), ...
    'VariableNamingRule', 'preserve');
fdradTbl = readtable(fullfile(dataDir, 'zju_whole_cortex_fdrad.csv'), ...
    'VariableNamingRule', 'preserve');

model = @(p,x) p(1) + p(2).*exp(-exp(-p(3).*(x-p(4))+1));
opts = statset('nlinfit');
opts.MaxIter = 2000;
opts.TolFun = 1e-10;
opts.TolX = 1e-10;

spec = {
    "curvature", curvTbl.ga_weeks, curvTbl.curvature, [0.1 0.5 0.5 25];
    "fdrad", fdradTbl.ga_weeks, fdradTbl.fdrad, [0.5 -0.5 0.5 25]
};

metric = strings(2,1); n = zeros(2,1); tini = nan(2,1); tiniSE = nan(2,1);
tiniCILow = nan(2,1); tiniCIHigh = nan(2,1); r2 = nan(2,1);
for i = 1:2
    metric(i) = spec{i,1};
    x = double(spec{i,2}); y = double(spec{i,3});
    keep = isfinite(x) & isfinite(y) & ~isoutlier(y);
    x = x(keep); y = y(keep); n(i) = numel(x);
    mdl = fitnlm(x, y, model, spec{i,4}, 'Options', opts);
    ci = coefCI(mdl, 0.05);
    tini(i) = mdl.Coefficients.Estimate(4);
    tiniSE(i) = mdl.Coefficients.SE(4);
    tiniCILow(i) = ci(4,1);
    tiniCIHigh(i) = ci(4,2);
    r2(i) = mdl.Rsquared.Ordinary;
end

result = table(metric, n, tini, tiniSE, tiniCILow, tiniCIHigh, r2);
writetable(result, fullfile(outDir, 'whole_cortex_gompertz.csv'));

delta = tini(1)-tini(2);
z = delta/sqrt(tiniSE(1)^2+tiniSE(2)^2);
comparison = table(delta, z, 'VariableNames', {'deltaWeeks','zIndependentParameterSE'});
writetable(comparison, fullfile(outDir, 'whole_cortex_tini_comparison.csv'));
disp(result)
disp(comparison)
