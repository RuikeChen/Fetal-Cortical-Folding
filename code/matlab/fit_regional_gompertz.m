%% Reproducible regional Gompertz analysis (FDrad versus curvature)
% Primary inclusion rule intentionally matches the original analysis:
% FDR(model P)<.05, FDR(Tini versus zero)<.05, and Tini inside the
% modality/ROI-specific observed GA range after outlier exclusion.
% Additional fit-quality flags are exported as sensitivity analyses only.

clear; close all; clc

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir = fullfile(rootDir,'results','generated','regional_gompertz');
if ~exist(outDir,'dir'), mkdir(outDir); end

afdTbl = readtable(fullfile(rootDir,'data','derived','zju_fdrad_roi_wide.csv'), ...
    'VariableNamingRule','preserve');
curvTbl = readtable(fullfile(rootDir,'data','derived','zju_curvature_roi_wide.csv'), ...
    'VariableNamingRule','preserve');

gaAfd = double(afdTbl{:,4});
gaCurv = double(curvTbl{:,4});

regionNames = string(afdTbl.Properties.VariableNames(5:end))';
nRegion = numel(regionNames);
model = @(b,x) b(1) + b(2).*exp(-exp(-b(3).*(x-b(4))+1));
opts = statset('nlinfit');
opts.MaxIter = 2000;
opts.TolFun = 1e-10;
opts.TolX = 1e-10;

metric = strings(2*nRegion,1);
region = strings(2*nRegion,1);
n = nan(2*nRegion,1); gaMin = n; gaMax = n;
a = n; amplitude = n; speed = n; tini = n; tiniSE = n;
tiniCILow = n; tiniCIHigh = n; r2 = n; pModel = n; pTini = n;
exitOK = false(2*nRegion,1); expectedDirection = false(2*nRegion,1);

for r = 1:nRegion
    yAfd = double(afdTbl{:,r+4});
    yCurv = double(curvTbl{:,r+4});
    datasets = {gaAfd,yAfd,'FDrad',[0.5 -0.5 0.5 25],-1; ...
                gaCurv,yCurv,'Curvature',[0.1 0.5 0.5 25],1};
    for m = 1:2
        row = (m-1)*nRegion+r;
        x0 = datasets{m,1}; y0 = datasets{m,2};
        keep = isfinite(x0) & isfinite(y0) & ~isoutlier(y0);
        x = x0(keep); y = y0(keep);
        metric(row) = datasets{m,3}; region(row) = regionNames(r);
        n(row)=numel(x); gaMin(row)=min(x); gaMax(row)=max(x);
        try
            mdl = fitnlm(x,y,model,datasets{m,4},'Options',opts);
            ci = coefCI(mdl,0.05);
            co = mdl.Coefficients;
            a(row)=co.Estimate(1); amplitude(row)=co.Estimate(2);
            speed(row)=co.Estimate(3); tini(row)=co.Estimate(4);
            tiniSE(row)=co.SE(4); pTini(row)=co.pValue(4);
            tiniCILow(row)=ci(4,1); tiniCIHigh(row)=ci(4,2);
            r2(row)=mdl.Rsquared.Ordinary;
            pModel(row)=coefTest(mdl);
            exitOK(row)=all(isfinite(co.Estimate)) && speed(row)>0;
            expectedDirection(row)=sign(amplitude(row))==datasets{m,5};
        catch ME
            warning('Fit failed for %s, %s: %s',region(row),metric(row),ME.message)
        end
    end
end

fitTable = table(metric,region,n,gaMin,gaMax,a,amplitude,speed,tini,tiniSE, ...
    tiniCILow,tiniCIHigh,r2,pModel,pTini,exitOK,expectedDirection);
fitTable.qModel = nan(height(fitTable),1);
fitTable.qTini = nan(height(fitTable),1);
for m = ["FDrad","Curvature"]
    ix = fitTable.metric==m;
    fitTable.qModel(ix)=bhFdr(fitTable.pModel(ix));
    fitTable.qTini(ix)=bhFdr(fitTable.pTini(ix));
end
fitTable.primaryPass = fitTable.qModel<0.05 & fitTable.qTini<0.05 & ...
    fitTable.tini>fitTable.gaMin & fitTable.tini<fitTable.gaMax;
fitTable.ciWidth = fitTable.tiniCIHigh-fitTable.tiniCILow;
fitTable.strictPass = fitTable.primaryPass & fitTable.exitOK & ...
    fitTable.expectedDirection & isfinite(fitTable.ciWidth) & fitTable.ciWidth<=8;
writetable(fitTable,fullfile(outDir,'all_regional_fits.csv'));

afdFit = fitTable(fitTable.metric=="FDrad",:);
curvFit = fitTable(fitTable.metric=="Curvature",:);
result = table(regionNames,afdFit.tini,curvFit.tini,curvFit.tini-afdFit.tini, ...
    afdFit.primaryPass & curvFit.primaryPass, ...
    afdFit.strictPass & curvFit.strictPass, ...
    'VariableNames',{'region','tiniFDrad','tiniCurvature','deltaWeeks', ...
    'primaryIncluded','strictIncluded'});
writetable(result,fullfile(outDir,'paired_regional_tini.csv'));

criteria = ["primaryIncluded";"strictIncluded"];
nIncluded = nan(2,1); pearsonR=nIncluded; pearsonP=nIncluded;
meanDelta=nIncluded; medianDelta=nIncluded; proportionFDradEarlier=nIncluded;
for j=1:2
    ix=result.(criteria(j)); nIncluded(j)=sum(ix);
    if nIncluded(j)>=4
        [rho,pv]=corr(result.tiniFDrad(ix),result.tiniCurvature(ix),'Rows','complete');
        pearsonR(j)=rho; pearsonP(j)=pv;
        meanDelta(j)=mean(result.deltaWeeks(ix),'omitnan');
        medianDelta(j)=median(result.deltaWeeks(ix),'omitnan');
        proportionFDradEarlier(j)=mean(result.deltaWeeks(ix)>0,'omitnan');
    end
end
summary=table(criteria,nIncluded,pearsonR,pearsonP,meanDelta,medianDelta,proportionFDradEarlier);
writetable(summary,fullfile(outDir,'summary.csv'));
disp(summary)

function q = bhFdr(p)
% Benjamini-Hochberg adjusted P values, preserving NaNs and input shape.
q=nan(size(p)); ok=isfinite(p); pv=p(ok); m=numel(pv);
if m==0, return; end
[ps,ord]=sort(pv(:)); adj=ps.*m./(1:m)';
adj=flipud(cummin(flipud(adj))); adj=min(adj,1);
tmp=nan(m,1); tmp(ord)=adj; q(ok)=tmp;
end
