clear; clc;

addpath('funs');

% load dataset
load('proteinFold_Kmatrix');
KH = knorm(kcenter(KH));

n = size(KH, 1);
numclass = numel(unique(Y));
Y_init = Y_Initialize(n, numclass);

tic;
[y_pred, alpha, obj] = DPMKKM(KH, Y_init);
toc
ClusteringMeasure_new(Y, y_pred)
