function spinn_main(n, num_trials, out_path)
% SPINN 数据生成主程序（采样 → 逐行优化落盘 → 清洗）
% 布局：1..25=输入，26..28=最优权重，29=最短到达时间
% 依赖：spinn_RandomNumberGeneration / spinn_DatasetGeneration
%
% 用法：
%   spinn_main();                                  % 默认 3000 行，每行 10 次重启
%   spinn_main(5000, 8, 'C:\...\SpinnMechanicalArmParams.mat');

    % ---- 参数与默认值 ----
    if nargin < 1 || isempty(n), n = 48; end
    if nargin < 2 || isempty(num_trials), num_trials = 10; end
    if nargin < 3 || isempty(out_path)
        out_path = fullfile('C:','Users','kndx9','Desktop','SpinnMechanicalArmParams.mat');
    end
    n = max(1, round(double(n)));
    num_trials = max(1, round(double(num_trials)));
    fprintf('[spinn_main] 目标样本数: %d, 每行重启: %d\n', n, num_trials);
    fprintf('[spinn_main] 输出文件: %s\n', out_path);

    % ---- 0) 预建空文件（无则建） ----
    ensure_outfile(out_path);

    % ---- 1) 采样（25 维，与下游严格同口径）----
    % 采样器定义：1..3 m；4..6 dq0；7..12 damp/zeta（阻尼取 7,9,11）；
    % 13..15 init；16..18 tgt；19..21 dtheta=tgt-init；22 Pmax；23..25 Prated
    PM25 = [];
    try
        PM25 = spinn_RandomNumberGeneration(n);                 % 标准 25 列输出
    catch ME
        error('[spinn_main] 采样器失败：%s', ME.message);
    end
    if isempty(PM25) || size(PM25,2) ~= 25
        error('[spinn_main] 采样失败或维度异常（期望 25 列）。');
    end
    fprintf('[spinn_main] 采样完成：%d 行 x %d 列。\n', size(PM25,1), size(PM25,2));

    % ---- 2) 逐行优化 + 追加落盘（29 列：25|w|t） ----
    m = size(PM25,1);
    n_failed = 0;
    t0 = tic;
    for i = 1:m
        try
            spinn_DatasetGeneration(PM25(i,:), num_trials, out_path);
            if mod(i,10)==0 || i==m
                dt = toc(t0);
                fprintf('[spinn_main] 已成功追加至第 %d/%d 行（%.1fs）。\n', i, m, dt);
            end
        catch ME
            n_failed = n_failed + 1;
            warning('[spinn_main] 第 %d/%d 行失败：%s', i, m, ME.message);
        end
        if mod(i, 20) == 0
            close all; drawnow;
        end
    end
    if n_failed>0
        warning('[spinn_main] 共 %d 行失败（请上滚查看第一条失败原因）。', n_failed);
    end

    % ---- 3) 清洗（去 NaN/0 行；仅对本 SPINN 文件）----
    try
        spinn_cleandata(out_path);     % 若工程里没有该函数，这里在 try 内，不会影响流程
    catch ME
        warning('[spinn_main] spinn_cleandata 失败：%s', ME.message);
    end

    % ---- 4) 汇总信息 ----
    try
        if isfile(out_path)
            S = load(out_path);
            if isfield(S,'params_matrix')
                fprintf('[spinn_main] 产出: %s; 大小: %dx%d\n', out_path, size(S.params_matrix,1), size(S.params_matrix,2));
            else
                warning('[spinn_main] %s 中未找到变量 params_matrix。', out_path);
            end
        else
            warning('[spinn_main] 未找到输出 MAT 文件: %s', out_path);
        end
    catch ME
        warning('[spinn_main] 汇总信息读取失败：%s', ME.message);
    end
end

% --- 创建空输出文件（带 params_matrix 变量），目录不存在则 mkdir ---
function ensure_outfile(fp)
    folder = fileparts(fp);
    if ~exist(folder,'dir'), mkdir(folder); end
    if ~isfile(fp)
        params_matrix = []; %#ok<NASGU>
        save(fp, 'params_matrix', '-v7.3');
        fprintf('[spinn_main] 已创建空文件：%s\n', fp);
    end
end
