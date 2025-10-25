function [w_opt, t_best] = spinn_optimizemechanicalarm(fixed25, num_trials, Prated, Pmax, varargin)
% SPINN 权重优化器：在“严格总功率 + 双顶帽”仿真核下最小化到达时间
%
% 输入
%   fixed25    : 1x25 物理与任务参数（不含 w），由上游生成（见 spinn_DatasetGeneration）
%   num_trials : 重启次数（默认 10）
%   Prated     : 1x3，各轴额定功率（W）
%   Pmax       : 标量，总功率上限（W）
% 名值可选
%   'cfg'       : 结构体，传给 spinn_MechanicArm；若未给，则使用本文件 local_defaults()
%   'objective' : 'time'(默认) 或 'joint'（时间 + λ·末端命中速度）
%   'lambdav'   : 联合目标速度权重 λ（默认 0.0）
%
% 输出
%   w_opt  : 1x3，满足 ∑w=1 且 0≤w≤Prated/Pmax 的最优权重
%   t_best : 标量，最优（有限）损失（time 或 joint）
%
% 说明
%   - 优先调用 spinn_MechanicArm(full28, cfg) 来评估 w 的命中时间；
%   - 若你的工程中仍保留旧核（MechanicAlarm 或 mechanical_arm2），会自动回退。
%
% 依赖
%   - 本函数供 spinn_DatasetGeneration 逐行调用以写出 29 列数据集。见上游脚本。 

    % ---------- 1) 参数解析 ----------
    if nargin < 2 || isempty(num_trials), num_trials = 10; end
    if nargin < 3 || isempty(Prated),    Prated = [Inf,Inf,Inf]; end
    if nargin < 4 || isempty(Pmax),      Pmax   =  Inf; end
    assert(isvector(fixed25) && numel(fixed25)==25, 'fixed25 必须是 1x25。');

    opts = struct('cfg',[], 'objective','time', 'lambdav', 0.0);
    if ~isempty(varargin)
        if mod(numel(varargin),2)~=0, error('名值对必须成对出现。'); end
        for k=1:2:numel(varargin)
            key = lower(string(varargin{k}));
            val = varargin{k+1};
            switch key
                case "cfg",        opts.cfg = val;
                case "objective",  opts.objective = char(val);
                case "lambdav",    opts.lambdav  = double(val);
                otherwise, warning('未知选项 %s 已忽略。', key);
            end
        end
    end
    cfg = local_defaults();
    if ~isempty(opts.cfg), cfg = merge_struct(cfg, opts.cfg); end

    use_joint = strcmpi(opts.objective,'joint');
    lambda_v  = max(0.0, double(opts.lambdav));

    % ---------- 2) 选择仿真核 ----------
    sim_fun = pick_simulator(fixed25, cfg);  % [t_raw, v_end] = sim_fun(w)

    % ---------- 3) 上界（由额定功率） ----------
    cap = Prated(:).' / max(Pmax, eps);
    cap(~isfinite(cap)) = 1;
    cap = max(0, min(1, cap));   % 0 ≤ cap_i ≤ 1

    % ---------- 4) 多重重启 + 投影梯度 ----------
    t_best = inf; 
    w_opt  = [1,0,0];

    seeds  = [ ones(1,3)/3; cap/max(sum(cap),eps); rand(1,3) ];
    n_seed = size(seeds,1);

    for tr = 1:max(num_trials, n_seed)
        if tr <= n_seed, w = seeds(tr,:); else, w = rand(1,3); end
        w = project_capped_simplex(w, cap);

        lr = 0.05; maxit = 300; tol = 1e-5; last = inf;
        for it = 1:maxit
            f = eval_loss(sim_fun, w, use_joint, lambda_v, cfg);
            g = fd_grad(@(z) eval_loss(sim_fun, project_capped_simplex(z,cap), use_joint, lambda_v, cfg), w);

            % 一步下降 + 带上界投影
            w_new = project_capped_simplex(w - lr*g, cap);

            % 简易回溯线搜
            f_new = eval_loss(sim_fun, w_new, use_joint, lambda_v, cfg);
            bt=0;
            while f_new > f && bt < 6
                lr = lr * 0.5;
                w_new = project_capped_simplex(w - lr*g, cap);
                f_new = eval_loss(sim_fun, w_new, use_joint, lambda_v, cfg);
                bt = bt + 1;
            end
            w = w_new; f = f_new;

            if abs(last - f) < tol, break; end
            last = f;
        end

        if f < t_best
            t_best = f;
            w_opt  = w;
        end
    end
end

% ====== 默认 cfg（与当前推理核口径对齐） ======
function cfg = local_defaults()
    cfg = struct( ...
        'dt',0.002, ...
        't_final',6.0, ...
        'radius',0.03, ...
        'omega_eps',1e-3, ...
        ... % —— 以下字段由仿真核读取，用于严格总功率 + 双顶帽 + 护栏 ——
        'limiter_gamma',0.5, ...
        'tau_smooth_tau',0.01, ...
        'tau_rate_max',600, ...
        'ddq_abs_max',600, ...
        'w_floor',0.12, ...
        'w_cap',[0.45 0.60 0.60], ...
        'Pmax_gate',[] ...
    );
end

% ====== 选择仿真核，返回“带惩罚/联合目标”的评估器 ======
function sim_fun = pick_simulator(fixed25, cfg)
    if exist('spinn_MechanicArm','file') == 2
        % 用 SPINN 核：full28 = [fixed25, w]
        sim_fun = @(w) sim_spinn([fixed25, w], cfg);
    elseif exist('MechanicAlarm','file') == 2
        % 兼容旧核（只拿时间）
        sim_fun = @(w) sim_old([fixed25, w]);
    else
        % 兜底
        assert(exist('mechanical_arm2','file')==2, '未找到仿真核。');
        sim_fun = @(w) sim_fallback([fixed25, w]);
    end
end

% --- 用 SPINN 核：返回 [t_raw, v_end]；失败返回 NaN ---
function [t_raw, v_end] = sim_spinn(full28, cfg)
    try
        try
            [t_raw, info] = spinn_MechanicArm(full28, cfg);
            if isstruct(info) && isfield(info,'end_speed')
                v_end = info.end_speed;
            else
                v_end = NaN;
            end
        catch
            t_raw = spinn_MechanicArm(full28, cfg);
            v_end = NaN;
        end
    catch
        t_raw = NaN; v_end = NaN;
    end
end

% --- 旧核（无 cfg）：只返回时间 ---
function [t_raw, v_end] = sim_old(full28)
    try
        t_raw = MechanicAlarm(full28); 
    catch
        t_raw = NaN;
    end
    v_end = NaN;
end

% --- 兜底 ---
function [t_raw, v_end] = sim_fallback(full28)
    try
        [~,~,~,~,t_raw] = mechanical_arm2(full28);
    catch
        t_raw = NaN;
    end
    v_end = NaN;
end

% ====== 将“原始时间/末速”映射为优化损失（含惩罚） ======
function loss = eval_loss(sim_fun, w, use_joint, lambda_v, cfg)
    [t_raw, v_end] = sim_fun(w);

    % 不可达或异常 → 罚 t_penalty（有限）
    t_penalty = max(cfg.t_final + 1, 60);
    if ~isfinite(t_raw) || t_raw <= 0
        loss = t_penalty;
        return;
    end

    if ~use_joint
        loss = t_raw;
    else
        v = v_end; if ~isfinite(v), v = 0.0; end
        loss = t_raw + lambda_v * max(0, v);
    end
end

% ====== 中心差分数值梯度 ======
function g = fd_grad(fun, w)
    h = 1e-2; g = zeros(size(w));
    for i=1:numel(w)
        e = zeros(size(w)); e(i)=1;
        g(i) = (fun(w + h*e) - fun(w - h*e)) / (2*h);
    end
end

% ====== 带上界的单纯形投影（∑=1, 0≤w≤cap），稳定二分 ======
function w = project_capped_simplex(w, cap)
    w   = max(w(:).', 0);
    cap = max(cap(:).', 0); cap(~isfinite(cap)) = 1;

    % 可行域判定：若 sum(cap) < 1，则严格满足 ∑w=1 不可行；退化为 cap 的相对比例
    if sum(cap) < 1 - 1e-12
        if sum(cap) <= 0
            w = ones(1,numel(w))/numel(w);
        else
            w = cap / sum(cap);
        end
        return;
    end

    lo = min(w - cap); hi = max(w);
    for it=1:60
        mu = 0.5*(lo+hi);
        v  = min(max(w - mu, 0), cap);
        s  = sum(v);
        if abs(s - 1) < 1e-12, w = v; return; end
        if s > 1, lo = mu; else, hi = mu; end
    end
    w = min(max(w - mu, 0), cap);  % 容错
end

% ====== 结构体浅合并 ======
function O = merge_struct(O, U)
    if nargin<2 || ~isstruct(U), return; end
    f = fieldnames(U);
    for i=1:numel(f), O.(f{i}) = U.(f{i}); end
end
