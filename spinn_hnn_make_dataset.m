function outPath = spinn_hnn_make_dataset(Ntraj, outPath, varargin)
% spinn_hnn_make_dataset
% 生成 H_θ（哈密顿网络）训练数据集：{Q,DQ,P,DPDT,TAU,Rdiag,Qn,Pn,dt,L,g}
%
% 关键：默认 cfg 与主线一致，确保数据分布与推理侧/采样侧口径统一
%   - dt = 0.002, t_final = 6.0, radius = 0.03
%   - 传入与主线一致的护栏/限幅配置（若老师核支持则生效，忽略则安全）
%
% I/O
%   Ntraj   : “轨迹条数”（建议 32~256）
%   outPath : 输出 .mat 路径（默认 'spinn_hnn_ds.mat'）
% 可选名值对：
%   'seed'      : 随机种子
%   'cfg'       : 结构体，覆盖仿真口径（默认见下）
%   'fix_dq0'   : 1x3 固定初始角速度（默认 [0 0 0]）
%   'Kp','Ki','Kd' : PID（默认 [50 50 50], [0.20 0.20 0.20], [0.10 0.10 0.10]）
%   'w'         : 1x3，老师核用的功率份额（默认均分）
%
% 依赖：spinn_RandomNumberGeneration / spinn_MechanicArm / computeDynamics
% 说明：仅“造数据”；HNN 训练在 spinn_hnn_train.m 进行。

    % ---------- 0) 解析参数 ----------
    if nargin < 1 || isempty(Ntraj), Ntraj = 64; end
    if nargin < 2 || isempty(outPath), outPath = 'spinn_hnn_ds.mat'; end
    p = inputParser; p.KeepUnmatched=true;

    % 与主线完全一致的默认 cfg（老师核若支持即采用）
    defCfg = struct( ...
        'dt',0.002, 't_final',6.0, 'radius',0.03, 'omega_eps',1e-3, ...
        'limiter_gamma',0.5, 'tau_smooth_tau',0.01, 'tau_rate_max',600, ...
        'ddq_abs_max',600, 'w_floor',0.12, 'w_cap',[0.45 0.60 0.60], ...
        'Pmax_gate',[] );

    addParameter(p,'seed',[],@(x) isempty(x) || isscalar(x));
    addParameter(p,'cfg', defCfg);                       % ★ 主线一致
    addParameter(p,'fix_dq0',[0 0 0],@(x) isnumeric(x) && numel(x)==3);
    addParameter(p,'Kp',[50 50 50]);                    % ★ 主线一致
    addParameter(p,'Ki',[0.20 0.20 0.20]);
    addParameter(p,'Kd',[0.10 0.10 0.10]);
    addParameter(p,'w',[1 1 1]/3);
    parse(p,varargin{:});
    opt = p.Results;

    if ~isempty(opt.seed), rng(double(opt.seed)); end

    % ---------- 1) 采样 25 维参数（与工程 schema 一致） ----------
    % 列义：1..3 m，4..6 dq0，7..12 damp/zeta，13..15 init(°)，16..18 tgt(°)，
    % 19..21 dθ=tgt-init，22 Pmax，23..25 Prated
    PM25 = spinn_RandomNumberGeneration(Ntraj, 'fix_dq0', opt.fix_dq0, 'seed', opt.seed);  % 25 列齐全

    % ---------- 2) 常量与存储 ----------
    g = 9.81; L = [0.24, 0.214, 0.324];
    Q=[]; DQ=[]; P=[]; DPDT=[]; TAU=[]; Rdiag=[]; Qn=[]; Pn=[];
    bad_total = 0;

    % ---------- 3) 逐条轨迹：老师核仿真 → 组装 (q,dq,p,dp/dt,τ) ----------
    for i = 1:size(PM25,1)
        row = PM25(i,:);                     % 25 列
        m       = row(1:3);
        dq0     = row(4:6);
        dampNom = [row(7) row(9) row(11)];   % ★ 阻尼索引与主线一致（7,9,11）
        initDeg = row(13:15);
        tgtDeg  = row(16:18);
        Pmax    = row(22);
        Prated  = row(23:25);                % 若老师核支持 cfg.Prated，则传入

        % —— 28 维参数：与 spinn_MechanicArm 完全对齐（PID + w）——
        params28 = [ m, dq0, ...
                     dampNom, tgtDeg, initDeg, Pmax, ...
                     opt.Kp(1), opt.Ki(1), opt.Kd(1), ...
                     opt.Kp(2), opt.Ki(2), opt.Kd(2), ...
                     opt.Kp(3), opt.Ki(3), opt.Kd(3), ...
                     normalize_w(opt.w) ];

        % —— cfg：显式补充 Prated，保持主线护栏参数一致 —— 
        cfg = opt.cfg;
        if ~isfield(cfg,'Prated'), cfg.Prated = Prated; end

        % —— 仿真（老师核返回完整轨迹信息）——
        try
            [~, info] = spinn_MechanicArm(params28, cfg);   % 老师核内部即用严格总功率/双顶帽/护栏
        catch ME
            warning('[%d/%d] spinn_MechanicAlarm 失败：%s', i, size(PM25,1), ME.message);
            bad_total = bad_total + 1; 
            continue;
        end

        % 时间与状态
        t  = info.t(:);    if numel(t) < 3, continue; end
        dt = mean(diff(t));    % 与 cfg.dt 对齐
        qh = info.q_history;      % k x 3 (rad)
        dqh= info.dq_history;     % k x 3 (rad/s)
        tau= info.tau_history;    % k x 3 (N·m)

        % —— 动量 p = M(q)dq（采用工程 computeDynamics）——
        K = size(qh,1);
        ptraj = zeros(K,3);
        for k = 1:K
            [Mq, ~, ~] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, qh(k,:).', dqh(k,:).');
            ptraj(k,:) = (Mq * dqh(k,:).').';
        end

        % —— 中心差分近似 dp/dt（端点复制）——
        dpdt = zeros(K,3);
        dpdt(2:K-1,:) = (ptraj(3:end,:) - ptraj(1:end-2,:)) / (2*dt);
        dpdt(1,:)     = dpdt(2,:); 
        dpdt(K,:)     = dpdt(K-1,:);

        % —— 组装 (k, k+1) 配对样本 —— 
        Q   = [Q;   qh(1:K-1,:)];           
        DQ  = [DQ;  dqh(1:K-1,:)];          
        P   = [P;   ptraj(1:K-1,:)];        
        DPDT= [DPDT;dpdt(1:K-1,:)];         
        TAU = [TAU; tau(1:K-1,:)];          
        Rdiag=[Rdiag; repmat(dampNom, K-1, 1)]; 
        Qn  = [Qn;  qh(2:K,:)];             
        Pn  = [Pn;  ptraj(2:K,:)];          
    end

    % ---------- 4) 清洗与落盘 ----------
    ALL = [Q DQ P DPDT TAU Rdiag Qn Pn];
    good = all(isfinite(ALL),2);
    bad = nnz(~good); 
    if bad>0
        warning('清理掉 %d 条含 NaN/Inf 的样本。', bad);
    end
    Q=Q(good,:); DQ=DQ(good,:); P=P(good,:); DPDT=DPDT(good,:); 
    TAU=TAU(good,:); Rdiag=Rdiag(good,:); Qn=Qn(good,:); Pn=Pn(good,:);

    save(outPath,'Q','DQ','P','DPDT','TAU','Rdiag','Qn','Pn','dt','L','g','-v7.3');
    fprintf('[spinn_hnn_make_dataset] 完成：Ntraj=%d → 样本数=%d，保存到：%s\n', Ntraj, size(Q,1), outPath);
    if bad_total>0
        fprintf('注意：有 %d 条轨迹在仿真阶段失败/跳过。\n', bad_total);
    end
end

% === 工具：份额归一化 ===
function w = normalize_w(w)
    w = max(w(:).',0);
    s = sum(w); if s<=0, w = ones(size(w))/numel(w); else, w = w/s; end
end
