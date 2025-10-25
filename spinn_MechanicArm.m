function [out1, out2] = spinn_MechanicArm(varargin)
% SPINN 机械臂核心（统一入口）
% - 工厂模式：  [mech] = spinn_MechanicArm(L, m, g, use_rst, eeName)
% - 仿真模式：  [t_hit, info] = spinn_MechanicAlarm(params28, cfg)
%
% 仿真模式对齐当前主线控制思想：
%   严格总功率：先扣 P_G = sum(|G ⊙ dq|)，任务/零空间仅用 P_avail = Pmax_eff - P_G
%   双顶帽：先总功率 → 分轴软帽（指数 γ）
%   执行护栏：扭矩低通 + 速率限 + ddq 限
%
% params28 布局（与数据生成/优化一致）：
%   [ m(3) dq0(3) damp(3) tgtDeg(3) initDeg(3) Pmax  PID(9)  w(3) ]
%
% 返回：
%   工厂模式：mech 结构体，含句柄：mech.fk(q)->(x,y), mech.jac(q), mech.dyn(q,dq)->[M,C,G]
%   仿真模式：t_hit（命中时间或 NaN），info（含时序轨迹/末端速度/扭矩等）

    % -------- 模式分流 --------
    if nargin>=1 && isnumeric(varargin{1}) && isvector(varargin{1}) && numel(varargin{1})>=25
        % ---------- 仿真模式 ----------
        params = double(varargin{1}(:).');
        cfg    = struct();
        if nargin>=2 && isstruct(varargin{2}), cfg = varargin{2}; end
        [t_hit, info] = local_simulate_from_params(params, cfg);
        out1 = t_hit; out2 = info;
    else
        % ---------- 工厂模式 ----------
        [L, m, g, use_rst, eeName] = local_parse_factory_inputs(varargin{:});
        mech = local_build_mech(L, m, g, use_rst, eeName);
        out1 = mech; out2 = [];
    end
end

% ===================== 仿真模式实现 =====================
function [t_hit, info] = local_simulate_from_params(P, cfg)
    % ----- 1) 解析参数（与数据/优化一致）-----
    L = [0.24 0.214 0.324];  g = 9.81;
    m        = P(1:3);
    dq       = P(4:6).';
    dampNom  = P(7:9).';
    qTdeg    = P(10:12);
    q0deg    = P(13:15);
    Pmax     = P(16);
    % PID 9 个占位（17..25），本仿真不直接用；保留口径一致
    if numel(P) >= 28, w_fix = P(26:28); else, w_fix = [1 1 1]/3; end
    w_fix = max(0, w_fix(:).'); if sum(w_fix)<=0, w_fix = [1 1 1]/3; else, w_fix = w_fix/sum(w_fix); end

    % ----- 2) cfg 默认（与主线一致）-----
    def = struct('dt',0.002,'t_final',6.0,'radius',0.03, ...
                 'Kp_xy',[80 80], 'Kd_xy',[14 14], ...
                 'limiter_gamma',0.5, ...
                 'tau_smooth_tau',0.01, 'tau_rate_max',600, ...
                 'ddq_abs_max',600, ...
                 'w_floor',0.12, 'w_cap',[0.45 0.60 0.60], ...
                 'Pmax_gate',[], 'Prated',[Inf Inf Inf], ...
                 'use_rst',true);
    cfg = local_merge(def, cfg);

    % ----- 3) 力学核（工厂）-----
    mech   = local_build_mech(L, m, g, cfg.use_rst, 'ee');
    fk_fun = mech.fk;    jac_fun = mech.jac;    dyn_fun = mech.dyn;

    % 初始/目标
    q  = deg2rad(q0deg(:));   dq = dq;
    qT = deg2rad(qTdeg(:));
    [xT,yT] = fk_fun(qT);

    % 记录器
    tvec = (0:cfg.dt:cfg.t_final).';
    K    = numel(tvec);
    q_hist  = nan(K,3); dq_hist = nan(K,3); tau_hist = nan(K,3);
    q_hist(1,:)=q.'; dq_hist(1,:)=dq.'; tau_prev = zeros(3,1);
    tau_lp = zeros(3,1);

    t_hit = NaN; end_speed = NaN;

    for k = 2:K
        % 动力学
        [M,C,G] = dyn_fun(q, dq);

        % 几何/任务状态
        [xE,yE] = fk_fun(q);
        J  = jac_fun(q);
        vE = J*dq;
        e  = [xT-xE; yT-yE]; d = hypot(e(1),e(2));
        % 任务空间 PD
        F = [cfg.Kp_xy(1)*e(1) - cfg.Kd_xy(1)*vE(1);
             cfg.Kp_xy(2)*e(2) - cfg.Kd_xy(2)*vE(2)];
        tau_task = J.' * F;

        % 重力不缩（但功率计入预算）
        tau_G = G;

        % 任务部分（本仿真不加零空间项，避免与不同上层策略耦合）
        tau_h = tau_task;

        % 扭矩幅值/速率限（仅对 tau_h；tau_G 不限）
        if isfinite(cfg.tau_rate_max)
            dr = cfg.tau_rate_max * cfg.dt;
            tau_h = min(max(tau_h, tau_prev - dr), tau_prev + dr);
        end
        tau_prev = tau_h;

        % ---- 严格总功率预算（含重力功率）----
        Pmax_eff = Pmax;
        if isa(cfg.Pmax_gate,'function_handle')
            Pmax_eff = max(0, min(Pmax, cfg.Pmax_gate(d, Pmax, k, tvec(k))));
        end

        % 真实 dq 计算功率，不加 eps
        pG = abs(tau_G(:) .* dq(:));  P_G = sum(pG);
        P_avail = max(0, Pmax_eff - P_G);

        % 先总功率（任务功率）
        S_task = sum(abs(tau_h(:).*dq(:)));
        if S_task > (P_avail + 1e-12) && S_task>0
            tau_h = tau_h * (P_avail / S_task);
        end

        % 分轴软帽：cap_i = min(w_i * P_avail, Prated_i)
        cap_i = min(w_fix(:) * P_avail, cfg.Prated(:));
        p_ax  = abs(tau_h(:) .* dq(:));
        gamma = max(0.2, min(1.0, cfg.limiter_gamma));
        for ii = 1:3
            if p_ax(ii) > (cap_i(ii) + 1e-12)
                r = cap_i(ii) / max(p_ax(ii), eps);
                tau_h(ii) = tau_h(ii) * r^gamma;
            end
        end

        % 合成 + 输出侧低通
        tau_cmd_pre = tau_h + tau_G;
        alpha_tau = 1 - exp(-cfg.dt / max(1e-6, cfg.tau_smooth_tau));
        tau_cmd = tau_lp + alpha_tau * (tau_cmd_pre - tau_lp);
        tau_lp  = tau_cmd;

        % 末端校核（低通后可能微越界）：只缩任务部分
        P_total = sum(abs(tau_cmd(:) .* dq(:)));
        if P_total > Pmax_eff + 1e-9
            tau_cmd_h  = tau_cmd - tau_G;
            S_task_now = sum(abs(tau_cmd_h(:) .* dq(:)));
            if S_task_now > 0
                sc = max(0, P_avail / S_task_now); sc = min(1, sc);
                tau_cmd = tau_G + tau_cmd_h * sc;
            else
                tau_cmd = tau_G;
            end
        end

        % 推进（ddq 限）
        rhs = (tau_cmd - C*dq - G - diag(max(0,dampNom))*dq);
        rc  = rcond(M);
        if ~isfinite(rc) || rc < 1e-10
            lam = max(1e-6, 1e-3*norm(M,'fro'));
            ddq = (M + lam*eye(3)) \ rhs;
        else
            ddq = M \ rhs;
        end
        ddq = min(max(ddq, -cfg.ddq_abs_max), cfg.ddq_abs_max);

        dq = dq + ddq*cfg.dt;
        q  = q  + dq *cfg.dt;

        % 记录
        q_hist(k,:)  = q.';   dq_hist(k,:) = dq.';  tau_hist(k,:) = tau_cmd.';

        % 命中检测
        [xE,yE] = fk_fun(q);
        if hypot(xE-xT, yE-yT) <= cfg.radius
            t_hit = tvec(k);
            end_speed = norm(J*dq);
            break;
        end
    end

    if ~isfinite(t_hit), t_hit = NaN; end
    info = struct();
    info.t = tvec(1:k);
    info.q_history  = q_hist(1:k,:);
    info.dq_history = dq_hist(1:k,:);
    info.tau_history= tau_hist(1:k,:);
    info.end_speed  = end_speed;
    info.hit        = isfinite(t_hit);
end

% ===================== 工厂模式实现 =====================
function mech = local_build_mech(L, m, g, use_rst, eeName)
    if nargin<5 || isempty(eeName), eeName = 'ee'; end
    if nargin<4 || isempty(use_rst), use_rst = true; end

    % 优先 RST（若无则解析式回退）
    haveRST = use_rst && (exist('rigidBodyTree','class') == 8);
    if ~haveRST
        % 解析式 FK/Jac
        mech.fk  = @(q) local_fk_end_analytic(L, q);
        mech.jac = @(q) local_jac_end_analytic(L, q);
        mech.dyn = @(q,dq) local_dyn_computeDynamics(L, m, g, q, dq);
        return;
    end

    % —— RST 构树（平面 3R）——
    try
        robot = local_make_planar3R_RBT(L, m, g, eeName);
        mech.fk  = @(q) local_fk_end_rst(robot, q, eeName);
        mech.jac = @(q) local_jac_end_rst(robot, q, eeName);
        mech.dyn = @(q,dq) local_dyn_rst(robot, q, dq);
    catch
        % RST 失败 → 回退解析式
        mech.fk  = @(q) local_fk_end_analytic(L, q);
        mech.jac = @(q) local_jac_end_analytic(L, q);
        mech.dyn = @(q,dq) local_dyn_computeDynamics(L, m, g, q, dq);
    end
end

function [L, m, g, use_rst, eeName] = local_parse_factory_inputs(varargin)
    assert(numel(varargin)>=3, '工厂模式至少需要 L,m,g。');
    L = varargin{1}; m = varargin{2}; g = varargin{3};
    use_rst = true; eeName = 'ee';
    if numel(varargin)>=4, use_rst = logical(varargin{4}); end
    if numel(varargin)>=5, eeName  = varargin{5}; end
    L = L(:).'; m = m(:).'; g = double(g);
end

% ===================== RST & 解析式细节 =====================
function robot = local_make_planar3R_RBT(L, m, g, eeName)
    import robotics.*;
    robot = rigidBodyTree('DataFormat','row','MaxNumBodies',3);
    b1 = rigidBody('b1'); j1 = rigidBodyJoint('j1','revolute');
    setFixedTransform(j1,trvec2tform([0 0 0])); j1.JointAxis=[0 0 1]; b1.Joint=j1;
    addBody(robot,b1,'base');

    b2 = rigidBody('b2'); j2 = rigidBodyJoint('j2','revolute');
    setFixedTransform(j2,trvec2tform([L(1) 0 0])); j2.JointAxis=[0 0 1]; b2.Joint=j2;
    addBody(robot,b2,'b1');

    b3 = rigidBody(eeName); j3 = rigidBodyJoint('j3','revolute');
    setFixedTransform(j3,trvec2tform([L(2) 0 0])); j3.JointAxis=[0 0 1]; b3.Joint=j3;
    addBody(robot,b3,'b2');

    robot.Gravity = [0 -g 0];
    % 简质量/转动惯量
    addVisual(robot.Bodies{1}, "Cylinder", [L(1)/40, L(1)], trvec2tform([L(1)/2 0 0]));
    addVisual(robot.Bodies{2}, "Cylinder", [L(2)/40, L(2)], trvec2tform([L(2)/2 0 0]));
    addVisual(robot.Bodies{3}, "Cylinder", [L(3)/40, L(3)], trvec2tform([L(3)/2 0 0]));
end

function [x,y] = local_fk_end_rst(robot, q, eeName)
    T = getTransform(robot, q.', eeName, 'base');
    x = T(1,4); y = T(2,4);
end
function J = local_jac_end_rst(robot, q, eeName)
    J6 = geometricJacobian(robot, q.', eeName);
    J  = J6(1:2,1:3);
end
function [M,C,G] = local_dyn_rst(robot, q, dq)
    M   = massMatrix(robot, q.');
    Cqd = velocityProduct(robot, q.', dq.');
    G   = gravityTorque(robot, q.').';
    C   = local_C_from_Cqd(Cqd, dq);
end

function [x3,y3] = local_fk_end_analytic(L, q)
    q1=q(1); q2=q(2); q3=q(3);
    x1=L(1)*cos(q1);            y1=L(1)*sin(q1);
    x2=x1+L(2)*cos(q1+q2);      y2=y1+L(2)*sin(q1+q2);
    x3=x2+L(3)*cos(q1+q2+q3);   y3=y2+L(3)*sin(q1+q2+q3);
end
function J = local_jac_end_analytic(L, q)
    q1=q(1); q2=q(2); q3=q(3);
    J = [ -L(1)*sin(q1)-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3),  -L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3),  -L(3)*sin(q1+q2+q3);
           L(1)*cos(q1)+L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),   L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),    L(3)*cos(q1+q2+q3) ];
end
function [M,C,G] = local_dyn_computeDynamics(L, m, g, q, dq)
    % 依赖你的工程内 computeDynamics（若不存在，给出保底近似）
    if exist('computeDynamics','file')==2
        [M,C,G] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, q, dq);
        return;
    end
    % 简单保底（不建议长期使用）
    I1=m(1)*L(1)^2/12; I2=m(2)*L(2)^2/12; I3=m(3)*L(3)^2/12;
    M = diag([I1+I2+I3, I2+I3, I3]) + 1e-3*eye(3);
    C = zeros(3); 
    y1=L(1)/2*sin(q(1));
    y2=L(1)*sin(q(1))+L(2)/2*sin(q(1)+q(2));
    y3=L(1)*sin(q(1))+L(2)*sin(q(1)+q(2))+L(3)/2*sin(q(1)+q(2)+q(3));
    G = g*[m(1)*y1 + m(2)*L(1)*sin(q(1));  m(2)*y2;  m(3)*y3];   % 粗略
end
function C = local_C_from_Cqd(Cqd, dq)
    dq = dq(:);
    denom = max( sum(dq.^2), 1e-12 );
    C = (Cqd(:) * dq.') / denom;  % 使 C*dq ≈ Cqd
end

% ===================== 小工具 =====================
function S = local_merge(S, U)
    if nargin<2 || ~isstruct(U), return; end
    f=fieldnames(U); for i=1:numel(f), S.(f{i}) = U.(f{i}); end
end
