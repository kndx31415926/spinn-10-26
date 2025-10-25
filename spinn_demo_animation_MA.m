function spinn_demo_animation_MA(params25_override, opts_override)
% 3R 机械臂（PID + 双顶帽功率限幅）— 单次对比演示
% 说明：
%   - 文件头部“数据块”写清所有物理/控制参数，三种方法请保持完全一致，便于对比。
%   - 目标点/末端轨迹使用“解析 FK”，与 PID 内核一致，避免 RST 回退口径差异。
%   - 命中半径用 opts.radius；命中后统一“截断”所有曲线（轨迹与功率）。
%   - 列义：25 维输入中，第 22 列就是 Pmax（总功率预算）。   <-- 对齐内核与数据链路
%
% 依赖：
%   spinn_mechanical_arm_pid.m  （仿真/功率限幅/日志）    ← 读取 params25(22)=Pmax，并支持 Pmax_gate。               [必需]
%   spinn_MechanicArm.m       （解析动力学 / FK / J）   ← 本演示只用到内核里的解析 FK/J（即使无 RST 也一致）。        [必需]

    %% ===================== ① 数据块（可拷贝到其他方法） =====================
    L = [0.24 0.214 0.324];        % 连杆长度（m）
    m = [3.2, 2.4, 1.6];            % 关节等效质量/转动惯量（示意）

    dq0   = [0 0 0];                % 初始角速度（rad/s）
    damp  = [3 3 3];          % 名义阻尼（轴1/2/3）
    zeta  = [0.9 0.9 0.9];          % 阻尼波动系数（若核内不使用，可忽略）
    q0deg = [0 0 0];                % 初始角（deg）
    qTdeg = [35 40 20];             % 目标角（deg）
    dtheta_deg = qTdeg - q0deg;     % 角度差（deg），保持列义 19..21

    Pmax   = 20;                    % ← 总功率预算（W）
    Prated = [110 110 110];         % 三轴额定功率上限（W）

    % 组装 1x25（与全工程口径一致）—— 1..25 输入，22=Pmax
    params25 = zeros(1,25);
    params25(1:3)    = m;
    params25(4:6)    = dq0;
    params25(7)      = damp(1); params25(8)  = zeta(1);
    params25(9)      = damp(2); params25(10) = zeta(2);
    params25(11)     = damp(3); params25(12) = zeta(3);
    params25(13:15)  = q0deg;
    params25(16:18)  = qTdeg;
    params25(19:21)  = dtheta_deg;
    params25(22)     = Pmax;        % ★ 总功率预算（右下虚线也用它）
    params25(23:25)  = Prated;

    % 允许从外部传入覆盖
    if nargin>=1 && ~isempty(params25_override)
        p = double(params25_override(:).');  assert(numel(p)==25, 'params25 必须 1x25');
        params25 = p;
        % 若调用方改了连杆长度，可在 opts_override.L 里传；本地 L 仅用于绘图/FK
    end

    %% ===================== ② 选项块（对齐 PID 内核） =====================
    opts = struct();
    opts.dt      = 0.002;
    opts.t_final = 10.0;
    opts.radius  = 0.03;            % 命中环半径（m）——三方法对比请保持一致
    opts.g       = 9.81;
    opts.L       = L;               % 仅用于绘图/辅助

    % 任务空间 PD（与内核一致）
    opts.Kp_xy = [80 80];
    opts.Kd_xy = [14 14];

    % 功率分配/限幅（与内核一致）
    opts.limiter_gamma = 0.5;
    opts.tau_smooth_tau = 0.01;     % ≤0 直通；>0 一阶低通
    opts.tau_rate_max  = 600;       % ≤0 关闭扭矩速率限
    opts.ddq_abs_max   = 800;
    opts.w_floor       = 0.12;
    opts.w_cap         = [0.45 0.60 0.60];   % 仅用于日志可视；分轴硬帽仍由 Prated 生效
    opts.fill_power    = true;               % 先填充再限帽（和内核一致）
    opts.fill_alpha    = 0.95;
    opts.p_boost_max   = 8;                  % 如需更贴近 Pmax，可调大（如 20~30）
    opts.Pmax_gate     = [];                 % 关闭动态门控，方便横向对比

    if nargin>=2 && ~isempty(opts_override)
        f = fieldnames(opts_override);
        for i=1:numel(f), opts.(f{i}) = opts_override.(f{i}); end
    end

    %% ===================== ③ 跑仿真（PID 内核） =====================
    % 内核读取 params25(22)=Pmax，且支持 Pmax_gate → 计算有效功率并双顶帽约束
    [t_hit, log] = spinn_mechanical_arm_pid(params25, opts); %#ok<NASGU>
    % 注：该内核日志字段包括 t/q_history/dq_history/tau_history/p_abs/sum_p_abs/reached 等。 
    %     目标点与 FK/J 采用解析式，和这里的一致。   (列义/口径参见工程内核)  :contentReference[oaicite:1]{index=1}

    %% ===================== ④ 统一字段 + 命中截断 =====================
    t  = fget(log,'t');
    Q  = fget(log,{'q','q_history'});
    DQ = fget(log,{'dq','dq_history'});
    P3 = fget(log,{'p','p_abs'});            % 每轴功率（限幅后）
    PS = fget(log,{'sum_p','sum_p_abs'});    % Σ|p|

    k_end = numel(t);
    if isfield(log,'reached') && any(log.reached)
        k_end = min(k_end, find(log.reached,1,'first'));
    elseif isfinite(t_hit)
        k_hit = find(t<=t_hit,1,'last'); if ~isempty(k_hit), k_end = min(k_end, k_hit); end
    end
    t  = t(1:k_end); Q = Q(1:k_end,:); DQ = DQ(1:k_end,:);
    if ~isempty(P3), P3 = P3(1:k_end,:); end
    if ~isempty(PS), PS = PS(1:k_end,:); end

    % 若日志无功率，退化为 |τ⊙dq|
    if isempty(P3) || isempty(PS)
        TAU = fget(log,{'tau','tau_history'}); if isempty(TAU), TAU=zeros(size(Q)); end
        TAU = TAU(1:k_end,:);
        P3  = abs(DQ .* TAU);
        PS  = sum(P3,2);
    end

    %% ===================== ⑤ 目标点 + 末端轨迹（解析 FK，与内核一致） =====================
    qT = deg2rad(params25(16:18));               % 目标角（deg→rad）
    [xT,yT] = fk_end(qT, L);
    XE = zeros(k_end,1); YE = XE;
    for k=1:k_end, [XE(k),YE(k)] = fk_end(Q(k,:).', L); end

    %% ===================== ⑥ 绘图 =====================
    Pmax_show = params25(22);                    % 右下虚线使用第22列 Pmax  :contentReference[oaicite:2]{index=2}
    figure('Color','w','Position',[80 80 1120 560]);

    % 左：末端轨迹
    subplot(1,3,1); hold on; axis equal; grid on;
    plot(XE, YE, 'Color',[0.35 0.7 1.0], 'LineWidth',1.6);
    drawArm(Q(end,:).', L, 'k-', 3.0);
    plot(xT, yT, 'o','MarkerSize',6,'MarkerEdgeColor',[0 0.45 0.74],'LineWidth',1.5);
    th = linspace(0,2*pi,200);
    plot(xT + opts.radius*cos(th), yT + opts.radius*sin(th), ':', 'Color',[.5 .5 .5]);
    title(sprintf('3R 机械臂（PID + 双顶帽功率限幅）  |  t=%.3f s  |  reached=%d', ...
          t(end), any(fget(log,'reached'))));
    xlabel('x / m'); ylabel('y / m');
    pad=0.1; axis([min([XE;xT])-pad, max([XE;xT])+pad, min([YE;yT])-pad, max([YE;yT])+pad]);

    % 右上：三轴功率
    subplot(2,3,2); hold on; grid on;
    plot(t, P3(:,1),'LineWidth',1.2);
    plot(t, P3(:,2),'LineWidth',1.2);
    plot(t, P3(:,3),'LineWidth',1.2);
    legend('P_1','P_2','P_3','Location','northeast');
    title('三轴功率（限幅后）'); xlabel('t / s'); ylabel('|p_i| / W'); xlim([0 t(end)]);

    % 右下：总功率
    subplot(2,3,5); hold on; grid on;
    plot(t, PS,'LineWidth',1.6);
    if isfinite(Pmax_show), yline(Pmax_show,'r--','P_{max}'); end
    title('总功率（限幅后）'); xlabel('t / s'); ylabel('\Sigma|p| / W'); xlim([0 t(end)]);

    % 左图角标信息（有就显示，无就略过）
    w_last  = lastrow(fget(log,{'w','w_history'}), [NaN NaN NaN]);
    sat_tot = lasts(fget(log,'sat_total'), NaN);
    sat_ax  = lastrow(fget(log,'sat_axis'), [NaN NaN NaN]);
    if all(isfinite([sat_tot sat_ax]))
        subplot(1,3,1);
        txt = sprintf('sat_total = %.1f%% | sat_axis = [%.1f %.1f %.1f]%% | w = [%.2f %.2f %.2f] | P_{max} = %.0f W', ...
            100*sat_tot, 100*sat_ax(1),100*sat_ax(2),100*sat_ax(3), w_last(1),w_last(2),w_last(3), Pmax_show);
        text(xT, yT, ['  ' txt], 'Color',[0.2 0.2 0.2], 'FontSize',8, 'Interpreter','none');
    end
end

%% ===================== 辅助：解析 FK / 画臂（与内核一致） =====================
function [x3,y3] = fk_end(q, L)
    q1=q(1); q2=q(2); q3=q(3);
    x1=L(1)*cos(q1);            y1=L(1)*sin(q1);
    x2=x1+L(2)*cos(q1+q2);      y2=y1+L(2)*sin(q1+q2);
    x3=x2+L(3)*cos(q1+q2+q3);   y3=y2+L(3)*sin(q1+q2+q3);
end
function drawArm(q, L, style, lw)
    if nargin<3, style='k-'; end
    if nargin<4, lw=2.0; end
    q1=q(1); q2=q(2); q3=q(3);
    x0=0; y0=0;
    x1=L(1)*cos(q1);            y1=L(1)*sin(q1);
    x2=x1+L(2)*cos(q1+q2);      y2=y1+L(2)*sin(q1+q2);
    x3=x2+L(3)*cos(q1+q2+q3);   y3=y2+L(3)*sin(q1+q2+q3);
    plot([x0 x1 x2 x3],[y0 y1 y2 y3], style, 'LineWidth', lw);
end

%% ===================== 辅助：日志字段抽取/安全取末帧 =====================
function A = fget(S, names)
    if ischar(names) || isstring(names), names={char(names)}; end
    A=[]; for i=1:numel(names)
        if isfield(S, names{i}) && ~isempty(S.(names{i})), A=S.(names{i}); return; end
    end
end
function v = lasts(vAll, def)
    if isempty(vAll), v=def; else, v=vAll(end); end
end
function r = lastrow(M, def)
    if isempty(M), r=def; else, r=M(end,:); end
end
