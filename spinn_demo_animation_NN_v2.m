function spinn_demo_animation_NN_v2(params25_override, opts_override, nn_conf_override)
% 3R 机械臂（NN 动态分配 w + 双顶帽功率限幅，v2）— 单次对比演示
% - 与 PID 版共享同一动力学/任务空间 PD/限幅顺序（fill‑power → 总帽 → 分轴帽）
% - 仅替换“w 的来源”为 v2 动态网络（若模型缺失自动回退 v1/ratio）
% - 头部“数据块”与 MA/NN 脚本一致，便于在相同条件下横向对比

    %% ============= ① 数据块（保持与其它演示一致） =============
    L = [0.24 0.214 0.324];
    m = [3.2, 2.4, 1.6];

    dq0   = [0 0 0];
    damp  = [0.6 0.5 0.4];
    zeta  = [0.9 0.9 0.9];
    q0deg = [0 0 0];
    qTdeg = [35 40 20];
    dtheta_deg = qTdeg - q0deg;

    Pmax   = 20;                   % ← 总功率预算（W）
    Prated = [110 110 110];        % 三轴额定功率（W）

    % 25 维输入（22=Pmax，23..25=Prated）
    params25 = zeros(1,25);
    params25(1:3)    = m;
    params25(4:6)    = dq0;
    params25(7)      = damp(1); params25(8)  = zeta(1);
    params25(9)      = damp(2); params25(10) = zeta(2);
    params25(11)     = damp(3); params25(12) = zeta(3);
    params25(13:15)  = q0deg;
    params25(16:18)  = qTdeg;
    params25(19:21)  = dtheta_deg;
    params25(22)     = Pmax;
    params25(23:25)  = Prated;

    if nargin>=1 && ~isempty(params25_override)
        p = double(params25_override(:).');  assert(numel(p)==25, 'params25 必须 1x25');
        params25 = p;
    end

    %% ============= ② 选项块（与 PID 内核一致） =============
    opts = struct();
    opts.dt      = 0.002;
    opts.t_final = 10.0;
    opts.radius  = 0.03;
    opts.g       = 9.81;
    opts.L       = L;

    % 任务空间 PD
    opts.Kp_xy = [80 80];
    opts.Kd_xy = [14 14];

    % 功率限幅策略（顺序与 PID 版一致）
    opts.limiter_gamma = 0.5;
    opts.tau_smooth_tau = 0.01;
    opts.tau_rate_max  = 600;
    opts.ddq_abs_max   = 800;
    opts.w_floor       = 0.12;
    opts.w_cap         = [0.45 0.60 0.60];
    opts.fill_power    = true;
    opts.fill_alpha    = 0.95;
    opts.p_boost_max   = 8;
    opts.Pmax_gate     = [];       % 对比时通常关闭门控

    if nargin>=2 && ~isempty(opts_override)
        f = fieldnames(opts_override);
        for i=1:numel(f), opts.(f{i}) = opts_override.(f{i}); end
    end

    %% ============= ③ NN 配置（强制 v2；缺模型自动回退） =============
    nn_conf = struct( ...
        'mode','v2', ...                           % v2→若缺退 v1→再缺退 ratio
        'model_v2','trained_model_spinn_v2.mat', ...
        'model_v1','trained_model_spinn.mat', ...
        'hnn_model','spinn_hnn_model.mat', ...
        'w_floor', opts.w_floor);
    if nargin>=3 && ~isempty(nn_conf_override)
        f = fieldnames(nn_conf_override);
        for i=1:numel(f), nn_conf.(f{i}) = nn_conf_override.(f{i}); end
    end

    %% ============= ④ 选择仿真内核（兼容两种文件名） =============
    if exist('spinn_mechanical_arm_nn','file')==2
        sim_fun = @spinn_mechanical_arm_nn;
    elseif exist('spinn_mechanical_armNN','file')==2
        sim_fun = @spinn_mechanical_armNN;   % 你的工程里是这个名字
    else
        error('未找到仿真内核：spinn_mechanical_arm_nn / spinn_mechanical_armNN');
    end

    %% ============= ⑤ 运行仿真（v2 动态 w） =============
    [t_hit, log] = sim_fun(params25, opts, nn_conf); %#ok<NASGU>

    %% ============= ⑥ 提取有效帧，构轨迹与功率 =============
    t  = fget(log,'t');
    Q  = fget(log,{'q','q_history'});
    DQ = fget(log,{'dq','dq_history'});
    P3 = fget(log,{'p','p_abs'});
    PS = fget(log,{'sum_p','sum_p_abs'});

    k_end = numel(t);
    if isfield(log,'reached') && any(log.reached)
        k_end = min(k_end, find(log.reached,1,'first'));
    elseif exist('t_hit','var') && isfinite(t_hit)
        kh = find(t<=t_hit,1,'last'); if ~isempty(kh), k_end = min(k_end, kh); end
    end
    t  = t(1:k_end); Q = Q(1:k_end,:); DQ = DQ(1:k_end,:);
    if ~isempty(P3), P3 = P3(1:k_end,:); end
    if ~isempty(PS), PS = PS(1:k_end,:); end

    % 若日志没功率，退化为 |τ⊙dq|
    if isempty(P3) || isempty(PS)
        TAU = fget(log,{'tau','tau_history'}); if isempty(TAU), TAU=zeros(size(Q)); end
        TAU = TAU(1:k_end,:);
        P3  = abs(DQ .* TAU);
        PS  = sum(P3,2);
    end

    %% ============= ⑦ 目标点 + 末端轨迹（解析 FK，与内核一致） =============
    qT = deg2rad(params25(16:18));
    [xT,yT] = fk_end(qT, L);
    XE = zeros(k_end,1); YE = XE;
    for k=1:k_end, [XE(k),YE(k)] = fk_end(Q(k,:).', L); end

    %% ============= ⑧ 可视化（与 MA 版统一） =============
    Pmax_show = params25(22);
    figure('Color','w','Position',[80 80 1120 560]);

    % 左：末端轨迹
    subplot(1,3,1); hold on; axis equal; grid on;
    plot(XE, YE, 'Color',[0.35 0.7 1.0], 'LineWidth',1.6);
    drawArm(Q(end,:).', L, 'k-', 3.0);
    plot(xT, yT, 'o','MarkerSize',6,'MarkerEdgeColor',[0 0.45 0.74],'LineWidth',1.5);
    th = linspace(0,2*pi,200);
    plot(xT + opts.radius*cos(th), yT + opts.radius*sin(th), ':', 'Color',[.5 .5 .5]);
    title(sprintf('3R 机械臂（NN v2 + 双顶帽）  |  t=%.3f s  |  reached=%d', ...
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

    % 角标（若字段缺省则自动跳过）
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

% ==== 小工具 ====
function A = fget(S, names)
    if ischar(names) || isstring(names), names={char(names)}; end
    A=[]; for i=1:numel(names)
        if isfield(S, names{i}) && ~isempty(S.(names{i})), A=S.(names{i}); return; end
    end
end
function v = lasts(vAll, def), if isempty(vAll), v=def; else, v=vAll(end); end, end
function r = lastrow(M, def),  if isempty(M), r=def; else, r=M(end,:); end, end

% ==== 解析 FK / 画臂（与内核一致）====
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
