function [t_hit, log] = spinn_mechanical_arm_nn(params25, opts, nn_conf)
% 3R 机械臂（任务空间 PD + SPINN 双顶帽功率限幅），w 由 NN 给出
% - 与 spinn_mechanical_arm_pid 的控制与限幅顺序完全一致；
% - 区别仅在 w 的来源：v2(动态)/v1(静态)/ratio/equal（自动回退）。
%
% 列义：params25(22)=Pmax, (23:25)=Prated（与采样/训练一致）

    assert(isvector(params25) && numel(params25)==25, 'params25 必须 1x25');
    p25 = double(params25(:).');

    % ---------- 默认 opts（与 PID 内核一致） ----------
    def = struct( ...
        'dt',0.002, 't_final',6.0, 'radius',0.03, ...
        'Kp_xy',[80 80], 'Kd_xy',[14 14], ...
        'limiter_gamma',0.5, ...
        'tau_smooth_tau',0.01, 'tau_rate_max',600, ...
        'ddq_abs_max',600, ...
        'w_floor',0.12, 'w_cap',[0.45 0.60 0.60], ...
        'Pmax_gate',[], ...
        'fill_power', true, 'fill_alpha', 0.95, 'p_boost_max', 8.0, ...
        'omega_eps', 1e-3, ...
        'debug', false, 'force_analytic_kin', true);
    if nargin<2 || isempty(opts), opts=struct(); end
    opts = local_merge(def, opts);

    % ---------- NN 配置 ----------
    defNN = struct( ...
        'mode','auto', ...                 % 'auto'|'v2'|'v1'|'ratio'|'equal'
        'model_v2','trained_model_spinn_v2.mat', ...
        'model_v1','trained_model_spinn.mat', ...
        'hnn_model','spinn_hnn_model.mat', ...
        'prefer_v2', true, ...
        'w_floor', opts.w_floor);
    if nargin<3 || isempty(nn_conf), nn_conf = struct(); end
    nn_conf = local_merge(defNN, nn_conf);

    % ---------- 解包 25 维 ----------
    m        = p25(1:3);
    dq       = p25(4:6).';
    dampNom  = p25([7 9 11]);
    q0deg    = p25(13:15);
    qTdeg    = p25(16:18);
    Pmax     = p25(22);
    Prated   = p25(23:25);

    % ---------- 力学核（与 PID 内核一致） ----------
    L=[0.24 0.214 0.324]; g=9.81;
    mech   = spinn_MechanicArm(L, m, g, true, 'ee');  dyn_fun = mech.dyn;

    q  = deg2rad(q0deg(:));
    qT = deg2rad(qTdeg(:));
    [xT,yT] = local_fk_end(qT, L);
    tvec = (0:opts.dt:opts.t_final).';  K=numel(tvec);

    % ---------- 准备 NN 的 w_fun ----------
    w_fun = make_w_fun(nn_conf, p25, Pmax, Prated, L, g);

    % ---------- 预分配日志 ----------
    qh=nan(K,3); dqh=nan(K,3); tauh=nan(K,3);
    pabs=nan(K,3); psum=nan(K,1); wh=nan(K,3); dEEv=nan(K,1); rech=false(K,1);
    qh(1,:)=q.'; dqh(1,:)=dq.'; [xE0,yE0]=local_fk_end(q,L); dEEv(1)=hypot(xE0-xT,yE0-yT); rech(1)=dEEv(1)<=opts.radius;

    tau_h_prev=zeros(3,1); tau_h_lp=zeros(3,1);
    t_hit=NaN;

    for k=2:K
        [M,C,G]=dyn_fun(q,dq);
        [xE,yE]=local_fk_end(q,L); J=local_jacobian_planar(q,L);
        vE=J*dq; e=[xT-xE; yT-yE];

        % 任务空间 PD（同 PID 内核）
        F=[opts.Kp_xy(1)*e(1)-opts.Kd_xy(1)*vE(1);
           opts.Kp_xy(2)*e(2)-opts.Kd_xy(2)*vE(2)];
        tau_task=J.'*F;

        % ------------------ w：来自 NN（或回退） ------------------
        w = w_fun(q, dq, tvec(k), tau_task, e, J);
        w = project_to_simplex_floor(w, nn_conf.w_floor);

        % 可用功率
        Pmax_eff = Pmax;
        if isa(opts.Pmax_gate,'function_handle')
            d_now = hypot(e(1), e(2));
            Pmax_eff = max(0, min(Pmax, opts.Pmax_gate(d_now, Pmax, k, tvec(k))));
        end
        P_G = sum(abs(G(:).*dq(:)));
        P_avail = max(0, Pmax_eff - P_G);

        % ====== 顺序与 PID 版一致：fill‑power → 总帽 → 分轴帽 ======
        tau_h = tau_task;

        % 1) fill‑power
        if opts.fill_power && P_avail>0
            omega = dq;
            omega_nz = omega;
            mask = ~isfinite(omega_nz) | (abs(omega_nz) < opts.omega_eps);
            sgn  = sign(tau_h); sgn(sgn==0) = 1;
            omega_nz(mask) = opts.omega_eps .* sgn(mask);
            S_est = sum(abs(tau_h(:).*omega_nz(:)));
            if S_est > 0 && S_est < opts.fill_alpha * P_avail
                scale = min(opts.p_boost_max, (opts.fill_alpha * P_avail) / S_est);
                tau_h = tau_h * scale;
            end
        end

        % 2) 总帽
        S_now = sum(abs(tau_h(:).*dq(:)));
        if S_now > (P_avail + 1e-12) && S_now > 0
            tau_h = tau_h * (P_avail / S_now);
        end

        % 3) 分轴软帽
        cap_i = min(w(:)*P_avail, Prated(:));
        p_ax  = abs(tau_h(:).*dq(:));
        gamma = max(0.2, min(1.0, opts.limiter_gamma));
        for ii=1:3
            if p_ax(ii) > (cap_i(ii) + 1e-12)
                r = cap_i(ii) / max(p_ax(ii), eps);
                tau_h(ii) = tau_h(ii) * r^gamma;
            end
        end

        % 速率限
        if isfinite(opts.tau_rate_max) && (opts.tau_rate_max>0)
            dr = opts.tau_rate_max * opts.dt;
            tau_h = min(max(tau_h, tau_h_prev - dr), tau_h_prev + dr);
        end
        tau_h_prev = tau_h;

        % 低通
        if isfinite(opts.tau_smooth_tau) && (opts.tau_smooth_tau>0)
            alpha = 1 - exp(-opts.dt/opts.tau_smooth_tau);
        else
            alpha = 1;
        end
        tau_h_lp = tau_h_lp + alpha*(tau_h - tau_h_lp);

        % 合成总扭矩 + 末端校核（只缩任务扭矩）
        tau_cmd = tau_h_lp + G;
        P_total = sum(abs(tau_cmd(:).*dq(:)));
        if P_total > Pmax_eff + 1e-9
            S_task = sum(abs(tau_h_lp(:).*dq(:)));
            if S_task > 0
                sc = max(0, (P_avail / S_task)); sc = min(1, sc);
                tau_h_lp = tau_h_lp * sc;
                tau_cmd  = G + tau_h_lp;
            else
                tau_cmd = G;
            end
        end

        % 日志
        pabs(k,:) = abs((tau_cmd(:).*dq(:))).';
        psum(k,1) = sum(pabs(k,:));
        wh(k,:)   = w; 

        % 动力学推进（含阻尼）
        D  = diag(max(0, dampNom(:)));
        rhs= (tau_cmd - C*dq - G - D*dq);
        rc = rcond(M);
        if ~isfinite(rc) || rc < 1e-10
            lam = max(1e-6, 1e-3*norm(M,'fro'));
            ddq = (M + lam*eye(3)) \ rhs;
        else
            ddq = M \ rhs;
        end
        ddq = min(max(ddq, -opts.ddq_abs_max), opts.ddq_abs_max);

        dq = dq + ddq*opts.dt;
        q  = q  + dq *opts.dt;

        % 保存轨迹/命中
        qh(k,:)=q.'; dqh(k,:)=dq.'; tauh(k,:)=tau_cmd.';        
        dEEv(k,1)=hypot(xE-xT,yE-yT); rech(k,1)=dEEv(k,1)<=opts.radius;
        if rech(k) && ~isfinite(t_hit), t_hit=tvec(k); end
    end

    % ---------- 输出日志（与 PID 版字段对齐） ----------
    last_idx = max(1, min(k, K));
    log=struct();
    log.t=tvec(1:last_idx);
    log.q_history=qh(1:last_idx,:); log.dq_history=dqh(1:last_idx,:);
    log.tau_history=tauh(1:last_idx,:); log.p_abs=pabs(1:last_idx,:);
    log.sum_p_abs=psum(1:last_idx,1); log.w_history=wh(1:last_idx,:);
    log.dEE=dEEv(1:last_idx,1); log.reached=rech(1:last_idx,1);
    log.q=log.q_history; log.dq=log.dq_history; log.tau=log.tau_history;
    log.p=log.p_abs; log.sum_p=log.sum_p_abs; log.w = wh(last_idx,:);
    log.Pmax=Pmax; log.Prated=Prated(:).';
end

% ======================== NN 接入与特征 ========================
function w_fun = make_w_fun(nn_conf, p25, Pmax, Prated, L, g)
    mode = lower(string(nn_conf.mode));
    if mode=="auto"
        hasV2 = isfile(nn_conf.model_v2) && exist('predict','file')==2;
        hasV1 = isfile(nn_conf.model_v1) && exist('predict','file')==2;
        if hasV2, mode="v2"; elseif hasV1, mode="v1"; else, mode="ratio"; end
    end

    switch mode
        case "v2"
            S = load_if_exist(nn_conf.model_v2);
            need = {'trainedNet','muX','sigmaX'};
            if ~all(isfield(S,need)), warning('[NN] v2 模型不完整，回退 ratio。'); w_fun=@ratio_w; return; end
            v2.trainedNet = S.trainedNet; v2.muX=S.muX(:).'; v2.sigmaX=S.sigmaX(:).';
            H = load_if_exist(nn_conf.hnn_model);
            if isfield(H,'model') && isstruct(H.model) && isfield(H.model,'dlnet')
                v2.hnn.dlnet = H.model.dlnet; v2.hnn.muX = H.model.muX; v2.hnn.sigmaX = H.model.sigmaX;
            elseif isfield(H,'dlnet') && isfield(H,'muX') && isfield(H,'sigmaX')
                v2.hnn.dlnet = H.dlnet;       v2.hnn.muX = H.muX;       v2.hnn.sigmaX = H.sigmaX;
            else
                warning('[NN] 找不到 HNN 或字段不全，v2 将退化为 ratio。');
                v2.hnn = [];
            end
            w_fun = @(q,dq,t,tau_task,e,J) w_v2_now(q,dq,p25,Pmax,Prated,L,g,v2,nn_conf.w_floor);
        case "v1"
            T = load_if_exist(nn_conf.model_v1);
            need = {'trainedNet','muX','sigmaX','muY','sigmaY'};
            if ~all(isfield(T,need)), warning('[NN] v1 模型不完整，回退 ratio。'); w_fun=@ratio_w; return; end
            v1=T; v1.muX=v1.muX(:).'; v1.sigmaX=v1.sigmaX(:).'; v1.muY=v1.muY(:).'; v1.sigmaY=v1.sigmaY(:).';
            w_fun = @(q,dq,~,tau_task,~,~) w_v1_now(q,dq,p25,Pmax,Prated,v1,nn_conf.w_floor);
        case "equal"
            w_fun = @(varargin) [1 1 1]/3;
        otherwise % 'ratio'
            w_fun = @ratio_w;
    end

    function w = ratio_w(q,dq,~,tau_task,~,~)
        omega = dq; mask = ~isfinite(omega) | (abs(omega) < 1e-3);
        sg = sign(tau_task); sg(sg==0)=1;
        omega(mask) = 1e-3 * sg(mask);
        p_des = abs(tau_task(:) .* omega(:));
        s=sum(p_des); if s>0, w=(p_des.'/s); else, w=[1 1 1]/3; end
        w = project_to_simplex_floor(w, 0.12);
    end
end

function w = w_v1_now(q_now,dq_now,p25,Pmax,Prated,T,w_floor)
    % v1：单步静态 w；无 DL 工具箱/预测失败时自动回退
    try
        init_deg_now = rad2deg(q_now(:)).';
        qT_deg = p25(16:18);
        dth_deg_now = qT_deg - init_deg_now;
        x = [ p25(1:3), dq_now(:).', p25(7:12), init_deg_now, qT_deg, dth_deg_now, Pmax, Prated ];
        xz = (x - T.muX) ./ T.sigmaX;
        yp = predict(T.trainedNet, xz);
        yp = yp .* T.sigmaY + T.muY;
        w  = project_to_simplex_floor(yp(1:3), w_floor);
        if ~all(isfinite(w)), w = project_to_simplex_floor([1 1 1]/3, w_floor); end
    catch
        w = project_to_simplex_floor([1 1 1]/3, w_floor);
    end
end

function w = w_v2_now(q,dq,p25,Pmax,Prated,L,g,V2,w_floor)
    if isempty(V2.hnn) || ~(exist('dlarray','file')==2)
        w = project_to_simplex_floor([1 1 1]/3, w_floor); return;
    end
    F10 = build_features_v2_safe(q, dq, deg2rad(p25(16:18)), p25(1:3), L, g, V2.hnn.dlnet, V2.hnn.muX, V2.hnn.sigmaX);
    x = [p25, F10];
    xz = (x - V2.muX) ./ V2.sigmaX;
    try
        y  = predict(V2.trainedNet, xz);
        w  = project_to_simplex_floor(y(1:3), w_floor);
        if ~all(isfinite(w)), w = project_to_simplex_floor([1 1 1]/3, w_floor); end
    catch
        w = project_to_simplex_floor([1 1 1]/3, w_floor);
    end
end

function F10 = build_features_v2_safe(q, dq, qT, m, L, g, dlnetH, muH, sigH)
    % F10 = [ dEE, M11 M22 M33, condM, H0, Hg, DeltaH, ||dHdp||, ||dHdq|| ]
    [Mq,~,~] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, q, dq);
    p = Mq * dq;
    [xE,yE] = local_fk_end(q,  L);
    [xT,yT] = local_fk_end(qT, L);
    dEE = hypot(xE-xT, yE-yT);

    rc = rcond(Mq); 
    if ~isfinite(rc) || rc<=0, condM=1e15; else, condM = max(1, 1/rc); end

    qv = dlarray(single(q(:)),'CB'); pv = dlarray(single(p(:)),'CB');
    muHv = dlarray(single(muH(:)),'CB'); sigHv = dlarray(single(sigH(:)),'CB');
    [H0dl,dHdq0dl,dHdp0dl] = dlfeval(@hnn_grad_fun, dlnetH, qv, pv, muHv, sigHv);
    [Hgdl,~,~] = dlfeval(@hnn_grad_fun, dlnetH, ...
                         dlarray(single(qT(:)),'CB'), dlarray(single(zeros(3,1)),'CB'), ...
                         muHv, sigHv);
    H0 = double(gather(extractdata(H0dl)));
    Hg = double(gather(extractdata(Hgdl)));
    dHdp_norm = double(gather(norm(extractdata(dHdp0dl))));
    dHdq_norm = double(gather(norm(extractdata(dHdq0dl))));
    DeltaH = H0 - Hg;

    F10 = [dEE, diag(Mq).', condM, H0, Hg, DeltaH, dHdp_norm, dHdq_norm];
end

function [H,dHdq,dHdp] = hnn_grad_fun(dlnet, qv, pv, muHv, sigHv)
    z  = ([qv; pv] - muHv) ./ sigHv;
    H  = forward(dlnet, z);
    Hs = sum(H,'all');
    [dHdq,dHdp] = dlgradient(Hs, qv, pv);
end

function S = load_if_exist(path)
    if isfile(path), S = load(path);
    else, S = struct(); end
end

% ======================== 几何/工具（与 PID 内核一致） ========================
function [x3,y3]=local_fk_end(q,L)
    q1=q(1);q2=q(2);q3=q(3);
    x1=L(1)*cos(q1);y1=L(1)*sin(q1);
    x2=x1+L(2)*cos(q1+q2);y2=y1+L(2)*sin(q1+q2);
    x3=x2+L(3)*cos(q1+q2+q3);y3=y2+L(3)*sin(q1+q2+q3);
end

function J=local_jacobian_planar(q,L)
    q1=q(1);q2=q(2);q3=q(3);
    J=[-L(1)*sin(q1)-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3), -L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3), -L(3)*sin(q1+q2+q3);
        L(1)*cos(q1)+L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),  L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),   L(3)*cos(q1+q2+q3)];
end

function w=project_to_simplex_floor(w,floor_val)
    w=double(w(:).'); floor_val=max(0,min(1/3-1e-6,floor_val)); free=1-3*floor_val;
    w=max(w,0); s=sum(w); if s==0, w=ones(size(w))/numel(w); else, w=w/s; end
    w=max(w-floor_val,0); s2=sum(w); if s2>0, w=w/s2*free; end; w=w+floor_val;
end

function O=local_merge(O,U)
    if nargin<2 || ~isstruct(U), return; end
    f=fieldnames(U); for i=1:numel(f), O.(f{i})=U.(f{i}); end
end
