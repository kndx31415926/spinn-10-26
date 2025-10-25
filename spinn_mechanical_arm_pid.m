function [t_hit, log] = spinn_mechanical_arm_pid(params25, opts)
    assert(isvector(params25) && numel(params25)==25);
    p25 = double(params25(:).');

    % ---- 默认参数（新增 fill-power / omega_eps）----
    def = struct( ...
        'dt',0.002, 't_final',6.0, 'radius',0.03, ...
        'Kp_xy',[80 80], 'Kd_xy',[14 14], ...
        'limiter_gamma',0.5, ...
        'tau_smooth_tau',0.01, 'tau_rate_max',600, ...  % <=0/Inf 关
        'ddq_abs_max',600, ...
        'w_floor',0.12, 'w_cap',[0.45 0.60 0.60], ...
        'Pmax_gate',[], ...
        'fill_power', true, 'fill_alpha', 0.95, 'p_boost_max', 8.0, ... % ★ 新增
        'omega_eps', 1e-3, ...                                         % ★ 新增
        'kick_on_stall', true, 'kick_steps', 6, ...
        'debug', false, 'force_analytic_kin', true);
    if nargin<2 || isempty(opts), opts=struct(); end
    opts = local_merge(def, opts);

    % ---- 解包 25 维（与采样/训练完全一致）----
    m        = p25(1:3);
    dq       = p25(4:6).';
    dampNom  = p25([7 9 11]);                % 阻尼三轴 7/9/11  :contentReference[oaicite:2]{index=2}
    q0deg    = p25(13:15);
    qTdeg    = p25(16:18);
    Pmax     = p25(22);
    Prated   = p25(23:25);

    % ---- 力学核（动力学仍走统一入口；FK/J 用解析避免回退异常）----
    L=[0.24 0.214 0.324]; g=9.81;
    mech   = spinn_MechanicArm(L, m, g, true, 'ee');  dyn_fun = mech.dyn;

    q  = deg2rad(q0deg(:));
    qT = deg2rad(qTdeg(:));
    [xT,yT] = local_fk_end(qT, L);
    tvec = (0:opts.dt:opts.t_final).';  K=numel(tvec);

    qh=nan(K,3); dqh=nan(K,3); tauh=nan(K,3);
    pabs=nan(K,3); psum=nan(K,1); wh=nan(K,3); dEEv=nan(K,1); rech=false(K,1);
    qh(1,:)=q.'; dqh(1,:)=dq.'; [xE0,yE0]=local_fk_end(q,L); dEEv(1)=hypot(xE0-xT,yE0-yT); rech(1)=dEEv(1)<=opts.radius;

    tau_h_prev=zeros(3,1); tau_h_lp=zeros(3,1);
    maxSigP=0; t_hit=NaN; k=1;

    for k=2:K
        [M,C,G]=dyn_fun(q,dq);
        [xE,yE]=local_fk_end(q,L); J=local_jacobian_planar(q,L);
        vE=J*dq; e=[xT-xE; yT-yE];
        F=[opts.Kp_xy(1)*e(1)-opts.Kd_xy(1)*vE(1);
           opts.Kp_xy(2)*e(2)-opts.Kd_xy(2)*vE(2)];
        tau_task=J.'*F;

        % —— 启发式 w（投影到带地板的单纯形）——
        p_des = abs(tau_task(:).*dq(:));
        s=sum(p_des); if s>0, w=(p_des.'/s); else, w=[1 1 1]/3; end
        w = project_to_simplex_floor(w, opts.w_floor);

        % —— 可用功率（扣掉重力功率）——
        Pmax_eff = Pmax;
        if isa(opts.Pmax_gate,'function_handle')
            d_now = hypot(e(1), e(2));
            Pmax_eff = max(0, min(Pmax, opts.Pmax_gate(d_now, Pmax, k, tvec(k))));
        end
        P_G = sum(abs(G(:).*dq(:)));
        P_avail = max(0, Pmax_eff - P_G);

        % ===================================================
        % ★ 先 fill‑power（把 τ_task 拉到 α·P_avail，上限 p_boost_max）
        % ===================================================
        tau_h = tau_task;
        if opts.fill_power && P_avail>0
            omega = dq;                                % 3x1
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

        % —— 再“总帽”（Σ|τ⊙dq| ≤ P_avail）——
        S_now = sum(abs(tau_h(:).*dq(:)));
        if S_now > (P_avail + 1e-12) && S_now > 0
            tau_h = tau_h * (P_avail / S_now);
        end

        % —— 分轴软帽（cap_i = min(w_i*P_avail, Prated_i)）——
        cap_i = min(w(:)*P_avail, Prated(:));
        p_ax  = abs(tau_h(:).*dq(:));
        gamma = max(0.2, min(1.0, opts.limiter_gamma));
        for ii=1:3
            if p_ax(ii) > (cap_i(ii) + 1e-12)
                r = cap_i(ii) / max(p_ax(ii), eps);
                tau_h(ii) = tau_h(ii) * r^gamma;
            end
        end

        % —— 速率限（仅 >0 时启用）——
        if isfinite(opts.tau_rate_max) && (opts.tau_rate_max>0)
            dr = opts.tau_rate_max * opts.dt;
            tau_h = min(max(tau_h, tau_h_prev - dr), tau_h_prev + dr);
        end
        tau_h_prev = tau_h;

        % —— 低通（<=0 则直通）——
        if isfinite(opts.tau_smooth_tau) && (opts.tau_smooth_tau>0)
            alpha = 1 - exp(-opts.dt/opts.tau_smooth_tau);
        else
            alpha = 1;
        end
        tau_h_lp = tau_h_lp + alpha*(tau_h - tau_h_lp);

        % —— 合成总扭矩 + 末端校核（只缩任务扭矩）——
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

        % —— 日志功率/权重 —— 
        pabs(k,:) = abs((tau_cmd(:).*dq(:))).';
        psum(k,1) = sum(pabs(k,:));
        wh(k,:)   = w; 
        maxSigP   = max(maxSigP, psum(k,1));

        % —— 推进（含阻尼 D，ddq 限）——
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
        if rech(k), t_hit=tvec(k); end
    end

    % —— 输出 log（与动画口径一致）——
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

% === 与内核一致的解析 FK/J ===
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
    if nargin<2 || ~isstruct(U), return; end; f=fieldnames(U); for i=1:numel(f), O.(f{i})=U.(f{i}); end
end
