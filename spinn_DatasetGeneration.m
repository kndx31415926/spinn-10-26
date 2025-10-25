function [w_opt, t_best] = spinn_DatasetGeneration(params25, num_trials, out_path, varargin)
% 单行数据生成（SPINN）
% - 输入 1x25：与采样器定义严格一致（见 spinn_RandomNumberGeneration）；
% - 调用优化器求 w_opt 与 t_best；
% - 以 29 列格式追加到 MAT：[25 | 3 | 1]（同时写 params_matrix 与 params_matrix_spinn）。
%
% 可选名值对（透传给优化器/仿真核）：
%   'objective' : 'time'(默认) 或 'joint'（时间 + λ·末端命中速度）
%   'lambda_v'  : 联合目标的速度权重 λ（传给优化器时键名为 'lambdav'）
%   'cfg'       : 结构体，覆盖仿真口径（例如 dt,t_final,radius,Pmax_gate 等）
%
% 重要索引（与 25 维 schema 一致）：
%   1..3   m1 m2 m3
%   4..6   dq0_1 dq0_2 dq0_3
%   7..12  damp1 zeta1 damp2 zeta2 damp3 zeta3   ← 仅取阻尼：idx=[7,9,11]
%   13..15 init_deg1..3
%   16..18 tgt_deg1..3
%   19..21 dtheta1..3（=tgt-init，仅用于静态模型/校验）
%   22     Pmax
%   23..25 Prated1..3

    % ---------- 默认参数 ----------
    if nargin < 2 || isempty(num_trials), num_trials = 10; end
    if nargin < 3 || isempty(out_path)
        out_path = fullfile('C:', 'Users', 'kndx9', 'Desktop', 'SpinnMechanicalArmParams.mat');  % 与 spinn_main 对齐
    end

    % ---------- 入参检查 ----------
    if ~isvector(params25) || numel(params25) ~= 25
        error('params25 必须是 1x25 向量。');
    end
    p = double(params25(:).');    % 强制行向量/双精度

    % ---------- 可选项解析（对外仍叫 lambda_v） ----------
    opts = struct('objective','time', 'lambda_v',[], 'cfg',[]);
    opts = parse_opts(opts, varargin{:});

    % ---------- 从 25 维读取关键量（索引严谨对齐 schema） ----------
    dq0     = p(4:6);                 % 初始角速度
    dampNom = p([7 9 11]);            % ★ 三轴阻尼：索引 [7,9,11]，避免把 zeta 混进来
    initDeg = p(13:15);               % 初始角(°)
    tgtDeg  = p(16:18);               % 目标角(°)
    Pmax    = p(22);                  % 总功率上限
    Prated  = p(23:25);               % 各轴额定功率（分轴帽上界）

    % 基本一致性校验（宽松容差）
    if any(Prated > Pmax + 1e-9)
        warning('Prated 中存在大于 Pmax 的项：将按优化器中的上界投影修正。');
    end

    % ---------- 组装“仿真核固定部分 25 维”（与 params28 的前 25 对齐） ----------
    % 约定的 25 维布局： [m(3), dq0(3), damp(3), tgt(3), init(3), Pmax, PID(9)]
    PID_Kp = [50 50 50]; PID_Ki = [0.20 0.20 0.20]; PID_Kd = [0.10 0.10 0.10];
    fixed25 = [ p(1:3), dq0, ...
                dampNom, ...
                tgtDeg, ...
                initDeg, ...
                Pmax, ...
                PID_Kp(1), PID_Ki(1), PID_Kd(1), ...
                PID_Kp(2), PID_Ki(2), PID_Kd(2), ...
                PID_Kp(3), PID_Ki(3), PID_Kd(3) ];

    % 保险断言：阻尼切片必须与 p([7,9,11]) 一致
    assert(all(abs(fixed25(7:9) - p([7,9,11])) < 1e-12), '阻尼切片不一致：fixed25(7:9) 应等于 p([7,9,11])。');

    % ---------- 调用优化器（把 lambda_v → lambdav，仅当非空时透传） ----------
    args = {'objective', opts.objective};
    if ~isempty(opts.lambda_v)
        args = [args, {'lambdav', opts.lambda_v}]; %#ok<AGROW>
    end
    if ~isempty(opts.cfg)
        args = [args, {'cfg', opts.cfg}]; %#ok<AGROW>
    end
    [w_opt, t_best] = spinn_optimizemechanicalarm(fixed25, num_trials, Prated, Pmax, args{:});

    % ---------- 29 列：前25输入 + 3权重 + 时间 ----------
    new_row = [p, w_opt, t_best];

    % ---------- 同时写入两个变量名，统一训练/生成口径 ----------
    spinn_append_row_dual(new_row, out_path);
end

% ==========================================================
% 追加写入：创建目录 → 如无文件则新建；
% 如有文件则在 params_matrix 与 params_matrix_spinn 末尾各追加一行
% 优先使用 matfile 逐行写；失败则回退“加载-追加-保存”
% ==========================================================
function spinn_append_row_dual(row_vec, filepath)
    if ~isvector(row_vec), error('row_vec 必须为向量'); end
    row_vec = row_vec(:).';                      % 行向量

    folder = fileparts(filepath);
    if ~exist(folder, 'dir'), mkdir(folder); end

    if isfile(filepath)
        % --- 优先用 matfile 追加，内存友好 ---
        try
            M = matfile(filepath, 'Writable', true);
            write_one(M, 'params_matrix',        row_vec);
            write_one(M, 'params_matrix_spinn',  row_vec);
            return;
        catch ME
            warning('[spinn_append_row_dual] matfile 追加失败：%s → 回退为加载-追加-保存。', ME.message);
            % ----- 回退路径：整读-整写 -----
            S = load(filepath);
            if isfield(S, 'params_matrix')
                PM = S.params_matrix;
            else
                PM = [];
            end
            if isfield(S, 'params_matrix_spinn')
                PMS = S.params_matrix_spinn;
            else
                PMS = [];
            end
            PM  = [PM;  row_vec]; %#ok<AGROW>
            PMS = [PMS; row_vec]; %#ok<AGROW>
            params_matrix = PM; params_matrix_spinn = PMS; %#ok<NASGU>
            save(filepath, 'params_matrix','params_matrix_spinn', '-v7.3');
            return;
        end
    else
        % --- 文件不存在 → 新建并写入第一行（两份同写） ---
        params_matrix = row_vec; %#ok<NASGU>
        params_matrix_spinn = row_vec; %#ok<NASGU>
        save(filepath, 'params_matrix','params_matrix_spinn', '-v7.3');
        return;
    end
end

% ---- 单个变量的逐行追加（matfile）----
function write_one(M, varname, row_vec)
    if any(strcmp(who(M), varname))
        sz = size(M, varname);
        nextRow = max(1, sz(1) + 1);
        M.(varname)(nextRow, 1:numel(row_vec)) = row_vec;
    else
        M.(varname) = row_vec;
    end
end

% ---- 名值对解析（保持对 'lambda_v' 的上游兼容）----
function opts = parse_opts(opts, varargin)
    if isempty(varargin), return; end
    if mod(numel(varargin),2) ~= 0, error('可选参数必须为名-值对。'); end
    for k = 1:2:numel(varargin)
        name = lower(string(varargin{k}));
        val  = varargin{k+1};
        switch name
            case "objective", opts.objective = char(val);
            case "lambda_v",  opts.lambda_v  = double(val);   % 对外仍叫 lambda_v
            case "cfg",       opts.cfg       = val;
            otherwise, warning('未知选项 %s 已忽略。', name);
        end
    end
end
