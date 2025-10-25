function [M, C, G] = computeDynamics_rst(robot, q, dq)
% q,dq: 3x1
    if isrow(q), q=q.'; end
    if isrow(dq), dq=dq.'; end

    % RST 提供：
    M   = massMatrix(robot, q.');           % 3x3
    Cqd = velocityProduct(robot, q.', dq.');% 3x1, 即 C(q,dq)*dq
    G   = gravityTorque(robot, q.').';      % 3x1

    % 兼容你现有代码里 "C*dq" 的用法：构造一个等效 C，使 C*dq = Cqd
    denom = max(1e-12, dq.'*dq);
    C = (Cqd * dq.') / denom;  % rank-1 矩阵，足够满足 C*dq 的数值等价
end
