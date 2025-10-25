function J2 = jacobian_planar_rst(robot, q, eeName)
    if isrow(q), q=q.'; end
    J = geometricJacobian(robot, q.', eeName);  % 6xN: [ang; lin]
    Jlin = J(4:6,:);    % 线速度
    J2   = Jlin(1:2,:); % 取 x,y 两行 → 2x3
end
