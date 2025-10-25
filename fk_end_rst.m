function [xE, yE] = fk_end_rst(robot, q, eeName)
    if isrow(q), q=q.'; end
    T = getTransform(robot, q.', eeName, 'base');
    xE = T(1,4); yE = T(2,4);
end

