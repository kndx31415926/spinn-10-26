function [robot, eeName] = spinn_robot_rst(L, m, g)
% L: 1x3 链长（你的 v2 里默认 L=[0.24 0.214 0.324]）
% m: 1x3 质量（来自 params25）
% g: 标量（9.81），重力沿 -Y
    arguments
        L (1,3) double {mustBePositive}
        m (1,3) double {mustBeNonnegative}
        g (1,1) double {mustBePositive} = 9.81
    end
    robot = rigidBodyTree('DataFormat','row','MaxNumBodies',4);
    robot.Gravity = [0 -g 0];  % 你当前模型 y 轴竖直向上

    % —— link1 ——
    body1 = rigidBody('link1');
    j1 = rigidBodyJoint('joint1','revolute'); j1.JointAxis = [0 0 1];
    setFixedTransform(j1, eye(4));
    body1.Joint = j1;
    body1.Mass = m(1); body1.CenterOfMass = [L(1)/2 0 0];
    Izz = m(1)*L(1)^2/12;  % 平面转动主要用 Izz
    body1.Inertia = [1e-9, Izz, Izz, 0, 0, 0];
    addBody(robot, body1, 'base');

    % —— link2 ——
    body2 = rigidBody('link2');
    j2 = rigidBodyJoint('joint2','revolute'); j2.JointAxis = [0 0 1];
    setFixedTransform(j2, trvec2tform([L(1) 0 0]));  % 从 link1 末端出发
    body2.Joint = j2;
    body2.Mass = m(2); body2.CenterOfMass = [L(2)/2 0 0];
    Izz = m(2)*L(2)^2/12;
    body2.Inertia = [1e-9, Izz, Izz, 0, 0, 0];
    addBody(robot, body2, 'link1');

    % —— link3 ——
    body3 = rigidBody('link3');
    j3 = rigidBodyJoint('joint3','revolute'); j3.JointAxis = [0 0 1];
    setFixedTransform(j3, trvec2tform([L(2) 0 0]));
    body3.Joint = j3;
    body3.Mass = m(3); body3.CenterOfMass = [L(3)/2 0 0];
    Izz = m(3)*L(3)^2/12;
    body3.Inertia = [1e-9, Izz, Izz, 0, 0, 0];
    addBody(robot, body3, 'link2');

    % —— 末端工具坐标（质量为 0 的固定体）——
    tool = rigidBody('ee');
    fix = rigidBodyJoint('fixee','fixed');
    setFixedTransform(fix, trvec2tform([L(3) 0 0]));
    tool.Joint = fix;
    addBody(robot, tool, 'link3');

    eeName = 'ee';
end
