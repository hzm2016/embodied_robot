#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class HANSKinematics {
public:
    HANSKinematics();

    void GetKinematics(Eigen::Matrix4d& kinematics, Eigen::Matrix<double,6,1>& joints);
    void GetJacobian(Eigen::Matrix<double,6,6>& jacobian, Eigen::Matrix<double,6,1>& joints);
    void GetNextJoints(Eigen::Matrix<double,6,1>& next_joints,
                       Eigen::Matrix<double,6,1>& cur_joints,
                       Eigen::Matrix4d& next_kinematics);

private:
    // 由 URDF 提取的固定变换
    Eigen::Matrix4d T_fixed_[7];
    // 关节限制
    Eigen::Matrix<double,6,1> q_lower_, q_upper_;
    // 数值差分步长
    double diff_h_;
    // 基础阻尼
    double dls_lambda_;

    // 实现细节
    Eigen::Matrix4d makeTransform(double x, double y, double z,
                                  double rr, double rp, double ry) const;
    Eigen::Matrix3d rpy2rot(double r, double p, double y) const;
    Eigen::Matrix4d rotZ(double q) const;
    void forwardKinematics(Eigen::Matrix4d& T_0e, const Eigen::Matrix<double,6,1>& q) const;
};

inline HANSKinematics::HANSKinematics() {
    // 旧版参数占位（不用于本实现）
    // 仅保留命名，避免外部依赖编译告警（若不存在可删除）
    // float lsb=0, lse=0, lew=0, lwt=0;

    const double PI = 3.14159265358979323846;

    // 从 URDF 提取的固定变换（parent->child，不含关节角，axis 为 Z）
    T_fixed_[0] = makeTransform(0, 0, 0.1475, 0, 0, 0);
    T_fixed_[1] = makeTransform(0, -0.1985, 0, PI/2, 0, 0);
    T_fixed_[2] = makeTransform(-0.607, 0, -0.138, 0, 0, PI);
    T_fixed_[3] = makeTransform(0.566, 0, 0, 0, 0, 0);
    T_fixed_[4] = makeTransform(0, 0, 0.127, -PI/2, 0, 0);
    T_fixed_[5] = makeTransform(0, 0, 0.127, PI/2, 0, 0);
    T_fixed_[6] = makeTransform(0, 0, 0.1155, 0, 0, PI);
    // 如果需要 elfin_dummy_gripper，改为：
    // T_fixed_[6] = makeTransform(0, -0.0935, 0, PI/2, PI/2, 0);

    // 关节限位（URDF）
    q_lower_ << -6.28, -3.3158, -2.7574, -6.28, -6.28, -6.28;
    q_upper_ <<  6.28,  0.1745,  2.7574,  6.28,  6.28,  6.28;

    // 数值差分步长与 DLS 阻尼
    diff_h_     = 1e-6;
    dls_lambda_ = 1e-5;
}

inline Eigen::Matrix4d HANSKinematics::makeTransform(double x, double y, double z,
                                            double rr, double rp, double ry) const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = rpy2rot(rr, rp, ry);
    T(0,3) = x; T(1,3) = y; T(2,3) = z;
    return T;
}

inline Eigen::Matrix3d HANSKinematics::rpy2rot(double r, double p, double y) const {
    // URDF rpy 顺序：先 Rx(r) 再 Ry(p) 再 Rz(y) => R = Rz(y)*Ry(p)*Rx(r)
    double cr = std::cos(r), sr = std::sin(r);
    double cp = std::cos(p), sp = std::sin(p);
    double cy = std::cos(y), sy = std::sin(y);

    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0,
          0, cr, -sr,
          0, sr,  cr;
    Ry <<  cp, 0, sp,
           0,  1, 0,
          -sp, 0, cp;
    Rz <<  cy, -sy, 0,
           sy,  cy, 0,
           0,    0, 1;
    return Rz * Ry * Rx;
}

inline Eigen::Matrix4d HANSKinematics::rotZ(double q) const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    double c = std::cos(q), s = std::sin(q);
    T(0,0) = c;  T(0,1) = -s;
    T(1,0) = s;  T(1,1) =  c;
    return T;
}

inline void HANSKinematics::forwardKinematics(Eigen::Matrix4d& T_0e, const Eigen::Matrix<double,6,1>& q) const {
    T_0e.setIdentity();
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix4d Ti = T_fixed_[i] * rotZ(q(i)); // axis=(0,0,1)
        T_0e = T_0e * Ti;
    }
    // 末端固定
    T_0e = T_0e * T_fixed_[6];
}

inline void HANSKinematics::GetKinematics(Eigen::Matrix4d& kinematics, Eigen::Matrix<double,6,1>& joints) {
    forwardKinematics(kinematics, joints);
}

inline void HANSKinematics::GetJacobian(Eigen::Matrix<double,6,6>& jacobian, Eigen::Matrix<double,6,1>& joints) {
    // 数值差分雅可比
    Eigen::Matrix4d T;
    forwardKinematics(T, joints);

    Eigen::Vector3d p = T.block<3,1>(0,3);
    Eigen::Matrix3d R = T.block<3,3>(0,0);

    jacobian.setZero();
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double,6,1> qh = joints;
        qh(i) += diff_h_;

        Eigen::Matrix4d Th;
        forwardKinematics(Th, qh);

        Eigen::Vector3d ph = Th.block<3,1>(0,3);
        Eigen::Matrix3d Rh = Th.block<3,3>(0,0);

        // 位置差分
        Eigen::Vector3d dp = (ph - p) / diff_h_;

        // 姿态差分：Rerr = R * Rh^T，取小角度近似
        Eigen::Matrix3d Rerr = R * Rh.transpose();
        Eigen::AngleAxisd aa(Rerr);
        // 取负号近似 dR/dq 方向
        Eigen::Vector3d domega = (-aa.axis() * aa.angle()) / diff_h_;

        jacobian.block<3,1>(0,i) = dp;
        jacobian.block<3,1>(3,i) = domega;
    }
}

inline void HANSKinematics::GetNextJoints(Eigen::Matrix<double,6,1>& next_joints,
                                 Eigen::Matrix<double,6,1>& cur_joints,
                                 Eigen::Matrix4d& next_kinematics) {
    // 当前正解
    Eigen::Matrix4d T;
    forwardKinematics(T, cur_joints);

    Eigen::Vector3d p  = T.block<3,1>(0,3);
    Eigen::Matrix3d R  = T.block<3,3>(0,0);
    Eigen::Vector3d pd = next_kinematics.block<3,1>(0,3);
    Eigen::Matrix3d Rd = next_kinematics.block<3,3>(0,0);

    // 位姿误差（将当前转向目标）
    Eigen::Matrix3d Rerr = R * Rd.transpose();
    Eigen::AngleAxisd aa(Rerr);
    Eigen::Vector3d e_rot = -aa.axis() * aa.angle();
    Eigen::Vector3d e_pos = pd - p;

    Eigen::Matrix<double,6,1> e;
    e.block<3,1>(0,0) = e_pos;
    e.block<3,1>(3,0) = e_rot;

    // 雅可比
    Eigen::Matrix<double,6,6> J;
    GetJacobian(J, cur_joints);

    // 自适应 DLS 阻尼：sigma 越小，lambda 越大
    Eigen::JacobiSVD<Eigen::Matrix<double,6,6>> svd(J); // 仅计算奇异值
    Eigen::VectorXd s = svd.singularValues();
    double sigma_min = s.minCoeff();

    const double lambda0   = dls_lambda_; // 基础阻尼
    const double sigma_ref = 0.05;        // 参考奇异值（与主程序一致）
    const double eps       = 1e-6;
    double lambda_adapt = lambda0 * std::pow( sigma_ref / std::max(sigma_min, eps), 2.0 );

    // DLS 伪逆：J^T (J J^T + λ I)^-1
    Eigen::Matrix<double,6,6> JJt = J * J.transpose();
    Eigen::Matrix<double,6,6> A   = JJt + lambda_adapt * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,6> pinv = J.transpose() * A.inverse();

    Eigen::Matrix<double,6,1> dq = pinv * e;

    next_joints = cur_joints + dq;

    // 关节限位裁剪
    for (int i = 0; i < 6; ++i) {
        if (next_joints(i) < q_lower_(i)) next_joints(i) = q_lower_(i);
        if (next_joints(i) > q_upper_(i)) next_joints(i) = q_upper_(i);
    }
}