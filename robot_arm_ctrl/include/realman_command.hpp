#pragma once

#include <memory>
#include <string>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <json.hpp>
#include <sys/shm.h>
#include <sys/types.h>
#include <Eigen/Dense>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

class RMCommand
{
public:
    // 通讯
    int rlm_port, rlm_socket, recv_times;
    char rlm_ip[16], send_msg[1000], recv_msg[1000];
    std::string cmd_str;
    nlohmann::json command_msg, return_msg;

    // 关节缓存
    // Realman 机器人在某些情况下可能不返回或只返回 "\n"（也可解析为 json），
    // 为避免读到错误关节，保留最近一次读取。
    Eigen::Matrix<double, 6, 1> cmd_joints;      // 兼容：6轴缓存（旧接口沿用）
    Eigen::Matrix<double, 7, 1> cmd_joints7;     // 新增：7轴缓存（新接口使用）

public:
    RMCommand();

    // 基础通讯
    void ConnectTCPSocket();
    void SetHighSpeedEth();

    // 7轴原生接口（推荐在 7 轴机器人上使用）
    void ReadJ7(Eigen::Matrix<double,7,1>& joints7);
    void MoveJ7(const Eigen::Matrix<double,7,1>& joints7, int velo);
    void ServoJ7(const Eigen::Matrix<double,7,1>& joints7, bool follow);

    // 6轴兼容接口（保持与你现有上层一致，内部做 6↔7 的映射）
    void ReadJ(Eigen::Matrix<double,6,1>& joints);
    void MoveJ(Eigen::Matrix<double,6,1>& joints, int velo);
    void ServoJ(Eigen::Matrix<double,6,1>& joints, bool follow);

    // 笛卡尔接口维持不变（6 自由度位姿）
    void MoveL(const Eigen::Matrix<double,6,1>& pose, int velo);
    void MoveJP(const Eigen::Matrix<double,6,1>& pose, int velo);
};