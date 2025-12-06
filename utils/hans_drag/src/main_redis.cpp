#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <atomic>
#include <string>
#include <cstdlib>

#include <unistd.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <hiredis/hiredis.h>

#include <force_haptron.hpp>
#include "HansRobot.hpp"
#include "HansKinematics.hpp"

static inline double deg2rad(double d){ return d * M_PI / 180.0; }
static inline double rad2deg(double r){ return r * 180.0 / M_PI; }

// ========= Redis Pub/Sub 配置与订阅 =========
struct RedisConnCfg {
    std::string host = "127.0.0.1";
    int port = 6379;
    int db = 0;
};

static RedisConnCfg parseRedisUrl() {
    RedisConnCfg cfg;
    const char* url = std::getenv("REDIS_URL");
    if (!url || !*url) return cfg;
    std::string u(url);
    const std::string prefix = "redis://";
    if (u.rfind(prefix, 0) == 0) {
        std::string rest = u.substr(prefix.size());
        auto slash = rest.find('/');
        std::string hostport = (slash == std::string::npos) ? rest : rest.substr(0, slash);
        if (slash != std::string::npos) {
            std::string dbs = rest.substr(slash + 1);
            try { cfg.db = std::stoi(dbs); } catch (...) {}
        }
        auto colon = hostport.find(':');
        if (colon != std::string::npos) {
            cfg.host = hostport.substr(0, colon);
            try { cfg.port = std::stoi(hostport.substr(colon + 1)); } catch (...) {}
        } else {
            cfg.host = hostport;
        }
    }
    return cfg;
}

static std::string g_drag_channel = []{
    const char* env = std::getenv("ARM_DRAG_CHANNEL");
    return env && *env ? std::string(env) : std::string("device:arm:drag");
}();

// 拖拽模式原子开关，由订阅线程更新
static std::atomic<bool> g_drag_mode{false};

// 极简 JSON 解析 drag_flag（为了避免额外依赖 JSON 库）
static std::string extract_drag_flag_from_json(const std::string& payload) {
    // 查找 "drag_flag":"..."
    const std::string key = "\"drag_flag\"";
    auto pos = payload.find(key);
    if (pos == std::string::npos) return {};
    auto colon = payload.find(':', pos + key.size());
    if (colon == std::string::npos) return {};
    auto first = payload.find_first_of("\"'", colon + 1);
    if (first == std::string::npos) return {};
    auto second = payload.find_first_of("\"'", first + 1);
    if (second == std::string::npos || second <= first) return {};
    return payload.substr(first + 1, second - first - 1);
}

static void RedisDragSubscriberThread()
{
    RedisConnCfg cfg = parseRedisUrl();

    for (;;) {
        redisContext* c = redisConnect(cfg.host.c_str(), cfg.port);
        if (!c || c->err) {
            std::cerr << "[redis] connect failed: " << (c ? c->errstr : "null") << ", retry in 1s\n";
            if (c) redisFree(c);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (cfg.db > 0) {
            redisReply* r = (redisReply*)redisCommand(c, "SELECT %d", cfg.db);
            if (!r || c->err) {
                std::cerr << "[redis] SELECT failed: " << (c ? c->errstr : "null") << "\n";
                if (r) freeReplyObject(r);
                redisFree(c);
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            freeReplyObject(r);
        }

        {
            redisReply* r = (redisReply*)redisCommand(c, "SUBSCRIBE %s", g_drag_channel.c_str());
            if (!r || c->err) {
                std::cerr << "[redis] SUBSCRIBE failed: " << (c ? c->errstr : "null") << "\n";
                if (r) freeReplyObject(r);
                redisFree(c);
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            freeReplyObject(r);
            std::cout << "[redis] subscribed: " << g_drag_channel << std::endl;
        }

        for (;;) {
            redisReply* reply = nullptr;
            if (redisGetReply(c, (void**)&reply) != REDIS_OK) {
                std::cerr << "[redis] get reply error: " << (c ? c->errstr : "unknown") << "\n";
                break; // 断线重连
            }
            if (!reply) continue;

            if (reply->type == REDIS_REPLY_ARRAY && reply->elements == 3) {
                auto kind = reply->element[0];
                auto ch   = reply->element[1];
                auto body = reply->element[2];
                if (kind && kind->type == REDIS_REPLY_STRING &&
                    std::string(kind->str) == "message" &&
                    ch && ch->type == REDIS_REPLY_STRING &&
                    body && body->type == REDIS_REPLY_STRING)
                {
                    std::string payload(body->str, body->len);
                    // 解析 drag_flag
                    std::string flag = extract_drag_flag_from_json(payload);
                    if (!flag.empty()) {
                        if (flag == "right") {
                            g_drag_mode.store(true);
                            std::cout << "[drag] drag_mode = true\n";
                        } else if (flag == "left") {
                            g_drag_mode.store(false);
                            std::cout << "[drag] drag_mode = false\n";
                        } else {
                            // 其他 flag 可按需使用（如方向）
                            std::cout << "[drag] flag=" << flag << "\n";
                        }
                    }
                }
            }

            freeReplyObject(reply);
        }

        if (c) redisFree(c);
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 重连退避
    }
}

// ========= 原有机器人拖拽逻辑 =========
float g_sensor[6] = {0}; 

// 传感器线程：串口读取六维力/矩
void ForceSensor()
{
    float sensor_force_offset[3] = {0};
    float sensor_torque_offset[3] = {0};
    int iniCount = 0;

    CLinuxSerial com(0, 115200);
    while (true)
    {
        if (com.InitPort(0, 115200))
        {
            unsigned char data[] = {0x01, 0x04, 0x00, 0x38, 0x00, 0x0C, 0x71, 0xC2};
            UINT result = com.WriteData(data, sizeof(data));
            if (result > 0)
            {
                unsigned char buf[31];
                com.ReadData(buf, 31);
                if ((buf[0] == 0x01) && (buf[1] == 0x04) && (buf[2] == 0x18))
                {
                    unsigned char fx_tmp[4] = {buf[6], buf[5], buf[4], buf[3]};
                    unsigned char fy_tmp[4] = {buf[10], buf[9], buf[8], buf[7]};
                    unsigned char fz_tmp[4] = {buf[14], buf[13], buf[12], buf[11]};
                    unsigned char tx_tmp[4] = {buf[18], buf[17], buf[16], buf[15]};
                    unsigned char ty_tmp[4] = {buf[22], buf[21], buf[20], buf[19]};
                    unsigned char tz_tmp[4] = {buf[26], buf[25], buf[24], buf[23]};
                    float fx = *((float *)(fx_tmp));
                    float fy = *((float *)(fy_tmp));
                    float fz = *((float *)(fz_tmp));
                    float tx = *((float *)(tx_tmp));
                    float ty = *((float *)(ty_tmp));
                    float tz = *((float *)(tz_tmp));

                    constexpr int INIT_NUM = 50;
                    if (iniCount < INIT_NUM)
                    {
                        // 可做零漂统计
                        iniCount++;
                    }
                    else
                    {
                        float s[6];
                        s[0] = fx - sensor_force_offset[0];
                        s[1] = fy - sensor_force_offset[1];
                        s[2] = fz - sensor_force_offset[2];
                        s[3] = tx - sensor_torque_offset[0];
                        s[4] = ty - sensor_torque_offset[1];
                        s[5] = tz - sensor_torque_offset[2];
                        for (int i=0;i<6;i++)
                            g_sensor[i] = (std::fabs(s[i]) > 0.0f) ? s[i] : 0.0f;
                    }
                }
            }
        }
        // 可按需 sleep
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

class RobotDrag
{
public:
    int kalman_filter_time = 0;
    float pos_m=7.0f, pos_d=30.0f, ori_m=0.1f, ori_d=15.0f, delta_t=0.012f;

    Eigen::Matrix<double, 6, 1> real_ft, gravity_error, tool_center, real_ft_last, pk_last;

    // 启动零偏（可选）
    Eigen::Matrix<double, 6, 1> real_ft_start_bias; // 偏置值
    std::atomic<bool> did_start_zero{false};         // 是否已采样完成
    std::atomic<bool> enable_start_bias{false};      // 是否启用偏置扣除

    Eigen::Matrix3d cur_rot, next_rot;
    Eigen::Vector3d cur_pos, next_pos, cur_velo, next_velo, delta_pos;
    Eigen::Vector3d cur_ori_velo, next_ori_velo, delta_theta;

    // 奇异性接近时的速度缩放（由主循环更新）
    double slow_scale = 1.0;

    void updateFTzero();
    void calRealForce();
    void PosCtr();
    void OriCtr();
    void AdjustOri();

    // 可选偏置功能接口
    void setStartBiasEnabled(bool on) { enable_start_bias.store(on); }
    bool isStartBiasEnabled() const { return enable_start_bias.load(); }
    bool hasStartBias() const { return did_start_zero.load(); }
    Eigen::Matrix<double,6,1> getStartBias() const { return real_ft_start_bias; }
    void clearStartBias() {
        real_ft_start_bias.setZero();
        did_start_zero.store(false);
    }

    template<typename RobotCmd, typename Kine>
    void doStartBiasSampling(RobotCmd& cmd, Kine& kine,
                             Eigen::Matrix<double,6,1>& q_rad,
                             Eigen::Matrix4d& T,
                             int samples, int period_ms = 12)
    {
        if (samples <= 0) {
            clearStartBias();
            return;
        }
        Eigen::Matrix<double,6,1> acc; acc.setZero();

        for (int i = 0; i < samples; ++i) {
            auto j_deg_tmp = cmd.readCurrentAct();
            for (int k = 0; k < 6; ++k) q_rad(k) = deg2rad(j_deg_tmp[k]);
            kine.GetKinematics(T, q_rad);
            cur_rot = T.block<3,3>(0,0);
            cur_pos = T.block<3,1>(0,3);

            calRealForce();
            acc += real_ft;

            std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
        }
        real_ft_start_bias = acc / static_cast<double>(samples);
        did_start_zero.store(true);
    }
};

void RobotDrag::updateFTzero()
{
    Eigen::Vector3d sensor_f(g_sensor[0], g_sensor[1], g_sensor[2]);
    Eigen::Vector3d sensor_t(g_sensor[3], g_sensor[4], g_sensor[5]);

    double f_max = std::max({std::fabs(sensor_f[0]), std::fabs(sensor_f[1]), std::fabs(sensor_f[2])});
    double t_max = std::max({std::fabs(sensor_t[0]), std::fabs(sensor_t[1]), std::fabs(sensor_t[2])});
    if (f_max > 100 || t_max > 10) {
        for(int i=0;i<3;i++)
        {
            if (sensor_f[i]>20) sensor_f[i]=20;
            if (sensor_f[i]<-20) sensor_f[i]=-20;
        }
        std::cout << "F/T out of range\n";
        std::cout<<sensor_f[0]<<","<<sensor_f[1]<<","<<sensor_f[2]<<std::endl;
        std::cout<<sensor_t[0]<<","<<sensor_t[1]<<","<<sensor_t[2]<<std::endl;
        std::cout << "F/T out of range\n";
        std::exit(0);
    }

    Eigen::Vector3d gravity_sensor = cur_rot.transpose() * gravity_error.head<3>();
    Eigen::Vector3d moment_gravity(
        gravity_sensor(2) * tool_center(1) - gravity_sensor(1) * tool_center(2),
        gravity_sensor(0) * tool_center(2) - gravity_sensor(2) * tool_center(0),
        gravity_sensor(1) * tool_center(0) - gravity_sensor(0) * tool_center(1));
    Eigen::Vector3d zero_f = sensor_f - gravity_sensor;
    Eigen::Vector3d zero_t = sensor_t - moment_gravity;
    Eigen::Vector3d zero_t_para(
        zero_t(0) - zero_f(2) * tool_center(1) + zero_f(1) * tool_center(2),
        zero_t(1) - zero_f(0) * tool_center(2) + zero_f(2) * tool_center(0),
        zero_t(2) - zero_f(1) * tool_center(0) + zero_f(0) * tool_center(1));
    gravity_error << gravity_error.head<3>(), zero_f;
    tool_center   << tool_center.head<3>(),   zero_t_para;
}

void RobotDrag::calRealForce()
{
    Eigen::Vector3d F_s(g_sensor[0], g_sensor[1], g_sensor[2]);
    Eigen::Vector3d T_s(g_sensor[3], g_sensor[4], g_sensor[5]);

    double th = (M_PI / 3.0 - M_PI / 12.0) + M_PI; // 240度
    Eigen::Matrix3d R_tool_sensor;
    R_tool_sensor <<
        std::cos(th), -std::sin(th), 0,
        std::sin(th),  std::cos(th), 0,
        0,             0,            1;

    Eigen::Vector3d sensor_f = R_tool_sensor * F_s;
    Eigen::Vector3d sensor_t = R_tool_sensor * T_s;

    Eigen::Vector3d gravity_sensor = cur_rot.transpose() * gravity_error.head<3>();
    Eigen::Vector3d moment_gravity(
        gravity_sensor(2) * tool_center(1) - gravity_sensor(1) * tool_center(2),
        gravity_sensor(0) * tool_center(2) - gravity_sensor(2) * tool_center(0),
        gravity_sensor(1) * tool_center(0) - gravity_sensor(0) * tool_center(1));
    Eigen::Vector3d moment_tool(
        tool_center(3) - gravity_error(4) * tool_center(2) + gravity_error(5) * tool_center(1),
        tool_center(4) - gravity_error(5) * tool_center(0) + gravity_error(3) * tool_center(2),
        tool_center(5) - gravity_error(3) * tool_center(1) + gravity_error(4) * tool_center(0));

    Eigen::Vector3d real_f = sensor_f - (gravity_sensor + gravity_error.tail<3>());
    Eigen::Vector3d real_t = sensor_t - (moment_gravity + moment_tool);

    real_ft << real_f, real_t;

    // 指数滤波
    float Rk = 0.0002f, Bk = 0.000001f;
    if (kalman_filter_time == 0) { pk_last << Bk,Bk,Bk,Bk,Bk,Bk; real_ft_last = real_ft; }
    for (int i=0;i<6;i++) {
        double xk_bar = real_ft_last[i];
        double pk_bar = pk_last[i] + Bk;
        real_ft[i] = xk_bar + (pk_bar / (pk_bar + Rk)) * (real_ft[i] - xk_bar);
        pk_last[i]  = pk_bar - (pk_bar / (pk_bar + Rk)) * pk_bar;
        real_ft_last[i] = real_ft[i];
    }
    kalman_filter_time++;

    if (enable_start_bias.load() && did_start_zero.load()) {
        real_ft -= real_ft_start_bias;
    }
}

void RobotDrag::PosCtr()
{
    float acc[3];
    for (int i=0;i<3;i++){
        if (std::fabs(real_ft[i]) > 0.0){
            acc[i] = (real_ft[i] - pos_d * cur_velo[i]) / pos_m;
            next_velo[i] = cur_velo[i] + acc[i] * delta_t;

            double v  = static_cast<double>(next_velo[i]);
            double lo = -1.0 * slow_scale;
            double hi =  1.0 * slow_scale;
            v = std::clamp(v, lo, hi);
            next_velo[i] = static_cast<float>(v);

            delta_pos[i] = (cur_velo[i] * delta_t + 0.5f * acc[i] * delta_t * delta_t);
        } else {
            next_velo[i] = 0; delta_pos[i] = 0;
        }
        if (std::fabs(delta_pos[i]) > 0.3) { std::cout << "Pos too fast!\n"; std::exit(0); }
    }
    cur_velo = next_velo;
    next_pos = cur_pos + cur_rot * delta_pos;
}

void RobotDrag::OriCtr()
{
    float acc[3];
    for (int i=0;i<3;i++){
        if (std::fabs(real_ft[i+3]) > 0.0){
            acc[i] = (real_ft[i+3] - ori_d * cur_ori_velo[i]) / ori_m;
            next_ori_velo[i] = cur_ori_velo[i] + acc[i] * delta_t;

            double w  = static_cast<double>(next_ori_velo[i]);
            double lo = -1.0 * slow_scale;
            double hi =  1.0 * slow_scale;
            w = std::clamp(w, lo, hi);
            next_ori_velo[i] = static_cast<float>(w);

            delta_theta[i]   = cur_ori_velo[i] * delta_t + 0.5f * acc[i] * delta_t * delta_t;
        } else {
            next_ori_velo[i] = 0; delta_theta[i] = 0;
        }
        if (std::fabs(delta_theta[i]) > 0.2) { std::cout << "Ori too fast!\n"; std::exit(0); }
    }
    cur_ori_velo = next_ori_velo;
}

void RobotDrag::AdjustOri()
{
    Eigen::Matrix3d Rx, Ry, Rz;
    double ax = 20.0 * delta_theta[0];
    double ay = 20.0 * delta_theta[1];
    double az = 30.0 * delta_theta[2];
    Rx << 1,0,0, 0,std::cos(ax),-std::sin(ax), 0,std::sin(ax),std::cos(ax);
    Ry << std::cos(ay),0,std::sin(ay), 0,1,0, -std::sin(ay),0,std::cos(ay);
    Rz << std::cos(az),-std::sin(az),0, std::sin(az),std::cos(az),0, 0,0,1;
    next_rot = cur_rot * Ry * Rx * Rz;
}

int main()
{
    try {
        // 传感器线程
        std::thread th_sensor(ForceSensor);
        // Redis 订阅线程：接收 drag_flag 并更新 g_drag_mode
        std::thread th_redis(RedisDragSubscriberThread);

        sleep(1);

        DCSCommand cmd("192.168.156.2", 10003, 0, 0);
        HANSKinematics kine;
        RobotDrag drag;

        // 配置：是否启用“上电零偏采样与扣除”和采样次数
        const bool enable_start_bias = true;
        const int  start_bias_samples = 100;
        drag.setStartBiasEnabled(enable_start_bias);

        // 加载重力补偿参数
        std::ifstream gravity_file("../data/gc_parameters.txt");
        if (!gravity_file) { std::cout << "Can't load GC parameters.\n"; return 0; }
        for (int i = 0; i < 12; i++) {
            if (i < 6) gravity_file >> drag.gravity_error(i, 0);
            else       gravity_file >> drag.tool_center(i - 6, 0);
        }

        // 连接与速度比
        cmd.connectTCPSocket();
        cmd.setSpeedRadio(0.2);

        // 初始位姿
        std::array<double,6> start_j_deg = {90,-140,80,60,90,0};
        cmd.moveJ(start_j_deg, /*vel*/10, /*acc*/20, /*radius*/0.0);
        while (!cmd.isCompleteMovement()) std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 初始状态
        Eigen::Matrix<double,6,1> q_rad, q_next;
        Eigen::Matrix4d T;
        auto j_deg = cmd.readCurrentAct();
        for (int i=0;i<6;i++) q_rad(i) = deg2rad(j_deg[i]);

        kine.GetKinematics(T, q_rad);
        drag.cur_rot = T.block<3,3>(0,0);
        drag.cur_pos = T.block<3,1>(0,3);
        drag.next_rot = drag.cur_rot;
        drag.next_pos = drag.cur_pos;
        drag.cur_velo.setZero();
        drag.cur_ori_velo.setZero();
        drag.next_velo.setZero();
        drag.next_ori_velo.setZero();
        drag.delta_pos.setZero();
        drag.delta_theta.setZero();
        drag.real_ft.setZero();
        drag.real_ft_last.setZero();
        drag.pk_last.setZero();
        drag.real_ft_start_bias.setZero();
        drag.clearStartBias();

        // 零偏采样（可选）
        if (drag.isStartBiasEnabled()) {
            std::cout << "[StartBias] Sampling " << start_bias_samples << " frames..." << std::endl;
            drag.doStartBiasSampling(cmd, kine, q_rad, T, start_bias_samples, /*period_ms*/12);
            std::cout << "[StartBias] Done. Bias = " << drag.getStartBias().transpose() << std::endl;

            drag.cur_velo.setZero();
            drag.next_velo.setZero();
            drag.cur_ori_velo.setZero();
            drag.next_ori_velo.setZero();
            drag.delta_pos.setZero();
            drag.delta_theta.setZero();
        } else {
            std::cout << "[StartBias] Disabled." << std::endl;
        }

        auto next_tick = std::chrono::steady_clock::now();

        // 奇异性阈值
        const double sigma_hard_stop = 0.06; // 硬停
        const double sigma_slow      = 0.08; // 软区

        // Servo 参数
        const double servo_time = 0.012; // 12ms
        const double lookahead  = 0.2;
        bool start_flag;

        while (true)
        {
            start_flag = true;
            std::cout<<"-----------NO DRAG-------------"<<std::endl;
            while(g_drag_mode.load())
            {   
                if (start_flag)
                {std::cout<<"-----------start servo-------------"<<std::endl;
                    cmd.startServo(servo_time, lookahead);
                    next_tick = std::chrono::steady_clock::now();
                    start_flag = false;
                }

                next_tick += std::chrono::milliseconds(int(servo_time*1000));

                // 读取当前状态
                j_deg = cmd.readCurrentAct();
                for (int i=0;i<6;i++) q_rad(i) = deg2rad(j_deg[i]);
                kine.GetKinematics(T, q_rad);
                drag.cur_rot = T.block<3,3>(0,0);
                drag.cur_pos = T.block<3,1>(0,3);

                // // 奇异性检测
                // Eigen::Matrix<double,6,6> J;
                // kine.GetJacobian(J, q_rad);
                // Eigen::JacobiSVD<Eigen::Matrix<double,6,6>> svd(J);
                // double sigma_min = svd.singularValues().minCoeff();

                // auto clamp01 = [](double x){ return std::min(1.0, std::max(0.0, x)); };
                // drag.slow_scale = 1.0;
                // if (sigma_min < sigma_slow) {
                //     drag.slow_scale = clamp01( (sigma_min - sigma_hard_stop) / (sigma_slow - sigma_hard_stop) );
                //     drag.slow_scale = std::max(0.2, drag.slow_scale); // 最低 20%
                // }

                // 传感器处理
                drag.calRealForce();
                std::cout<<"F:::::"<<drag.real_ft.transpose()<<std::endl;
                if (std::fabs(drag.real_ft[2]) > 50.0) {
                    std::cout << "Error! Z-axis force beyond sensor range.\n";
                    break;
                }

                // 顺应控制
                drag.PosCtr();
                drag.OriCtr();
                drag.AdjustOri();

                // 目标位姿 -> 一步 IK
                Eigen::Matrix4d T_next = Eigen::Matrix4d::Identity();

                // if(g_drag_mode.load())
                // {
                     std::cout<<"-----------IN DRAG-------------"<<std::endl;
                    T_next.block<3,3>(0,0) = drag.next_rot;
                    T_next.block<3,1>(0,3) = drag.next_pos;
                // }
                // else
                // {
                //     std::cout<<"-----------NO DRAG-------------"<<std::endl;
                //     T_next.block<3,3>(0,0) = drag.cur_rot;
                //     T_next.block<3,1>(0,3) = drag.cur_pos;
                // }

                Eigen::Matrix<double,6,1> q_next_local;
                kine.GetNextJoints(q_next_local, q_rad, T_next);

                bool deadzone =
                    std::fabs(drag.real_ft[0]) < 3.5 &&
                    std::fabs(drag.real_ft[1]) < 3.5 &&
                    std::fabs(drag.real_ft[2]) < 3.5 &&
                    std::fabs(drag.real_ft[3]) < 0.5 &&
                    std::fabs(drag.real_ft[4]) < 0.5 &&
                    std::fabs(drag.real_ft[5]) < 0.5;

                std::array<double,6> q_cmd_deg{};
                if (deadzone) {
                    for (int i=0;i<6;i++) q_cmd_deg[i] = rad2deg(q_rad(i));
                } else {
                    for (int i=0;i<6;i++) q_cmd_deg[i] = rad2deg(q_next_local(i));
                }

                std::cout<<"cur_pos "<<drag.cur_pos.transpose()<<std::endl;
                std::cout<<"T_next "<<drag.next_pos.transpose()<<std::endl;
                std::cout<<"cur_joint "<<q_rad.transpose()<<std::endl;
                std::cout<<"next_joint "<<q_next_local.transpose()<<std::endl;


                // 下发伺服关节位置（度）
                cmd.pushServoJ(q_cmd_deg);
                // 睡到下一个周期
                std::this_thread::sleep_until(next_tick);
            }
            
        }

        if (th_sensor.joinable()) th_sensor.detach();
        if (th_redis.joinable()) th_redis.detach();
        cmd.closeTCPSocket();
    }
    catch (const HansError& he) {
        std::cerr << "HansError: " << he.what() << "\n";
        return he.code();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return -1;
    }
    return 0;
}