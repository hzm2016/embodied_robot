#include <chrono>
#include <time.h>
#include <thread>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>  

#include <sw/redis++/redis++.h>  
#include <iostream>
#include <nlohmann/json.hpp>   
#include <force_haptron.hpp>
#include <realman_command.hpp>  
#include <realman_kinematics.hpp>  

#include <mutex>

using json = nlohmann::json;  
using namespace sw::redis;     

float sensor[6] = {0};  
std::mutex g_mutex;
Eigen::Matrix<double, 7, 1> cur_joints, next_joints;   

// F/T measurement
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
                unsigned char buf[31] = {0};
                com.ReadData(buf, 31);

                if ((buf[0] == 0x01) && (buf[1] == 0x04) && (buf[2] == 0x18))
                {
                    unsigned char fx_tmp[4] = {buf[6],  buf[5],  buf[4],  buf[3]};
                    unsigned char fy_tmp[4] = {buf[10], buf[9],  buf[8],  buf[7]};
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
                        // 累计偏移用于零点
                        sensor_force_offset[0] += fx / INIT_NUM;
                        sensor_force_offset[1] += fy / INIT_NUM;
                        sensor_force_offset[2] += fz / INIT_NUM;
                        sensor_torque_offset[0] += tx / INIT_NUM;
                        sensor_torque_offset[1] += ty / INIT_NUM;
                        sensor_torque_offset[2] += tz / INIT_NUM;
                        iniCount++;
                    }
                    else
                    {
                        std::vector<float> sensor_pressure(6, 0); // 力和力矩
                        sensor_pressure[0] = fx - sensor_force_offset[0];
                        sensor_pressure[1] = fy - sensor_force_offset[1];
                        sensor_pressure[2] = fz - sensor_force_offset[2];

                        // 修正：力矩应减去 torque 偏移，而不是 force 偏移
                        sensor_pressure[3] = tx - sensor_torque_offset[0];
                        sensor_pressure[4] = ty - sensor_torque_offset[1];
                        sensor_pressure[5] = tz - sensor_torque_offset[2];

                        std::cout << std::fixed << std::setprecision(4);
                        // 阈值滤波
                        for (int i = 0; i < 6; ++i)
                        {
                            if (i < 3)
                                sensor[i] = (std::fabs(sensor_pressure[i]) > 1.0f) ? sensor_pressure[i] : 0.0f;
                            else
                                sensor[i] = (std::fabs(sensor_pressure[i]) > 0.1f) ? sensor_pressure[i] : 0.0f;
                        }
                    }
                }
            }
        }
        // 可考虑添加适度休眠避免占满CPU
        // usleep(1000);
    }
}

void PrintJ(Eigen::Matrix<double, 6, 1> &joints)
{
    std::cout << joints.transpose() * 180.0 / M_PI << std::endl;
}

class RobotDrag
{
public:
    // Calculate real F/T and kalman filter parameters
    int kalman_filter_time;
    Eigen::Matrix<double, 6, 1> real_ft, gravity_error, tool_center, real_ft_last, pk_last;

    // Force control parameters
    float pos_m, pos_d, ori_m, ori_d, delta_t;   
    Eigen::Matrix3d cur_rot, next_rot;   
    Eigen::Vector3d cur_pos, next_pos, cur_velo, next_velo, delta_pos;   
    Eigen::Vector3d cur_ori_velo, next_ori_velo, delta_theta;   

public:
    RobotDrag();
    void updateFTzero();
    void calRealForce();
    void PosCtr();
    void OriCtr();
    void AdjustOri();
};

RobotDrag::RobotDrag()
{
    // Force control parameters
    pos_m = 0.01f;
    pos_d = 1.0f;
    ori_m = 0.03f;
    ori_d = 1.0f;
    cur_velo.setZero();
    next_velo.setZero();
    delta_pos.setZero();
    cur_ori_velo.setZero();
    next_ori_velo.setZero();
    delta_theta.setZero();
    cur_rot.setIdentity();
    next_rot.setIdentity();
    cur_pos.setZero();
    next_pos.setZero();

    // Kalman Filter parameters
    kalman_filter_time = 0;
    real_ft.setZero();
    real_ft_last.setZero();
    pk_last.setZero();

    // Admittance control time
    delta_t = 0.01f;
}

void RobotDrag::updateFTzero()  
{  
    Eigen::Vector3d sensor_f, sensor_t, zero_f, zero_t, gravity_sensor, moment_gravity;
    sensor_f << sensor[0], sensor[1], sensor[2];
    sensor_t << sensor[3], sensor[4], sensor[5];

    // Check if F/T is out of sensor range
    double f_max = std::max({std::fabs(sensor[0]), std::fabs(sensor[1]), std::fabs(sensor[2])});
    double t_max = std::max({std::fabs(sensor[3]), std::fabs(sensor[4]), std::fabs(sensor[5])});
    if (f_max > 100 || t_max > 10)
    {
        std::cout << "Warning! Force is beyond F/T sensor's range" << std::endl;
        std::exit(0);
    }

    // Gravity compensation
    gravity_sensor = cur_rot.transpose() * gravity_error.head<3>();
    moment_gravity << gravity_sensor(2) * tool_center(1, 0) - gravity_sensor(1) * tool_center(2, 0),
        gravity_sensor(0) * tool_center(2, 0) - gravity_sensor(2) * tool_center(0, 0),
        gravity_sensor(1) * tool_center(0, 0) - gravity_sensor(0) * tool_center(1, 0);
    zero_f = sensor_f - gravity_sensor;
    zero_t = sensor_t - moment_gravity;

    // Update gravity compensation parameters
    Eigen::Vector3d zero_t_para;
    zero_t_para << zero_t(0) - zero_f(2) * tool_center(1, 0) + zero_f(1) * tool_center(2, 0),
        zero_t(1) - zero_f(0) * tool_center(2, 0) + zero_f(2) * tool_center(0, 0),
        zero_t(2) - zero_f(1) * tool_center(0, 0) + zero_f(0) * tool_center(1, 0);
    gravity_error << gravity_error.head<3>(), zero_f;
    tool_center << tool_center.head<3>(), zero_t_para;

    std::cout << "Tool's gravity = " << gravity_error.head<3>().transpose() << std::endl;
    std::cout << "TCP = " << tool_center.head<3>().transpose() << std::endl;
    std::cout << "Force zero = " << zero_f.transpose() << std::endl;
    std::cout << "Torque zero = " << zero_t_para.transpose() << std::endl;
}

void RobotDrag::calRealForce()
{
    Eigen::Vector3d sensor_f, sensor_t, real_f, real_t, gravity_sensor, moment_gravity, moment_tool;
    sensor_f << sensor[0], sensor[1], sensor[2];
    sensor_t << sensor[3], sensor[4], sensor[5];

    double f_max = std::max({std::fabs(sensor[0]), std::fabs(sensor[1]), std::fabs(sensor[2])});
    double t_max = std::max({std::fabs(sensor[3]), std::fabs(sensor[4]), std::fabs(sensor[5])});
    if (f_max > 100 || t_max > 10)
    {
        std::cout << sensor_f.transpose() << " | " << sensor_t.transpose() << std::endl;
        std::cout << "Warning! Force is beyond F/T sensor's range" << std::endl;
        std::exit(0);
    }

    // Gravity compensation
    gravity_sensor = cur_rot.transpose() * gravity_error.head<3>();
    moment_gravity << gravity_sensor(2) * tool_center(1, 0) - gravity_sensor(1) * tool_center(2, 0),
        gravity_sensor(0) * tool_center(2, 0) - gravity_sensor(2) * tool_center(0, 0),
        gravity_sensor(1) * tool_center(0, 0) - gravity_sensor(0) * tool_center(1, 0);
    moment_tool << tool_center(3, 0) - gravity_error(4, 0) * tool_center(2, 0) + gravity_error(5, 0) * tool_center(1, 0),
        tool_center(4, 0) - gravity_error(5, 0) * tool_center(0, 0) + gravity_error(3, 0) * tool_center(2, 0),
        tool_center(5, 0) - gravity_error(3, 0) * tool_center(1, 0) + gravity_error(4, 0) * tool_center(0, 0);
    real_f = sensor_f - (gravity_sensor + gravity_error.tail<3>());
    real_t = sensor_t - (moment_gravity + moment_tool);

    real_ft << real_f, real_t;

    // Kalman Filter
    float Rk = 0.0002f;
    float Bk = 0.000001f;
    if (kalman_filter_time == 0)
    { // Initialization
        pk_last.setConstant(Bk);
        real_ft_last = real_ft;
    }
    for (int i = 0; i < 6; i++)
    {
        float xk_bar = real_ft_last[i];
        float pk_bar = pk_last[i] + Bk;
        real_ft[i] = xk_bar + (pk_bar / (pk_bar + Rk)) * (real_ft[i] - xk_bar);
        pk_last[i] = pk_bar - (pk_bar / (pk_bar + Rk)) * pk_bar;
        real_ft_last[i] = real_ft[i];
    }
    kalman_filter_time++;
}

void RobotDrag::PosCtr()  
{
    float acc[3] = {0, 0, 0};

    // Check if contact occurs
    for (int i = 0; i < 3; i++)
    {
        if (std::fabs(real_ft[i]) > 1.0)
        {
            acc[i] = (real_ft[i] - pos_d * cur_velo[i]) / pos_m;
            next_velo[i] = cur_velo[i] + acc[i] * delta_t;
            delta_pos[i] = (cur_velo[i] * delta_t + 0.5 * acc[i] * delta_t * delta_t);
            if (next_velo[i] > 1) next_velo[i] = 1;
            if (next_velo[i] < -1) next_velo[i] = -1;
        }
        else
        {
            next_velo[i] = 0;
            delta_pos[i] = 0;
        }
        // Security protect
        if (std::fabs(delta_pos[i]) > 0.3)
        {
            std::cout << "Error! Pos too fast!" << std::endl;
            std::exit(0);
        }
    }
    cur_velo = next_velo;
    // from Sensor Coordinate to World Coordinate
    next_pos = cur_pos + cur_rot * delta_pos;
    std::cout << "delta_pos = " << delta_pos.transpose() << std::endl;
}

void RobotDrag::OriCtr()
{
    float acc[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++)
    {
        if (std::fabs(real_ft[i + 3]) > 0.05)
        {
            acc[i] = (real_ft[i + 3] - ori_d * cur_ori_velo[i]) / ori_m;
            next_ori_velo[i] = cur_ori_velo[i] + acc[i] * delta_t;
            delta_theta[i] = (cur_ori_velo[i] * delta_t + 0.5 * acc[i] * delta_t * delta_t);
            if (next_ori_velo[i] > 1) next_ori_velo[i] = 1;
            if (next_ori_velo[i] < -1) next_ori_velo[i] = -1;
        }
        else
        {
            next_ori_velo[i] = 0;
            delta_theta[i] = 0;
        }
        // Security protect
        if (std::fabs(delta_theta[i]) > 0.2)
        {
            std::cout << "Error! Ori too fast!" << std::endl;
            std::exit(0);
        }
    }
    cur_ori_velo = next_ori_velo;
    std::cout << "delta_theta = " << delta_theta.transpose() << std::endl;
}

void RobotDrag::AdjustOri()
{
    Eigen::Matrix3d tmp_rot_x, tmp_rot_y, tmp_rot_z;

    double theta_rot_x = -20.0 * delta_theta[1];
    tmp_rot_x << std::cos(theta_rot_x), 0, std::sin(theta_rot_x),
        0, 1, 0,
        -std::sin(theta_rot_x), 0, std::cos(theta_rot_x);

    double theta_rot_y = -20.0 * delta_theta[0];
    tmp_rot_y << 1, 0, 0,
        0, std::cos(theta_rot_y), -std::sin(theta_rot_y),
        0, std::sin(theta_rot_y), std::cos(theta_rot_y);

    double theta_rot_z = -30.0 * delta_theta[2];
    tmp_rot_z << std::cos(theta_rot_z), -std::sin(theta_rot_z), 0,
        std::sin(theta_rot_z), std::cos(theta_rot_z), 0,
        0, 0, 1;

    next_rot = cur_rot * tmp_rot_y * tmp_rot_x * tmp_rot_z;
}

// int main()
// {
//     // // Force sensor
//     // std::thread th1(ForceSensor);
//     // th1.detach(); // 后台采集

//     //////////// Realman Robot //////////////
//     // RobotDrag Rbtd;  
//     // RMCommand RMcmd;  
//     // RMKinematics RMkine;   
//     /////////////////////////////////////////

//     using namespace sw::redis;
//     auto redis = Redis("tcp://127.0.0.1:6379");

//     while(true) {
//         auto val = redis.get("robot_state");
//         if (val) {
//             auto data = nlohmann::json::parse(*val);
//             std::cout << "C++ 读取状态：pos=" << data["pos"]
//                       << " vel=" << data["vel"] << std::endl;
//         }
//         usleep(500000);
//     }

//     // // Load GC parameters
//     // std::ifstream gravity_file("../data/gc_parameters.txt", std::ios::in);
//     // if (!gravity_file)   
//     // {
//     //     std::cout << "Error! Can't load GC parameters." << std::endl;
//     //     return 0;
//     // }   
//     // Rbtd.gravity_error.setZero();    
//     // Rbtd.tool_center.setZero();   
//     // for (int i = 0; i < 12; i++)   
//     // {
//     //     if (i < 6)
//     //         gravity_file >> Rbtd.gravity_error(i, 0);
//     //     else
//     //         gravity_file >> Rbtd.tool_center(i - 6, 0);
//     // }
//     // std::cout << "gravity_error = " << Rbtd.gravity_error.transpose() << std::endl;
//     // std::cout << "tool_center = " << Rbtd.tool_center.transpose() << std::endl;

//     // // Connect robot
//     // RMcmd.ConnectTCPSocket();
//     // // RMcmd.SetHighSpeedEth();
//     // // sleep(100);

//     // // Initialization
//     // Eigen::Matrix<double, 6, 1> start_joints;
//     // start_joints << 90 * M_PI / 180.0, 30 * M_PI / 180.0, 60 * M_PI / 180.0,
//     //     0 * M_PI / 180.0, -90 * M_PI / 180.0, 0 * M_PI / 180.0;

//     // // RMcmd.MoveJ(start_joints, 10);
//     // sleep(1);

//     // Eigen::Matrix<double, 6, 1> cur_joints, next_joints;
//     // Eigen::Matrix4d cur_kinematics, next_kinematics;

//     // // Update force sensor zero
//     // RMcmd.ReadJ(cur_joints);
//     // std::cout << "cur_joints = " << cur_joints.transpose() << std::endl;

//     // RMkine.GetKinematics(cur_kinematics, cur_joints);
//     // Rbtd.cur_rot = cur_kinematics.block<3,3>(0,0);
//     // Rbtd.cur_pos = cur_kinematics.block<3,1>(0,3);

//     // // Rbtd.updateFTzero();
//     // sleep(1);

//     // std::chrono::duration<double, std::milli> fp_ms;
//     // auto t1 = std::chrono::high_resolution_clock::now();
//     // auto t2 = t1;  

//     // while (true)
//     // {
//     //     t1 = std::chrono::high_resolution_clock::now();
//     //     std::cout << "------------------------" << std::endl;
//     //     RMcmd.ReadJ(cur_joints);  
//     //     RMkine.GetKinematics(cur_kinematics, cur_joints);  
//     //     Rbtd.cur_rot = cur_kinematics.block<3,3>(0,0);  
//     //     Rbtd.cur_pos = cur_kinematics.block<3,1>(0,3);  
//     //     std::cout << "Rbtd.cur_pos = " << Rbtd.cur_pos.transpose() << std::endl;

//     //     // Gravity compensation and kalman filter   
//     //     Rbtd.calRealForce();  

//     //     std::cout << "Contact Force = " << Rbtd.real_ft.transpose() << std::endl;
//     //     if (std::fabs(Rbtd.real_ft[2]) > 50)   
//     //     {
//     //         std::cout << "Error! Z-axis force beyond sensor range." << std::endl;
//     //         return 0;
//     //     }

//     //     Rbtd.PosCtr();  
//     //     Rbtd.OriCtr();  
//     //     Rbtd.AdjustOri();  
//     //     Rbtd.next_rot = Rbtd.cur_rot;   

//     //     // Robot motion  
//     //     next_kinematics << Rbtd.next_rot, Rbtd.next_pos,  
//     //         0, 0, 0, 1;  

//     //     // Robot next joints  
//     //     RMkine.GetNextJoints(next_joints, cur_joints, next_kinematics);  
//     //     std::cout << "cur_joints = " << cur_joints.transpose() << std::endl;   
//     //     std::cout << "next_joints = " << next_joints.transpose() << std::endl;   
        
//     //     // 小力时保持当前关节，避免抖动；有力时跟随
//     //     if (std::fabs(Rbtd.real_ft[0]) < 1 && std::fabs(Rbtd.real_ft[1]) < 1 && std::fabs(Rbtd.real_ft[2]) < 1 &&
//     //         std::fabs(Rbtd.real_ft[3]) < 0.5 && std::fabs(Rbtd.real_ft[4]) < 0.5 && std::fabs(Rbtd.real_ft[5]) < 0.5)
//     //     {

//     //         RMcmd.ServoJ(cur_joints, false);  // 6轴兼容接口，底层做7轴映射  
//     //     }
//     //     else
//     //     {
//     //         RMcmd.ServoJ(next_joints, false); // 6轴兼容接口，底层做7轴映射
            
//     //     }

//     //     // Wait servo time ~50ms
//     //     while (true)
//     //     {
//     //         usleep(1000);
//     //         t2 = std::chrono::high_resolution_clock::now();
//     //         fp_ms = t2 - t1;
//     //         if (fp_ms.count() > 50.0)
//     //             break;
//     //     }
//     //     std::cout << std::endl;
//     // }
//     return 0;
// }

int main()
{
    //////////////////////// Force sensor //////////////////////////
    // std::thread th1(ForceSensor);
    // th1.detach(); // 后台采集
    //////////////////////// Force sensor //////////////////////////

    auto redis = Redis("tcp://localhost:6379");  
    auto sub_target_robot_state = redis.subscriber();   

    std::cout << "Robot is ready." << std::endl;
    // Eigen::Matrix<double, 7, 1> cur_joints, next_joints;  
    Eigen::Matrix<double, 7, 1> cur_joints_vel, target_joints_vel;   

    Eigen::Matrix4d cur_kinematics, next_kinematics;

    cur_joints.setZero();   
    next_joints.setZero();    
    cur_joints_vel.setZero(); 
    target_joints_vel.setZero();  
 
    sub_target_robot_state.on_message([](std::string channel, std::string msg) {
        json j = json::parse(msg);    
        std::vector<double> arr_joint_angle = j["target_joint_angle"].get<std::vector<double>>();  
        // Eigen::Map<Eigen::VectorXd> next_joints(arr_joint_angle.data(), arr_joint_angle.size());   
        // std::cout << "received target position: " << j["target_joint_angle"][0] << std::endl; 

        Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(arr_joint_angle.data(), arr_joint_angle.size());  
        std::lock_guard<std::mutex> lock(g_mutex);   
        next_joints = vec;  
        // std::cout << "next joints:" << next_joints.transpose() << std::endl;  
    });  

    sub_target_robot_state.subscribe("target_position");   

    ////////////////////// Realman Robot Connection ///////////////  
    RobotDrag Rbtd;   
    RMCommand RMcmd;   
    RMKinematics RMkine;   

    //////////////////// Load GC parameters /////////////////////// 
    // std::ifstream gravity_file("../data/gc_parameters.txt", std::ios::in);
    // if (!gravity_file)
    // {
    //     std::cout << "Error! Can't load GC parameters." << std::endl;
    //     return 0;
    // }
    // Rbtd.gravity_error.setZero();
    // Rbtd.tool_center.setZero();
    // for (int i = 0; i < 12; i++)
    // {
    //     if (i < 6)
    //         gravity_file >> Rbtd.gravity_error(i, 0);
    //     else
    //         gravity_file >> Rbtd.tool_center(i - 6, 0);
    // }
    // std::cout << "gravity_error = " << Rbtd.gravity_error.transpose() << std::endl;
    // std::cout << "tool_center = " << Rbtd.tool_center.transpose() << std::endl;
    //////////////////// Load GC parameters ///////////////////////   

    // Connect robot
    RMcmd.ConnectTCPSocket(); 

    // RMcmd.SetHighSpeedEth(); 
    std::cout << "Waiting for robot to be ready..." << std::endl;
    sleep(1);  

    // Initialization
    Eigen::Matrix<double, 7, 1> start_joints;
    start_joints << 0, 90 * M_PI / 180.0, 30 * M_PI / 180.0, 60 * M_PI / 180.0,
        0 * M_PI / 180.0, -90 * M_PI / 180.0, 0 * M_PI / 180.0;

    // RMcmd.MoveJ(start_joints, 10);
    // sleep(1);  

    // Update force sensor zero
    // RMcmd.ReadJ7(cur_joints);   

    // std::cout << "cur_joints = " << cur_joints.transpose() << std::endl;    
    
    // RMkine.GetKinematics(cur_kinematics, cur_joints);
    // Rbtd.cur_rot = cur_kinematics.block<3,3>(0,0);
    // Rbtd.cur_pos = cur_kinematics.block<3,1>(0,3);

    // // Rbtd.updateFTzero();  
    // sleep(1);  

    std::chrono::duration<double, std::milli> fp_ms;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1;   
    // Eigen::VectorXd next_joints_new; 

    std::string channel;
    std::string payload;
    Eigen::VectorXd move_joints_7;  

    RMcmd.ReadJ7(cur_joints);   
    move_joints_7 = cur_joints; 
    // next_joints   = cur_joints; 
    // next_joints.setZero(); 

    while (true)   
    {
        t1 = std::chrono::high_resolution_clock::now();  
        std::cout << "------------------------" << std::endl;  
        RMcmd.ReadJ7(cur_joints);   
        // next_joints   = cur_joints;   
        move_joints_7 = cur_joints;  


        // next_joints = cur_joints;    

        // cur_joints[0] += 0.1;  
        
        /////// publish robot state ////////
        json robot_state_info;  
        robot_state_info["joint_angle"] = std::vector<double>(cur_joints.data(), cur_joints.data() + cur_joints.size()); 
        std::string str_robot_state_info = robot_state_info.dump();    
        std::cout << str_robot_state_info << std::endl;  
        redis.publish("actual_joint_angle", str_robot_state_info);      
        std::cout << "Published: " << str_robot_state_info << std::endl;   

        ////////// subscriber ////////// 

        // auto msg = sub_target_robot_state.consume();
        // std::pair<std::string, std::string> msg_pair;
        // sub_target_robot_state.consume(msg_pair);

        // // 消息内容
        // std::string channel = msg.channel();
        // std::string payload = msg.message();  

        // sub_target_robot_state.consume();  
        // std::lock_guard<std::mutex> lock(g_mutex);
        // move_joints_7 = cur_joints + next_joints;  

        if (move_joints_7.norm() > 0.1)  
        {
            move_joints_7 = next_joints;  
        }
        else
        {
            move_joints_7 = cur_joints;  
        }
       
        std::cout << "next joints: " << move_joints_7.transpose() << std::endl; 

        ///////// Servo Control /////////////
        // RMcmd.ServoJ7(move_joints_7, false);      
        // RMcmd.MoveJ7(next_joints, 2);     

        // while (true) {
        //     sub_target_robot_state.consume();  // 等待消息
        //     std::cout << "next joints: " << next_joints.transpose() << std::endl;
        // }
        // // RMcmd.ReadJ7(cur_joints);    
        // next_joints = cur_joints;    
        // std::cout << "current joints: " << cur_joints.transpose() * 180.0 / M_PI << " deg" << std::endl;     
        
        // auto val = redis.get("target_position");  
        // if (val) {
        //     auto data = nlohmann::json::parse(*val); 
        //     std::vector<double> arr_joint_angle = data["target_joint_angle"].get<std::vector<double>>();  
        //     Eigen::Map<Eigen::VectorXd> next_joints(arr_joint_angle.data(), arr_joint_angle.size());
        //     std::cout << "C++ 读取状态 : pos=" << data["target_joint_angle"] << std::endl;
        //     // next_joints << arr_joint_angle[0], arr_joint_angle[1], arr_joint_angle[2], arr_joint_angle[3], arr_joint_angle[4], arr_joint_angle[5], arr_joint_angle[6];
        // }   
        // else
        // {
        //     next_joints = cur_joints; 
        // }

        // std::cout << "next joints:" << next_joints.transpose() << std::endl;
        // std::cout << "next joints: " << next_joints.transpose() * 180.0 / M_PI << " deg" << std::endl;     

        ///////// Robot Kinematics //////////
        // RMkine.GetKinematics(cur_kinematics, cur_joints);  
        // Rbtd.cur_rot = cur_kinematics.block<3,3>(0,0); 
        // Rbtd.cur_pos = cur_kinematics.block<3,1>(0,3);   
        // std::cout << "Rbtd.cur_pos = " << Rbtd.cur_pos.transpose() << std::endl;   
        // // Gravity compensation and kalman filter
        ///////// Robot Kinematics //////////   
        
        //////////// RCM Control ////////////
        // null space matrix //
        // MatrixXd N = MatrixXd::Identity(nv, nv) - J6_pinv * J6;
        // VectorXd qdot = qdot_task + N * qdot_RCM;   

        // next_joints = ; 
        //////////// RCM Control ////////////

        ///////// Admittance control ////////
        // Rbtd.calRealForce();
        // std::cout << "Contact Force = " << Rbtd.real_ft.transpose() << std::endl;
        // if (std::fabs(Rbtd.real_ft[2]) > 50)
        // {
        //     std::cout << "Error! Z-axis force beyond sensor range." << std::endl;
        //     return 0;
        // }
        // Rbtd.PosCtr();  
        // Rbtd.OriCtr();
        // Rbtd.AdjustOri();
        // Rbtd.next_rot = Rbtd.cur_rot;  
        // // Robot motion
        // next_kinematics << Rbtd.next_rot, Rbtd.next_pos,
        //     0, 0, 0, 1;  
        // RMkine.GetNextJoints(next_joints, cur_joints, next_kinematics); 
        // std::cout << "cur_joints = " << cur_joints.transpose() << std::endl; 
        // std::cout << "next_joints = " << next_joints.transpose() << std::endl;  
        
        // // // 小力时保持当前关节，避免抖动；有力时跟随
        // // if (std::fabs(Rbtd.real_ft[0]) < 1 && std::fabs(Rbtd.real_ft[1]) < 1 && std::fabs(Rbtd.real_ft[2]) < 1 &&
        // //     std::fabs(Rbtd.real_ft[3]) < 0.5 && std::fabs(Rbtd.real_ft[4]) < 0.5 && std::fabs(Rbtd.real_ft[5]) < 0.5)
        // // {

        // //     RMcmd.ServoJ(cur_joints, false);  // 6轴兼容接口，底层做7轴映射  
        // // }
        // // else
        // // {
        // //     RMcmd.ServoJ(next_joints, false); // 6轴兼容接口，底层做7轴映射
            
        // // }
        ///////// Admittance control ////////
    
        // Wait servo time ~50ms
        while (true)
        {
            usleep(1000);
            t2 = std::chrono::high_resolution_clock::now();
            fp_ms = t2 - t1;
            if (fp_ms.count() > 50.0)
                break;
        }  
        std::cout << std::endl;
    }
    
    return 0;   
}