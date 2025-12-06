#include <realman_command.hpp>

// 假设类里已有这些成员：
// int rlm_port; int rlm_socket; char rlm_ip[64];
// nlohmann::json command_msg, return_msg; std::string cmd_str;
// char send_msg[1000], recv_msg[1000]; int recv_times;
// Eigen::Matrix<double,7,1> cmd_joints7; // 新增：7轴缓存
// Eigen::Matrix<double,6,1> cmd_joints6; // 可选：6轴缓存（兼容用）
int    TCP_NODELAY=1;
RMCommand::RMCommand(){
    rlm_port = 8080;
    std::string string_ip = "192.168.2.18";
    strcpy(rlm_ip, string_ip.c_str());
}

void RMCommand::ConnectTCPSocket(){
    struct sockaddr_in server_addr;
    rlm_socket = socket(AF_INET,SOCK_STREAM, 0);
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(rlm_ip);
    server_addr.sin_port = htons(rlm_port);
    rlm_socket = socket(AF_INET, SOCK_STREAM, 0);
    if(connect(rlm_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0){
        std::cout << "ERROR! Can't connect robot!" << std::endl;
        std::exit(0);
    }else{
        std::cout << "Robot connect!" << std::endl; 

    }
}

void RMCommand::SetHighSpeedEth(){
    // Open high speed ethernet
    command_msg.clear();
    command_msg["command"] = "set_high_speed_eth";
    command_msg["mode"] = 1;
    cmd_str = command_msg.dump()+"\r\n";
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000);recv_times = 0;
        while(recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                std::cout << return_msg.dump() << std::endl;
            }
        }
    }

    // Set IP
    command_msg.clear();
    command_msg["command"] = "set_high_ethernet";
    command_msg["ip"] = "192.168.1.18";
    command_msg["mask"] = "255.255.255.0";
    command_msg["gateway"] = "192.168.1.1";
    cmd_str = command_msg.dump()+"\r\n";
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000);recv_times = 0;
        while(recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                std::cout << return_msg.dump() << std::endl;
            }
        }
    }

    // Save info
    command_msg.clear();
    command_msg["command"] = "save_device_info_all";
    cmd_str = command_msg.dump()+"\r\n";
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000);recv_times = 0;
        while(recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                std::cout << return_msg.dump() << std::endl;
            }
        }
    }
    
    std::cout << "Successfully set high speed ethernet! Change port and restart the robot." << std::endl;
}

/**********************
 * 7轴原生读/写接口
 **********************/

// 读7轴关节（弧度）
void RMCommand::ReadJ7(Eigen::Matrix<double,7,1>& joints7){
    command_msg.clear();
    command_msg["command"] = "get_joint_degree";
    cmd_str = command_msg.dump()+"\r\n";
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000); recv_times=0;
        while (recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                // 假设返回 joint 数组长度为 7（单位：0.001度）
                for (int i = 0; i < std::min<int>(7, return_msg["joint"].size()); i++){
                    cmd_joints7[i] = return_msg["joint"][i].get<double>();
                }
            }            
        }
    }
    // 转成弧度
    joints7 = (cmd_joints7/1000.0)*M_PI/180.0;
}

// 关节运动（7轴，弧度输入）
void RMCommand::MoveJ7(const Eigen::Matrix<double,7,1>& joints7, int velo){
    command_msg.clear();
    command_msg["command"] = "movej";
    for(int i=0; i<7; i++){
        command_msg["joint"][i] = int(1000.0*joints7[i]*180.0/M_PI); // 0.001 deg
    }
    command_msg["v"] = velo;
    command_msg["r"] = 0;
    cmd_str = command_msg.dump()+"\r\n";
    std::cout << cmd_str << std::endl;
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000); recv_times=0;
        while (recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                if(return_msg.contains("trajectory_state") && return_msg["trajectory_state"].get<bool>()){
                    std::cout << "MoveJ7 OK!\t" << return_msg.dump() << std::endl;
                }else{
                    std::cout << "ERROR! MoveJ7 False!\t" << return_msg.dump() << std::endl;
                    std::exit(0);
                }
            }
        }
    }
}

// 直线运动（末端 6 自由度，与原逻辑相同）
void RMCommand::MoveL(const Eigen::Matrix<double,6,1>& pose, int velo){
    command_msg.clear();
    command_msg["command"] = "movel";
    for(int i = 0; i < 6; i++){
        if(i < 3){
            command_msg["pose"][i] = int(1000*1000*pose[i]); // 0.001 mm
        }else{
            command_msg["pose"][i] = int(1000*pose[i]); // 0.001 rad
        }
    }
    command_msg["v"] = velo;
    command_msg["r"] = 0;
    cmd_str = command_msg.dump()+"\r\n";
    std::cout << cmd_str << std::endl;
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000); recv_times=0;
        while (recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                if(return_msg.contains("trajectory_state") && return_msg["trajectory_state"].get<bool>()){
                    std::cout << "MoveL OK!\t" << return_msg.dump() << std::endl;
                }else{
                    std::cout << "ERROR! MoveL False!\t" << return_msg.dump() << std::endl;
                    std::exit(0);
                }
            }
        }
    }
}

// 直线运动（末端 6 自由度，与原逻辑相同）  
void RMCommand::ServoJ7(const Eigen::Matrix<double,7,1>& joints7, bool follow)
{
    // 1) 构造 JSON 行
    command_msg.clear();
    command_msg["command"] = "movej_canfd";
    for (int i = 0; i < 7; ++i)
        command_msg["joint"][i] = int(1000.0 * joints7[i] * 180.0 / M_PI);
    // if (protocol_supports_follow) command_msg["follow"] = follow;

    cmd_str = command_msg.dump();
    cmd_str += "\r\n";

    // 2) 发送（send_all 确保完整发送）
    auto send_all = [&](const char* buf, size_t len) -> bool {
        size_t sent = 0;
        while (sent < len) {
            ssize_t n = ::send(rlm_socket, buf + sent, len - sent, 0);
            if (n < 0) {
                if (errno == EINTR) continue;
                std::cout << "ERROR! send() failed: " << strerror(errno) << std::endl;
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    };

    // 可选：仅初始化一次，放到 ConnectTCPSocket 更好
    static bool sock_opts_set = false;
    if (!sock_opts_set) {
        int yes = 1;
        setsockopt(rlm_socket, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        // 非阻塞读取策略：我们不设置 SO_RCVTIMEO，而是用 MSG_DONTWAIT
        // 如需阻塞+超时，可改为 SO_RCVTIMEO
        sock_opts_set = true;
    }

    if (!send_all(cmd_str.c_str(), cmd_str.size())) {
        std::cout << "ERROR! Can't send message (ServoJ7)" << std::endl;
        std::exit(0);
    }

    // 3) 接收（非阻塞、基于行结束符拼包；可在实时环里安全失败）
    // 说明：
    // - 若此接口在你的控制器上“并不保证回包”，建议直接跳过接收段，函数直接 return。
    // - 若需要回包，我们尝试在短时间窗口内读取一整行 JSON。
    char buf[1024];
    std::string line;
    const int max_wait_ms = 20;    // 短超时窗口，避免卡住实时周期
    const int slice_us    = 1000;  // 间隔轮询步长

    int waited_ms = 0;
    while (waited_ms < max_wait_ms) {
        ssize_t n = ::recv(rlm_socket, buf, sizeof(buf) - 1, MSG_DONTWAIT);
        if (n > 0) {
            buf[n] = '\0';
            line.append(buf, static_cast<size_t>(n));
            // 查找一整行（以 \r\n 结束）
            size_t pos = line.find("\r\n");
            if (pos != std::string::npos) {
                std::string one_line = line.substr(0, pos);
                // 如果后面还有数据，保留给下次（这里简单忽略多余部分，如需可做缓冲）
                // 4) 解析 JSON
                nlohmann::json ret = nlohmann::json::parse(one_line, nullptr, false);
                if (ret.is_discarded()) {
                    std::cout << "WARNING! ServoJ7: invalid JSON: " << one_line << std::endl;
                } else {
                    if (ret.contains("arm_err")) {
                        int err = 0;
                        try { err = ret["arm_err"].get<int>(); } catch (...) {}
                        if (err != 0) {
                            std::cout << "WARNING! ServoJ7: arm_err=" << err << " resp=" << ret.dump() << std::endl;
                        }
                    }
                }
                return; // 成功收到一行就返回
            }
            // 未完整行，继续轮询直到超时或读完
        } else if (n == 0) {
            // 对端关闭
            std::cout << "ERROR! ServoJ7: peer closed connection." << std::endl;
            std::exit(0);
        } else { // n < 0
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 没数据，等一下
                usleep(slice_us);
                waited_ms += slice_us / 1000;
                continue;
            } else if (errno == EINTR) {
                continue;
            } else {
                std::cout << "ERROR! recv() failed: " << strerror(errno) << std::endl;
                // 可选择返回或退出，这里不直接退出以提升鲁棒性
                return;
            }
        }
        // 控制节奏
        usleep(slice_us);
        waited_ms += slice_us / 1000;
    }

    // 超时无回包：在实时模式下属于可接受情形，打印一次提示即可
    // 若你希望严格确认，应在外层统计连续超时次数并触发重连/停机
    // std::cout << "INFO: ServoJ7 no ACK within " << max_wait_ms << "ms, continue." << std::endl;
    return;
}

/********************************
 * 6轴兼容适配（q4 永远为 0）
 ********************************/

// 将 6 轴关节映射为 7 轴：在 index=3 处插入 q4=0（以 0 基为例：q1,q2,q3,[q4=0],q5,q6,q7）
static inline Eigen::Matrix<double,7,1> map6to7_with_q4zero(const Eigen::Matrix<double,6,1>& q6){
    Eigen::Matrix<double,7,1> q7;
    q7[0] = q6[0];
    q7[1] = q6[1];
    q7[2] = 0.0;
    q7[3] = q6[2];       // 冗余轴固定为 0（i=4）
    q7[4] = q6[3];
    q7[5] = q6[4];
    q7[6] = q6[5];
    return q7;
}

// 将 7 轴关节映射回 6 轴：丢弃 index=3
static inline Eigen::Matrix<double,6,1> map7to6_drop_q4(const Eigen::Matrix<double,7,1>& q7){
    Eigen::Matrix<double,6,1> q6;
    q6[0] = q7[0];
    q6[1] = q7[1];
    q6[2] = q7[3];
    q6[3] = q7[4];
    q6[4] = q7[5];
    q6[5] = q7[6];
    return q6;
}

// 兼容读（返回 6 轴，内部读 7 轴再丢弃 q4）
void RMCommand::ReadJ(Eigen::Matrix<double,6,1>& joints6){
    Eigen::Matrix<double,7,1> j7;
    ReadJ7(j7);
    joints6 = map7to6_drop_q4(j7);
}

// 兼容 MoveJ：6 轴输入，映射为 7 轴并固定 q4=0
void RMCommand::MoveJ(Eigen::Matrix<double,6,1>& joints6, int velo){
    Eigen::Matrix<double,7,1> j7 = map6to7_with_q4zero(joints6);
    MoveJ7(j7, velo);
}

// 兼容 MoveJ_P（末端位姿与原逻辑一致，无需改）
// 已在上方保留 MoveL、MoveJ_P（若你需要 MoveJ_P 的名字与实现，直接使用你现有的那套即可）

// 兼容 ServoJ：6 轴输入 -> 7 轴，q4=0
void RMCommand::ServoJ(Eigen::Matrix<double,6,1>& joints6, bool follow){
    Eigen::Matrix<double,7,1> j7 = map6to7_with_q4zero(joints6);
    ServoJ7(j7, follow);
}

// 如果你还需要 MoveJ_P 兼容版，直接复用原函数签名即可，无需动关节维度。
void RMCommand::MoveJP(const Eigen::Matrix<double,6,1>& pose, int velo){
    command_msg.clear();
    command_msg["command"] = "movej_p";
    for(int i = 0; i < 6; i++){
        if(i < 3){
            command_msg["pose"][i] = int(1000*1000*pose[i]); // 0.001 mm
        }else{
            command_msg["pose"][i] = int(1000*pose[i]); // 0.001 rad
        }
    }
    command_msg["v"] = velo;
    command_msg["r"] = 0;
    cmd_str = command_msg.dump()+"\r\n";
    std::cout << cmd_str << std::endl;
    memset(send_msg, 0, 1000);
    strcpy(send_msg, cmd_str.c_str());
    if(send(rlm_socket, send_msg, strlen(send_msg), 0) < 0){
        std::cout << "ERROR! Can't send message! " << std::endl;
        std::exit(0);
    }else{
        memset(recv_msg, 0, 1000); recv_times=0;
        while (recv(rlm_socket, recv_msg, 1000, 0) < 10 && recv_times < 3) recv_times++;
        if(recv_times == 3){
            std::cout << "ERROR! Can't recive message! " << std::endl;
            std::exit(0);
        }else{
            return_msg.clear();
            return_msg = nlohmann::json::parse(recv_msg, nullptr, false);
            if (return_msg.is_discarded()){
                std::cout << "WARNING! Missing a return message." << return_msg.dump() << std::endl;
            }else{
                if(return_msg.contains("trajectory_state") && return_msg["trajectory_state"].get<bool>()){
                    std::cout << "MoveJ_P OK!\t" << return_msg.dump() << std::endl;
                }else{
                    std::cout << "ERROR! MoveJ_P False!\t" << return_msg.dump() << std::endl;
                    std::exit(0);
                }
            }
        }
    }
}