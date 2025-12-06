#pragma once
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <iomanip>

// ================= SDK 外部函数声明（按您提供的原型） =================
// 实际项目中应 #include 官方头文件，这里的声明只为演示。
extern "C" {
    // 连接
    int HRIF_Connect(unsigned int boxID, const char* hostName, unsigned short nPort);
    int HRIF_DisConnect(unsigned int boxID);
    int HRIF_Connect2Box(unsigned int boxID);
    int HRIF_Connect2Controller(unsigned int boxID);

    // 您提供的清单里“Blackout”大小写为 HRIF_Blackout
    int HRIF_Blackout(unsigned int boxID);

    // 速度比/状态
    int HRIF_SetOverride(unsigned int boxID, unsigned int rbtID, double dOverride);
    int HRIF_ReadRobotState(unsigned int boxID, unsigned int rbtID,
                            int &nMovingState,
                            int &nEnableState,
                            int &nErrorState,
                            int &nErrorCode,
                            int &nErrorAxis,
                            int &nBreaking,
                            int &nPause,
                            int &nEmergencyStop,
                            int &nSafeGuard,
                            int &nElectrify,
                            int &nIsConnectToBox,
                            int &nBlendingDone,
                            int &nInpos);
    int HRIF_IsBlendingDone(unsigned int boxID, unsigned int rbtID, bool& bDone);

    // 读位置
    // 简化的 joints 容器：SDK 实际应有 JointsData 类型，请用真实类型替换
    struct JointsData { double J[6]; };
    int HRIF_ReadActJointPos_nJ(unsigned int boxID, unsigned int rbtID, JointsData& joints);
    int HRIF_ReadActTcpPos(unsigned int boxID, unsigned int rbtID,
                           double& dX, double& dY, double& dZ,
                           double& dRx, double& dRy, double& dRz);

    // 运动
    int HRIF_MoveJ(unsigned int boxID, unsigned int rbtID,
                   double dX, double dY, double dZ, double dRx, double dRy, double dRz,
                   double dJ1, double dJ2, double dJ3, double dJ4, double dJ5, double dJ6,
                   std::string sTcpName, std::string sUcsName,
                   double dVelocity, double dAcc, double dRadius,
                   int nIsUseJoint, int nIsSeek, int nIOBit, int nIOState, std::string strCmdID);

    int HRIF_MoveL(unsigned int boxID, unsigned int rbtID,
                   double dX, double dY, double dZ, double dRx, double dRy, double dRz,
                   double dJ1, double dJ2, double dJ3, double dJ4, double dJ5, double dJ6,
                   std::string sTcpName, std::string sUcsName,
                   double dVelocity, double dAcc, double dRadius,
                   int nIsSeek, int nIOBit, int nIOState, std::string strCmdID);

    // Servo
    int HRIF_StartServo(unsigned int boxID, unsigned int rbtID, double dServoTime, double dLookaheadTime);
    int HRIF_PushServoJ(unsigned int boxID, unsigned int rbtID,
                        double dJ1, double dJ2, double dJ3, double dJ4, double dJ5, double dJ6);
    // PushServoP 原型在您的片段末尾被截断，这里给出常见形式（请据官方头文件修正）
    int HRIF_PushServoP(unsigned int boxID, unsigned int rbtID,
                        const std::vector<double>& vecCoord,
                        const std::vector<double>& vecUcs,
                        const std::vector<double>& vecTcp);
}

// ================= 错误类型 =================
class HansError : public std::runtime_error {
public:
    explicit HansError(int code, const std::string& msg = "")
        : std::runtime_error(buildMsg(code, msg)), code_(code), msg_(msg) {}
    int code() const noexcept { return code_; }
    const std::string& detail() const noexcept { return msg_; }
private:
    int code_;
    std::string msg_;
    static std::string buildMsg(int code, const std::string& msg) {
        std::ostringstream oss;
        oss << "Hans SDK 调用失败，错误码=" << code;
        if (!msg.empty()) oss << "，" << msg;
        return oss.str();
    }
};

// 如果有“获取错误码文本”的 SDK 接口，可在这里查询并拼接更详细的说明。
// 目前仅抛出错误码。
inline void check_ret(int ret) {
    if (ret != 0) {
        throw HansError(ret);
    }
}

// ================== 封装类：DCSCommand ==================
class DCSCommand {
public:
    DCSCommand(std::string ip = "192.168.1.10",
               unsigned short port = 10003,
               unsigned int box_id = 0,
               unsigned int rbt_id = 0)
        : ip_(std::move(ip)), port_(port), box_id_(box_id), rbt_id_(rbt_id), connected_(false) {}

    // 连接/断开
    void connectTCPSocket() {
        check_ret(HRIF_Connect(box_id_, ip_.c_str(), port_));
        connected_ = true;
    }

    void closeTCPSocket() {
        try {
            check_ret(HRIF_DisConnect(box_id_));
        } catch (...) {
            connected_ = false;
            throw;
        }
        connected_ = false;
    }

    // 上电/断电（注意：您给的清单未提供 Electrify。Connect2Box 可作为“连接电箱”；断电用 Blackout）
    void powerON() {
        // 等价 Python: Connect2Box + Electrify。此处仅做 Connect2Box。
        check_ret(HRIF_Connect2Box(box_id_));
        // 若 SDK 有 HRIF_Electrify(boxID) 请在此调用。
    }

    void powerOFF() {
        // 对应 HRIF_Blackout（注意大小写按您清单）
        check_ret(HRIF_Blackout(box_id_));
    }

    // 使能/去使能：您清单里提供的是 HRIF_Connect2Controller 与 HRIF_GrpDisable
    void rbtEnable() {
        // 连接控制器、主站启动、参数配置并进入 DISABLE（SDK 描述如此），
        // 真正“上使能”的接口在清单中未给出。如有 HRIF_GrpEnable 请改为调用它。
        check_ret(HRIF_Connect2Controller(box_id_));
        // check_ret(HRIF_GrpEnable(box_id_, rbt_id_)); // 如果头文件提供了该函数
    }


    void clearErrorState() {
        // Python 里用 HRIF_GrpReset；清单未给出该函数，这里仅占位提示。
        // 请替换为 SDK 的复位接口，如：check_ret(HRIF_GrpReset(box_id_, rbt_id_));
        throw std::runtime_error("未实现：请使用 SDK 的 HRIF_GrpReset 接口");
    }

    void stop() {
        // Python 里用 HRIF_GrpStop；清单未给出该函数，这里仅占位提示。
        // 请替换为 SDK 的停止接口，如：check_ret(HRIF_GrpStop(box_id_, rbt_id_));
        throw std::runtime_error("未实现：请使用 SDK 的 HRIF_GrpStop 接口");
    }

    // 速度
    void setSpeedRadio(double vel_ratio) {
        if (vel_ratio < 0.01 || vel_ratio > 1.0) {
            throw std::invalid_argument("速度比 vel_ratio 应在[0.01, 1.0]");
        }
        check_ret(HRIF_SetOverride(box_id_, rbt_id_, vel_ratio));
    }

    // 状态读取
    struct RobotState {
        bool moving{};
        bool enabled{};          // SDK 未直接定义“enabled”：由 nEnableState 返回
        bool has_error{};
        int  error_code{};
        int  error_axis{};
        bool brake_released{};
        bool paused{};
        bool estop{};
        bool safelight{};
        bool electrified{};
        bool box_connected{};
        bool waypoint_done{};
        bool in_position{};
    };

    RobotState readRobotState() {
        int nMoving=0, nEnable=0, nError=0, nErrCode=0, nErrAxis=0;
        int nBreaking=0, nPause=0, nEStop=0, nSafe=0, nElectrify=0, nConnBox=0, nBlend=0, nInpos=0;
        check_ret(HRIF_ReadRobotState(box_id_, rbt_id_,
                                      nMoving, nEnable, nError, nErrCode, nErrAxis,
                                      nBreaking, nPause, nEStop, nSafe, nElectrify,
                                      nConnBox, nBlend, nInpos));
        RobotState s{};
        s.moving = nMoving != 0;
        s.enabled = nEnable != 0;
        s.has_error = nError != 0;
        s.error_code = nErrCode;
        s.error_axis = nErrAxis;
        s.brake_released = nBreaking != 0;
        s.paused = nPause != 0;
        s.estop = nEStop != 0;
        s.safelight = nSafe != 0;
        s.electrified = nElectrify != 0;
        s.box_connected = nConnBox != 0;
        s.waypoint_done = nBlend != 0;
        s.in_position = nInpos != 0;
        return s;
    }

    bool isCompleteMovement() {
        bool done = false;
        check_ret(HRIF_IsBlendingDone(box_id_, rbt_id_, done));
        return done;
    }

    // 位置读取
    std::array<double, 6> readCurrentAct() {
        JointsData joints{};
        check_ret(HRIF_ReadActJointPos_nJ(box_id_, rbt_id_, joints));
        return { joints.J[0], joints.J[1], joints.J[2], joints.J[3], joints.J[4], joints.J[5] };
    }

    std::array<double, 6> readCurrentPos() {
        double x=0,y=0,z=0,rx=0,ry=0,rz=0;
        check_ret(HRIF_ReadActTcpPos(box_id_, rbt_id_, x,y,z,rx,ry,rz));
        return { x,y,z,rx,ry,rz };
    }

    // 基本运动
    void moveJ(const std::array<double,6>& joint_deg,
               double vel=50.0, double acc=50.0, double radius=0.0,
               int is_use_joint=1,
               int seek_di=0, int io_bit=0, int io_state=0,
               const std::string& cmd_id="0") {
        // pose 未使用时传 0
        const double x=0, y=0, z=0, rx=0, ry=0, rz=0;
        const std::string sTcp="TCP";
        const std::string sUcs="Base";
        check_ret(HRIF_MoveJ(
            box_id_, rbt_id_,
            x, y, z, rx, ry, rz,
            joint_deg[0], joint_deg[1], joint_deg[2], joint_deg[3], joint_deg[4], joint_deg[5],
            sTcp, sUcs,
            vel, acc, radius,
            is_use_joint, seek_di, io_bit, io_state, cmd_id
        ));
    }

    void moveL(const std::array<double,6>& pose,
               const std::array<double,6>* ref_joint_deg=nullptr,
               double vel=100.0, double acc=250.0, double radius=0.0,
               int seek_di=0, int io_bit=0, int io_state=0,
               const std::string& cmd_id="0") {
        std::array<double,6> j = ref_joint_deg ? *ref_joint_deg
                                               : std::array<double,6>{0,0,0,0,0,0};
        const std::string sTcp="TCP";
        const std::string sUcs="Base";
        check_ret(HRIF_MoveL(
            box_id_, rbt_id_,
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5],
            j[0], j[1], j[2], j[3], j[4], j[5],
            sTcp, sUcs,
            vel, acc, radius,
            seek_di, io_bit, io_state, cmd_id
        ));
    }

    // Servo
    void startServo(double servo_time, double lookahead_time) {
        check_ret(HRIF_StartServo(box_id_, rbt_id_, servo_time, lookahead_time));
    }

    void pushServoJ(const std::array<double,6>& joint_deg) {
        check_ret(HRIF_PushServoJ(box_id_, rbt_id_,
                                  joint_deg[0], joint_deg[1], joint_deg[2],
                                  joint_deg[3], joint_deg[4], joint_deg[5]));
    }

    void pushServoP(const std::array<double,6>& pose,
                    const std::array<double,6>* tcp=nullptr,
                    const std::array<double,6>* ucs=nullptr) {
        std::vector<double> vPose(pose.begin(), pose.end());
        std::vector<double> vTcp  = tcp ? std::vector<double>(tcp->begin(), tcp->end())
                                        : std::vector<double>{0,0,0,0,0,0};
        std::vector<double> vUcs  = ucs ? std::vector<double>(ucs->begin(), ucs->end())
                                        : std::vector<double>{0,0,0,0,0,0};
        check_ret(HRIF_PushServoP(box_id_, rbt_id_, vPose, vUcs, vTcp));
    }

    // 单位换算（与 Python 一致，Rx/Ry/Rz 度<->弧度）
    static std::array<double,6> convertDeg2Rad(std::array<double,6> pose) {
        const double k = 3.14159265358979323846 / 180.0;
        pose[3] *= k; pose[4] *= k; pose[5] *= k;
        return pose;
    }
    static std::array<double,6> convertRad2Deg(std::array<double,6> pose) {
        const double k = 180.0 / 3.14159265358979323846;
        pose[3] *= k; pose[4] *= k; pose[5] *= k;
        return pose;
    }

    // 兼容旧命名
    void serveON() { rbtEnable(); }

    // 基本属性访问
    const std::string& ip() const noexcept { return ip_; }
    unsigned short port() const noexcept { return port_; }
    unsigned int box_id() const noexcept { return box_id_; }
    unsigned int rbt_id() const noexcept { return rbt_id_; }
    bool connected() const noexcept { return connected_; }

private:
    std::string ip_;
    unsigned short port_;
    unsigned int box_id_;
    unsigned int rbt_id_;
    bool connected_;
};