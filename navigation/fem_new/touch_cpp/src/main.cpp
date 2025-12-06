/******************************************************************************************
 * @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
 * @filename:  main.cpp
 * @brief:     main function
 * @author:    Jian Chen, Yuanrui Huang
 * @version:   2.0
 * @date:      2021.7.22
 *******************************************************************************************/

#include "motor_comm.hpp"
#include "touchx.hpp"
#include "kinematics.hpp"
#include "arm_movementZ.hpp"
#include "Parameters.hpp"
#include "RobotCmu.hpp"
#include <thread>
#include "joy.hpp"
#include "get_sensor_data.hpp"
#include <fstream>

using namespace xmate;
using CartesianControl = std::function<CartesianPose(RCI::robot::RobotState robot_state)>;
std::array<double, 16> lastpose;
std::array<double, 3> yaxis;
// for simulator robot position
double init_robot = 0;
std::array<double, 16> toolposition;

double cvalue = 0;
double insert_dis = 0;
bool stopFlag = false;
bool leftbutton = true;
bool rightbutton = true;
double leftaxis = leftaxis_zero;
double rightaxis = leftaxis_zero;
double add_axis = 0;
bool sheathSteering = false;
bool endoSteering = false;
bool sheathInsert = false;
bool endoInsert = false;
bool endoRESET = false;
bool sheathRESET = false;
double xydata[2] = {0.0, 0.0};
UCHAR presInfo[5] = {};
double sheathTheta = 0;
double sheathPhi = 0;
double endoTheta = 0;
double endoPhi = 0;

// Set the callback function used to control robot.
CartesianControl cart_position_callback = [&](RCI::robot::RobotState robot_state) -> CartesianPose
{
    CartesianPose output{};
    output.toolTobase_pos_c = lastpose;
    // set robot y initial position
    if (toolposition[7] != 0 && init_robot == 0)
        init_robot = toolposition[7];
    /*
    output.toolTobase_pos_c[3] =  output.toolTobase_pos_c[3] + cvalue * zaxis[0];
    output.toolTobase_pos_c[7] =  output.toolTobase_pos_c[7] +  cvalue *zaxis[1];
    output.toolTobase_pos_c[11] =  output.toolTobase_pos_c[11] + cvalue * zaxis[2];
    */
    toolposition = robot_state.toolTobase_pos_m;
    output.toolTobase_pos_c[3] = output.toolTobase_pos_c[3] - cvalue * yaxis[0];
    output.toolTobase_pos_c[7] = output.toolTobase_pos_c[7] - cvalue * yaxis[1];
    output.toolTobase_pos_c[11] = output.toolTobase_pos_c[11] - cvalue * yaxis[2];
    lastpose = output.toolTobase_pos_c;
    // std::cout << cvalue << std::endl;
    return output;
};
void armInsert()
{
    // Initialize Robot
    std::string ipaddr = "192.168.0.160";
    uint16_t port = 1337;

    xmate::Robot robot(ipaddr, port, xmate::XmateType::XMATE3_PRO);
    std::cout << "BUILD CONNECT" << std::endl;
    sleep(1);

    std::array<double, 3> loadcentor = {0, 0, 85};
    std::array<double, 9> InertialMat = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    robot.setFilters(100, 100, 100);
    std::cout << "SUCCESSFULLY SETUP" << std::endl;
    robot.setLoad(3, loadcentor, InertialMat);

    robot.automaticErrorRecovery();
    std::cout << "RECOVERY CHECKED" << std::endl;

    // sleep(2);

    // const double PI = 3.14159;
    // std::array<double, 7> q_init;
    // std::array<double, 7> q_drag = {{0 * PI / 18, 25*PI/180, 0, 5.5 * PI / 18, 0, 100*PI / 180, 0}};
    // q_init = robot.receiveRobotState().q;
    // MOVEJ(0.2, q_init, q_drag, robot);
    // 开启运动模式和控制模式
    robot.startMove(RCI::robot::StartMoveRequest::ControllerMode::kCartesianPosition,
                    RCI::robot::StartMoveRequest::MotionGeneratorMode::kCartesianPosition);

    // 使用全局变量存储机械臂末端朝向
    lastpose = robot.receiveRobotState().toolTobase_pos_m;
    // zaxis = {{lastpose[2],lastpose[6],lastpose[10]}};
    yaxis = {{lastpose[1], lastpose[5], lastpose[9]}};

    while (!stopFlag)
    {
        robot.Control(cart_position_callback);

        std::cout << "Robot 6" << std::endl;
    }
    std::cout << "Robot Stop" << std::endl;
    robot.stopMove();
    robot.setMotorPower(0);
}

static std::string getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    static const int MAX_BUFFER_SIZE = 128;
    char timestamp_str[MAX_BUFFER_SIZE];
    time_t sec = static_cast<time_t>(tv.tv_sec);
    int ms = static_cast<int>(tv.tv_usec) / 1000;

    struct tm tm_time;
    localtime_r(&sec, &tm_time);
    static const char *formater = "%02d:%02d:%02d.%03d";
    int wsize = snprintf(timestamp_str, MAX_BUFFER_SIZE, formater,
                         tm_time.tm_hour, tm_time.tm_min, tm_time.tm_sec, ms);

    timestamp_str[std::min(wsize, MAX_BUFFER_SIZE - 1)] = '\0';
    return std::string(timestamp_str);
}
void get_nowtime(std::string &CurrentTime)
{
    time_t time_seconds = time(0);

    struct tm *p = new tm;

    localtime_r(&time_seconds, p);
    int year = p->tm_year + 1900, month = p->tm_mon + 1, day = p->tm_mday;
    int hour = p->tm_hour, minute = p->tm_min, second = p->tm_sec;
    delete p;
    CurrentTime = std::to_string(year) +
                  std::string(2 - std::to_string(month).length(), '0') + std::to_string(month) +
                  std::string(2 - std::to_string(day).length(), '0') + std::to_string(day) +
                  "_" +
                  std::string(2 - std::to_string(hour).length(), '0') + std::to_string(hour) +
                  std::string(2 - std::to_string(minute).length(), '0') + std::to_string(minute) +
                  std::string(2 - std::to_string(second).length(), '0') + std::to_string(second);
}

int main(int argc, char **argv)
{

    // Initialize swing motors
    SerialSlaveCtrl InsertMotor("/dev/ttyUSB", InsertPortNo, MotorBaudRate);
    InsertMotor.set_swing_zero_position(InsetMot);
    InsertMotor.close_port();

    // Initialize TouchX
    TouchX hDev;
    hDev.init_touchx();
    hDev.get_hbutton();

    // thread: robot arm move L
    std::thread t_arm(armInsert);
    t_arm.detach();
    // thread: get thrusmaster joy data
    joy joyhandle;
    std::thread t_joy(std::bind(&joy::get_joy, &joyhandle, &leftbutton, &rightbutton, &sheathSteering, &endoSteering,
                                &sheathInsert, &endoInsert, &sheathRESET, &endoRESET, &leftaxis, &rightaxis, &add_axis));
    t_joy.detach();

    // udp to ECAT
    int sock_fd;
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0)
    {
        perror("socket");
        exit(1);
    }
    struct sockaddr_in addr_serv;
    int len;
    memset(&addr_serv, 0, sizeof(addr_serv));
    addr_serv.sin_family = AF_INET;
    addr_serv.sin_addr.s_addr = inet_addr(DEST_IP_ADDRESS);
    addr_serv.sin_port = htons(DEST_PORT);
    len = sizeof(addr_serv);

    char steering_pos[100];
    for (int i = 0; i < 100; i++)
    {
        steering_pos[i] = '0';
    }

    std::string CurrentTime;
    get_nowtime(CurrentTime);
    std::string filename = "/home/cas/Documents/micronb-control/docs/operations_" + CurrentTime + ".csv";
    std::ofstream outfile;
    outfile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);

    // udp 2 fem
    char command2fem[20];
    int sock_fem;
    sock_fem = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fem < 0)
    {
        perror("socket");
        exit(1);
    }
    struct sockaddr_in addr_serv_fem;
    int len_fem;
    memset(&addr_serv_fem, 0, sizeof(addr_serv_fem));
    addr_serv_fem.sin_family = AF_INET;
    addr_serv_fem.sin_addr.s_addr = inet_addr(FEM_IP_ADDRESS);
    addr_serv_fem.sin_port = htons(PORT2FEM);
    len_fem = sizeof(addr_serv_fem);

    while (!stopFlag)
    {

        stopFlag = !(leftbutton && rightbutton);

        // get TouchX states
        hDev.start_frame();
        hDev.get_hposition();
        // std::cout << hDev.hPosition[0] << " " << hDev.hPosition[1] << " " << hDev.hPosition[2] << std::endl;
        hDev.get_hgimbalangle(); //  double 3
        hDev.get_hbutton();      //  int 1
        hDev.get_htransform();   //  double 16
        hDev.get_hjointangle();  //  double 3
        hDev.get_haptic();
        hDev.end_frame();
        // std::cout << hDev.Cbuttons << std::endl;

        Kinematics SteerCtrl;
        int positionM[3];
        std::string steering_mode;
        int steer_mode;
        if (endoSteering == true) // endoscope steering
        {
            steering_mode = "EndoSteering";
            steer_mode = 1;
            SteerCtrl.get_motor_pos(Catlength, deltaradius, thetamax,
                                    hDev.hPosition, hDev.hGimbalAngle, positionM, steer_mode);
            sprintf(steering_pos, "%d,%d,%d,%d\n", steer_mode, positionM[0], positionM[1], positionM[2]);
            // std::cout << steering_pos << std::endl;
            int send_num;
            send_num = sendto(sock_fd, steering_pos, strlen(steering_pos), 0, (struct sockaddr *)&addr_serv, len);
            // usleep(10000);
            if (send_num < 0)
            {
                std::cout << "send error" << std::endl;
            }

            endoTheta = SteerCtrl.theta;
            endoPhi = SteerCtrl.phi;
        }
        else if (sheathSteering == true) // sheath steering
        {
            steering_mode = "SheathSteering";
            steer_mode = 2;
            SteerCtrl.get_motor_pos(SHCatlength, SHdeltaradius, SHthetamax,
                                    hDev.hPosition, hDev.hGimbalAngle, positionM, steer_mode);
            sprintf(steering_pos, "%d,%d,%d,%d\n", steer_mode, positionM[0], positionM[1], positionM[2]);
            // std::cout << steering_pos << std::endl;
            int send_num;
            send_num = sendto(sock_fd, steering_pos, strlen(steering_pos), 0, (struct sockaddr *)&addr_serv, len);
            // usleep(10000);
            if (send_num < 0)
            {
                std::cout << "send error" << std::endl;
            }
            sheathTheta = SteerCtrl.theta;
            sheathPhi = SteerCtrl.phi;
        }
        else if (sheathSteering == false || endoSteering == false) // Non-steering mode
        {
            steering_mode = "NoSteering";
            steer_mode = 0;
            if (endoRESET) // endoscope go back to zero pos
            {
                steering_mode = "EndoRESET";
                sprintf(steering_pos, "%d,%d,%d,%d\n", 1, 0, 0, 0);
                int send_num;
                send_num = sendto(sock_fd, steering_pos, strlen(steering_pos), 0, (struct sockaddr *)&addr_serv, len);

                if (send_num < 0)
                {
                    std::cout << "send error" << std::endl;
                }
                endoTheta = 0;
                endoPhi = 0;
            }
            else if (sheathRESET) // sheath go back to zero pos
            {
                steering_mode = "SheathRESET";
                sprintf(steering_pos, "%d,%d,%d,%d\n", 2, 0, 0, 0);
                int send_num;
                send_num = sendto(sock_fd, steering_pos, strlen(steering_pos), 0, (struct sockaddr *)&addr_serv, len);

                if (send_num < 0)
                {
                    std::cout << "send error" << std::endl;
                }
                sheathTheta = 0;
                sheathPhi = 0;
            }
        }

        /* Using thrustmaster joystick to  insert */
        std::string insert_mode;
        if ((sheathInsert == false) && (endoInsert == false)) // none of sheath or endoscope insert
        {
            insert_mode = "NoInsert";
            SerialSlaveCtrl InsertMotor("/dev/ttyUSB", InsertPortNo, MotorBaudRate);
            InsertMotor.set_swing_motor_position(InsetMot, SMdataN, SMSPtype, 0, 0,
                                                 SMstart_rate, SMstop_rate, SMcurrent, SMtrigger);

            InsertMotor.close_port();

            SerialSlaveCtrl umot("/dev/ttyUSB", UmotPortNo, MotorBaudRate);
            umot.disableUmot(umot_high);
            umot.disableUmot(umot_low);
            umot.close_port();
            cvalue = 0;
        }
        else if (sheathInsert) // sheath insert
        {
            if (abs(rightaxis) < 5000)
            {
                rightaxis = 0;
            }

            insert_mode = "SheathInsert";
            ArmMoveZAxis robotArmMove;
            double insertSpd;
            insertSpd = robotArmMove.insert_speed(0 - rightaxis) * 0.5;
            SerialSlaveCtrl InsertMotor("/dev/ttyUSB", InsertPortNo, MotorBaudRate);
            InsertMotor.set_swing_motor_position(InsetMot, SMdataN, SMSPtype, 0, -insertSpd * 67,
                                                 SMstart_rate, SMstop_rate, SMcurrent, SMtrigger);

            InsertMotor.close_port();

            // Sending the speed to robot arm.
            insertSpd = 0.6 * insertSpd;
            cvalue = insertSpd * M_PI * 12.29 * 0.001 * 0.001 / (spdRatio * 30);
        }
        else if (endoInsert) // endoscope insert
        {
            if (abs(add_axis) < 5000)
            {
                add_axis = 0;
            }
            insert_mode = "EndoInsert";
            ArmMoveZAxis robotArmMove;
            double insertSpd;

            insertSpd = robotArmMove.insert_speed(0 - add_axis) * 0.15;
            std::cout << insertSpd << std::endl;
            SerialSlaveCtrl umot("/dev/ttyUSB", UmotPortNo, MotorBaudRate);
            if (insertSpd > 0)
            {
                umot.setUmotSpd(umot_high, insertSpd);
                umot.setUmotSpd(umot_low, -insertSpd * 0.6);
            }
            else if (insertSpd < 0)
            {
                umot.setUmotSpd(umot_high, insertSpd * 0.3);
                umot.setUmotSpd(umot_low, -insertSpd * 0.3 * 0.6);
            }
            umot.enableUmot(umot_high);
            umot.enableUmot(umot_low);
            umot.close_port();

            cvalue = 0;
        }

        std::string op_time;
        op_time = getCurrentTime();
        outfile << op_time << ", "
                << steering_mode << ", "
                << insert_mode << ", "
                << "endoTheta = " << endoTheta << ", "
                << "endoPhi = " << endoPhi << ", "
                << "sheathTheta = " << sheathTheta << ", "
                << "sheathPhi = " << sheathPhi
                << std::endl;
        int send_fem_num;
        double sim_sheathinsert = init_robot * 1000 - toolposition[7] * 1000;
        sprintf(command2fem, "%11.6f,%11.6f,%11.6f,%1d\n", hDev.hPosition[0], hDev.hPosition[2], sim_sheathinsert, steer_mode);
        // std::cout << command2fem << std::endl;
        send_fem_num = sendto(sock_fem, command2fem, strlen(command2fem), 0, (struct sockaddr *)&addr_serv_fem, len_fem);
        if (send_fem_num < 0)
        {
            std::cout << "send error" << std::endl;
        }
    }

    // steering return to pos 0
    sprintf(steering_pos, "%d,%d,%d,%d\n", 0, 0, 0, 0);
    int send_num1;
    send_num1 = sendto(sock_fd, steering_pos, strlen(steering_pos), 0, (struct sockaddr *)&addr_serv, len);
    sprintf(steering_pos, "%d,%d,%d,%d\n", 1, 0, 0, 0);
    int send_num2;
    send_num2 = sendto(sock_fd, steering_pos, strlen(steering_pos), 0, (struct sockaddr *)&addr_serv, len);

    if (send_num1 < 0 || send_num2 < 0)
    {
        std::cout << "send error" << std::endl;
    }
    outfile << "RETURN TO ZERO POSITION" << std::endl;
    outfile.close();
    return 0;
}