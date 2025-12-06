#include <force_haptron.hpp>
#include <string>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <iomanip>
CLinuxSerial::CLinuxSerial(UINT portNo /*=1*/, UINT baudRate /*= 115200*/)
{
    InitPort(portNo, baudRate);
}

CLinuxSerial::~CLinuxSerial()
{
    ClosePort();
}

bool CLinuxSerial::OpenPort(UINT portNo)
{
    char portStr[20];
    sprintf(portStr, "/dev/ttyUSB%d", portNo);
    m_iSerialID = open(portStr, O_RDWR); // block mode
    if (m_iSerialID < 0)
    {
        printf("open serial error!\n");
        return false;
    }
    return true;
}

void CLinuxSerial::ClosePort()
{
    close(m_iSerialID);
    m_iSerialID = -1;
}

bool CLinuxSerial::InitPort(UINT portNo /*= 1*/, UINT baudRate /*= 115200*/)
{
    char data_bits = 8;
    char parity_bits = 'N';
    char stop_bits = 1;
    ClosePort();
    bool res = OpenPort(portNo);
    if (!res)
        return false;

    int st_baud[] =
        {
            B4800,
            B9600,
            B19200,
            B38400,
            B57600,
            B115200,
            B230400,
            B1000000,
            B1152000,
            B3000000,
        };
    int std_rate[] =
        {
            4800,
            9600,
            19200,
            38400,
            57600,
            115200,
            230400,
            1000000,
            1152000,
            3000000,
        };

    int i, j;
    struct termios newtio, oldtio;
    /* save current port parameter */
    if (tcgetattr(m_iSerialID, &oldtio) != 0)
    {
        printf("%s,%d:ERROR\n", __func__, __LINE__);
        return -1;
    }
    bzero(&newtio, sizeof(newtio));

    /* config the size of char */
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;

    /* config data bit */
    switch (data_bits)
    {
    case 7:
        newtio.c_cflag |= CS7;
        break;
    case 8:
        newtio.c_cflag |= CS8;
        break;
    }
    /* config the parity bit */
    switch (parity_bits)
    {
        /* odd */
    case 'O':
    case 'o':
        newtio.c_cflag |= PARENB;
        newtio.c_cflag |= PARODD;
        break;
        /* even */
    case 'E':
    case 'e':
        newtio.c_cflag |= PARENB;
        newtio.c_cflag &= ~PARODD;
        break;
        /* none */
    case 'N':
    case 'n':
        newtio.c_cflag &= ~PARENB;
        break;
    }
    /* config baudrate */
    j = sizeof(std_rate) / 4;
    for (i = 0; i < j; i++)
    {
        if (std_rate[i] == baudRate)
        {
            /* set standard baudrate */
            cfsetispeed(&newtio, st_baud[i]);
            cfsetospeed(&newtio, st_baud[i]);
            break;
        }
    }
    /* config stop bit */
    if (stop_bits == 1)
        newtio.c_cflag &= ~CSTOPB;
    else if (stop_bits == 2)
        newtio.c_cflag |= CSTOPB;

    /* config waiting time & min number of char */
    newtio.c_cc[VTIME] = 1;
    newtio.c_cc[VMIN] = 1;

    /* using the raw data mode */
    newtio.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    newtio.c_oflag &= ~OPOST;

    /* flush the hardware fifo */
    tcflush(m_iSerialID, TCIFLUSH);

    /* activite the configuration */
    if ((tcsetattr(m_iSerialID, TCSANOW, &newtio)) != 0)
    {
        printf("%s,%d:ERROR\n", __func__, __LINE__);
        return -1;
    }
    return true;
}

UINT CLinuxSerial::ReadData(UCHAR *data, UINT length)
{
    if (m_iSerialID < 0)
        return 0;
    int ret = -1;
    if (data == NULL)
        return 0;

    ret = read(m_iSerialID, data, length);

    return ret;
}

UINT CLinuxSerial::WriteData(UCHAR *data, UINT length)
{
    if (m_iSerialID < 0)
        return 0;
    return write(m_iSerialID, data, length);
}

UINT CLinuxSerial::GetBytesInCom()
{
    return 0;
}

unsigned char CLinuxSerial::CheckSum(unsigned char *buf, const int len)
{ // buf为数组，len为数组长度
    unsigned char ret = 0;
    for (int i = 0; i < len; i++)
    {
        ret += *(buf++);
    }
    return ret;
}

void CLinuxSerial::Sensor()
{
    if (InitPort(0, 115200))
    {
        unsigned char buf[31];
        unsigned char data[24];
        ReadData(buf, 31);
        if (buf[0] == 0xaa && buf[1] == 0x55)
        {
            for (int data_buf = 0; data_buf < 24; data_buf++)
            {
                data[data_buf] = buf[data_buf + 6];
            }
            unsigned char checkbit = CheckSum(data, 24);
            if (checkbit == buf[30])
            {
                for (int index = 0; index < 6; index++)
                {
                    unsigned char temp[4] = {data[4 * index], data[4 * index + 1], data[4 * index + 2], data[4 * index + 3]};
                    float *ret = (float *)temp;
                    sensor[index] = *ret;
                }
            }
        }
    }
}

// int main()
// {
//     float sensor_force_offset[3] = {0};
//     float sensor_torque_offset[3] = {0};
//     int iniCount = 0;

//     CLinuxSerial com(0, 115200);
//     while (true)
//     {
//         if (com.InitPort(0, 115200))
//         {
//             unsigned char data[] = {0x01, 0x04, 0x00, 0x38, 0x00, 0x0C, 0x71, 0xC2};

//             UINT result = com.WriteData(data, sizeof(data));

//             if (result > 0)
//             {
//                 // std::cout << "Data written successfully: " << result << " bytes." << std::endl;
//                 unsigned char buf[31];
//                 com.ReadData(buf, 31);
//                 // std::cout << buf << std::endl;
//                 if ((buf[0] == 0x01) && (buf[1] == 0x04) && (buf[2] == 0x18))
//                 {
//                     // RCLCPP_INFO(this->get_logger(), "ReadData");
//                     // printf("%#02X %#02X %#02X\n", buf[0], buf[1], buf[2]);

//                     unsigned char fx_tmp[4] = {buf[6], buf[5], buf[4], buf[3]};
//                     unsigned char fy_tmp[4] = {buf[10], buf[9], buf[8], buf[7]};
//                     unsigned char fz_tmp[4] = {buf[14], buf[13], buf[12], buf[11]};
//                     unsigned char tx_tmp[4] = {buf[18], buf[17], buf[16], buf[15]};
//                     unsigned char ty_tmp[4] = {buf[22], buf[21], buf[20], buf[19]};
//                     unsigned char tz_tmp[4] = {buf[26], buf[25], buf[24], buf[23]};
//                     float fx = *((float *)(fx_tmp));
//                     float fy = *((float *)(fy_tmp));
//                     float fz = *((float *)(fz_tmp));
//                     float tx = *((float *)(tx_tmp));
//                     float ty = *((float *)(ty_tmp));
//                     float tz = *((float *)(tz_tmp));

//                     constexpr auto INIT_NUM = 50;
//                     if (iniCount < INIT_NUM)
//                     {
//                         sensor_force_offset[0] += fx / INIT_NUM;
//                         sensor_force_offset[1] += fy / INIT_NUM;
//                         sensor_force_offset[2] += fz / INIT_NUM;
//                         sensor_torque_offset[0] += tx / INIT_NUM;
//                         sensor_torque_offset[1] += ty / INIT_NUM;
//                         sensor_torque_offset[2] += tz / INIT_NUM;
//                         iniCount = iniCount + 1;
//                     }
//                     else
//                     {
//                         std::vector<float> sensor_pressure(6, 0); // 力和力矩

//                         sensor_pressure[0] = fx - sensor_force_offset[0];
//                         sensor_pressure[1] = fy - sensor_force_offset[1];
//                         sensor_pressure[2] = fz - sensor_force_offset[2];

//                         sensor_pressure[3] = tx - sensor_force_offset[3];
//                         sensor_pressure[4] = ty - sensor_force_offset[4];
//                         sensor_pressure[5] = tz - sensor_force_offset[5];
//                         std::cout << std::fixed << std::setprecision(4); // Set precision to 4 decimal places

//                         for (auto i = 0; i < 6; ++i)
//                         {
//                             std::cout << sensor_pressure[i];
//                             if (i < 5)
//                             {
//                                 std::cout << ", ";
//                             }
//                         }
//                         usleep(5000);
//                         std::cout << std::endl;
//                     }
//                 }
//             }
//         }
//     }

//     return 0;
// }