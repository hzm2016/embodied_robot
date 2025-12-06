/******************************************************************************************
 * @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
 * @filename:  touchx.hpp
 * @brief:     Declaration of TouchX configuration and effect setting class
 * @author:    Jian Chen
 * @version:   2.0
 * @date:      2021.7.22
 *******************************************************************************************/

#pragma once

#include <HD/hd.h>
#include <HL/hl.h>
#include <HDU/hduVector.h>
#include <HDU/hduError.h>
#include <QH/QHHeadersGLUT.h>
#include <GL/glut.h>

class TouchX
{
public:
    // Initialize TouchX
    void init_touchx();

    // Start/End the haptic frame
    void start_frame();
    void end_frame();

    // Get the button state
    // Pressed: 1
    void get_hbutton();

    // Get the position
    void get_hposition();

    // Get angles of the device gimbal
    void get_hgimbalangle();

    // Get the velocity
    void get_hvelocity();

    // Set few haptic effect
    void get_haptic();

    // Get the joint angles
    void get_hjointangle();
    // Get the column-major transform of the device end-effector
    void get_htransform();

    // Return the button state and position
    HDint Cbuttons;
    HDint Lbuttons;
    hduVector3Dd hPosition;
    HDdouble LastZ;
    HDdouble hGimbalAngle[3];
    HDdouble hVelocity[3];
    HDdouble hJointAngle[3];
    HDdouble hTransform[16];
};
