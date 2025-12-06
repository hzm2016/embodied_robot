/******************************************************************************************
 * @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
 * @filename:  touchx.cpp
 * @brief:     Defination of TouchX configuration and effect setting class.
 * @author:    Jian Chen
 * @version:   2.0
 * @date:      2021.7.22
 *******************************************************************************************/

#include "touchx.hpp"

// Initialize TouchX
void TouchX::init_touchx()
{
    // set TouchX
    HDErrorInfo error;
    HHD ghHD = hdInitDevice(HD_DEFAULT_DEVICE); // Handle to haptic device
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        hduPrintError(stderr, &error, "Failed to initialize haptic device");
        fprintf(stderr, "\nPress any key to quit.\n");
        (void)getchar();
        exit(-1);
    }

    // Create a haptic context for the device.  The haptic context maintains
    // the state that persists between frame intervals and is used for haptic rendering.
    HHLRC ghHLRC = hlCreateContext(ghHD); // Handle to haptic rendering context
    hlMakeCurrent(ghHLRC);
    hlDisable(HL_USE_GL_MODELVIEW);
    std::cout << "Initialize TouchX successfully!" << std::endl;
}

// Get the button state
void TouchX::get_hbutton()
{
    hdGetIntegerv(HD_CURRENT_BUTTONS, &Cbuttons);
    // std::cout << "button = " << Cbuttons << std::endl;
}

// Get the position
void TouchX::get_hposition()
{
    hlGetDoublev(HL_DEVICE_POSITION, hPosition);
    // std::cout << "postion = " << hPosition[0] << " " << hPosition[1] << " " << hPosition[2] << std::endl;
}

// Get angles of the device gimbal
void TouchX::get_hgimbalangle()
{
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, hGimbalAngle);
    // std::cout << "gimbal angles = " << hGimbalAngle[0] << " " << hGimbalAngle[1] << " " << hGimbalAngle[2] << std::endl;
}

// Get the velocity
void TouchX::get_hvelocity()
{
    hdGetDoublev(HD_CURRENT_VELOCITY, hVelocity);
    // std::cout << hVelocity[0] << " " << hVelocity[1] << " " << hVelocity[2] << std::endl;
    // std::cout << hVelocity[2] << std::endl;
}

void TouchX::get_hjointangle()
{
    hdGetDoublev(HD_CURRENT_JOINT_ANGLES, hJointAngle);
    // std::cout << "joint angle = " << hJointAngle[0] << hJointAngle[1] << hJointAngle[2] << std::endl;
}

void TouchX::get_htransform()
{
    hdGetDoublev(HD_CURRENT_TRANSFORM, hTransform);
    // std::cout << "transform = ";
    // for (int i = 0; i < 16; i++)
    // {
    //     std::cout << hTransform[i];
    // }
    // std::cout << std::endl;
}
void TouchX::get_haptic()
{
    // Define four types of effects
    // static const HLuint Effect_xyz = hlGenEffects(1);
    static const HLuint Effect_y_Gra = hlGenEffects(2);
    // static const HLuint Effect_xyz_border = hlGenEffects(3);
    // static const HLuint Effect_z_vel = hlGenEffects(4);

    // Direction, positive(p),negative(n)
    HDdouble direction_x_p[3] = {1, 0, 0};  // +x
    HDdouble direction_x_n[3] = {-1, 0, 0}; // -x
    HDdouble direction_y_p[3] = {0, 1, 0};  // +y
    HDdouble direction_y_n[3] = {0, -1, 0}; // -y
    HDdouble direction_z_p[3] = {0, 0, 1};  // +z
    HDdouble direction_z_n[3] = {0, 0, -1}; // -z

    // position Z at last frame
    float last_position_z = hPosition[2];

    // // Simulate resistance to XYZ direction movement in body fluids
    // hlEffectd(HL_EFFECT_PROPERTY_GAIN, 0.2);
    // hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, 1);
    // hlStartEffect(HL_EFFECT_VISCOUS, Effect_xyz);

    //  Y direction gravity compensation
    hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_y_n);
    hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, 0.6);
    hlStartEffect(HL_EFFECT_CONSTANT, Effect_y_Gra);

    // XYZ direction boundary
    // Touch the boundary will produce elastic force
    // if (abs(hPosition[0]) >= 70)
    // {
    //     if (hPosition[0] >= 0)
    //     {
    //         hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_x_n);
    //     }

    //     else
    //     {
    //         hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_x_p);
    //     }

    //     float MAGNITUDE_xyz = 0.03 * (abs(hPosition[0]) - 70);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, MAGNITUDE_xyz);
    //     hlStartEffect(HL_EFFECT_CONSTANT, Effect_xyz_border);
    // }

    // if (hPosition[1] >= 60)
    // {
    //     hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_y_n);
    //     float MAGNITUDE_xyz = 0.03 * (hPosition[1] - 60);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, MAGNITUDE_xyz);
    //     hlStartEffect(HL_EFFECT_CONSTANT, Effect_xyz_border);
    // }

    // if (hPosition[1] <= -40)
    // {
    //     hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_y_p);
    //     float MAGNITUDE_xyz = 0.05 * (abs(hPosition[1]) - 40);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, MAGNITUDE_xyz);
    //     hlStartEffect(HL_EFFECT_CONSTANT, Effect_xyz_border);
    // }

    // if (hPosition[2] <= 99)
    // {
    //     hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_z_p);
    //     float MAGNITUDE_xyz = 0.1 * (abs(hPosition[2]) -99);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, MAGNITUDE_xyz);
    //     hlStartEffect(HL_EFFECT_CONSTANT, Effect_xyz_border);
    // }

    // if (hPosition[2] >= 101)
    // {
    //     hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_z_n);
    //     float MAGNITUDE_xyz = 0.1 * (abs(hPosition[2]) - 101);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, MAGNITUDE_xyz);
    //     hlStartEffect(HL_EFFECT_CONSTANT, Effect_xyz_border);
    // }

    // hlUpdateEffect(Effect_xyz_border);

    // // Additional resistance to Z direction movement
    // float diffZ = 0.1;
    // float ratio = 0.05;
    // float velo = 0;
    // velo = abs(hPosition[2] - last_position_z) * ratio;
    // if (velo > 0.1)
    // {
    //     velo = 0.1;
    // }
    // if ((hPosition[2] - last_position_z) >= diffZ)
    // {
    //     hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_z_n);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, velo);
    // }
    // if ((hPosition[2] - last_position_z) <= -diffZ)
    // {
    //     hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, direction_z_p);
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, velo);
    // }
    // if (((hPosition[2] - last_position_z) < diffZ) && ((hPosition[2] - last_position_z) > -diffZ))
    // {
    //     hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, 0);
    // }
    // hlStartEffect(HL_EFFECT_CONSTANT, Effect_z_vel);
    // hlUpdateEffect(Effect_z_vel);
    // last_position_z = hPosition[2];

    // Check for any errors.
    HLerror error;
    while (HL_ERROR(error = hlGetError()))
    {
        if (error.errorCode == HL_DEVICE_ERROR)
        {
            hduPrintError(stderr, &error.errorInfo, "Error during haptic rendering\n");
        }
    }
}

// Start the haptic frame
void TouchX::start_frame()
{
    hlBeginFrame();
}

// End the haptic frame
void TouchX::end_frame()
{
    hlEndFrame();
}