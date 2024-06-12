import numpy as np
import os

delta_angle_x_file = os.path.join('./data', 'delta_x_angle.txt')
delta_angle_y_file = os.path.join('./data', 'delta_y_angle.txt')
delta_angle_z_file = os.path.join('./data', 'delta_z_angle.txt')

cnt = 0

def check_gyro_balance(x, y, z, OFF_BALANCE_THRESH=70):
    """
    x: list of x rotation angular velocity readings
    y: list of y rotation angular velocity readings
    z: list of z rotation angular velocity readings
    OFF_BALANCE_THRESH: threshold gyro value for off balance
    returns: bool (True if gyro is on balance)
    """    
    # trapezoid_x = 0
    # trapezoid_y = 0
    # trapezoid_z = 0

    # last_five_readings = \
    # [[x_cur, y_cur, z_cur] for x_cur, y_cur, z_cur 
    #     in zip(x[-5:], y[-5:], z[-5:])]    
    # delta = 1        
    # for i in range(len(last_five_readings)):
    #     x_angle = last_five_readings[i][0]
    #     y_angle = last_five_readings[i][1]
    #     z_angle = last_five_readings[i][2]
        
    #     if i == 0 or i == len(last_five_readings)-1:
    #         trapezoid_x += x_angle
    #         trapezoid_y += y_angle
    #         trapezoid_z += z_angle
    #     else:
    #         trapezoid_x += 2 * x_angle
    #         trapezoid_y += 2 * y_angle
    #         trapezoid_z += 2 * z_angle
    
    #     trapezoid_x *= delta/2
    #     trapezoid_y *= delta/2
    #     trapezoid_z *= delta/2

    trapezoid_x = np.trapz(x, np.arange(0, 5))
    trapezoid_y = np.trapz(y, np.arange(0, 5))
    trapezoid_z = np.trapz(z, np.arange(0, 5))

    # print(trapezoid_x, trapezoid_y, trapezoid_z)

    # save data
    global cnt
    if cnt == 0:
        with open(delta_angle_x_file, "w") as file:
            file.write(str(trapezoid_x) + '\n')

        with open(delta_angle_y_file, "w") as file:
            file.write(str(trapezoid_y) + '\n')

        with open(delta_angle_z_file, "w") as file:
            file.write(str(trapezoid_z) + '\n')
    else:
        with open(delta_angle_x_file, "a") as file:
            file.write(str(trapezoid_x) + '\n')

        with open(delta_angle_y_file, "a") as file:
            file.write(str(trapezoid_y) + '\n')

        with open(delta_angle_z_file, "a") as file:
            file.write(str(trapezoid_z) + '\n')

    cnt += 1


    # Now check if OFF BALANCE
    if (abs(trapezoid_x) > OFF_BALANCE_THRESH or abs(trapezoid_y) > OFF_BALANCE_THRESH or abs(trapezoid_z) > OFF_BALANCE_THRESH):
        return False
    
    return True

