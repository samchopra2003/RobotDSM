from mpu6050 import mpu6050
import time
import board
# import adafruit_mpu6050

learning = False

def run_gyro(pipe):
    sensor = mpu6050(0x68)

    # i2c = board.I2C()  # uses board.SCL and board.SDA
    # # mpu = adafruit_mpu6050.MPU6050(i2c, address=0x68)
    # mpu = adafruit_mpu6050.MPU6050(i2c)

    t = 0
    res = 10000
    # res = 1
    st_time = 10000

    while True:
        if t % res == 0 and t > st_time:
            # global learning
            # if pipe.poll():
            #     cmd = pipe.recv()
            #     if cmd == 'Learning':   
            #         learning = True
            #     elif cmd == 'Finished learning':
            #         learning = False

            # if not learning:
            #     try:
            #         x, y, z = mpu.gyro
            #         print("Gyro X:%.2f, Y: %.2f, Z: %.2f degrees/s"%(x, y, z))
            #         pipe.send((x, y, z))
            #         time.sleep(2.0)
            #     except OSError as e:
            #         time.sleep(4.0)
            #         print("error occured: ", e)
            #         i2c = board.I2C()  # uses board.SCL and board.SDA
            #         mpu = adafruit_mpu6050.MPU6050(i2c)
            #     except ValueError as e:
            #         time.sleep(4.0)
            #         print("error occured: ", e)
            #         i2c = board.I2C()  # uses board.SCL and board.SDA
            #         mpu = adafruit_mpu6050.MPU6050(i2c)


                try:
                    gyro_data = sensor.get_gyro_data()
                    x, y, z = gyro_data['x'], gyro_data['y'], gyro_data['z']
                    # pipe.send((x, y, z))
                    time.sleep(0.3)
                    print(x, y, z)
                except OSError as e:
                    time.sleep(0.3)
                    print("error occured: ", e)
                    try:
                        sensor = mpu6050(0x68)
                    except:
                        # time.sleep(1.0)
                        sensor = mpu6050(0x68)
            # pass

        t += 1
