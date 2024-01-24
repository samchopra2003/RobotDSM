from mpu6050 import mpu6050
import time

learning = False

def run_gyro(pipe):
    sensor = mpu6050(0x68)
    t = 0
    res = 100
    while True:
        if t % res == 0:
            global learning
            if pipe.poll():
                if pipe.recv() == 'Learning':   
                    learning = True
                elif pipe.recv() == 'Finished learning':
                    learning = False

            if not learning:
                try:
                    gyro_data = sensor.get_gyro_data()
                    x, y, z = gyro_data['x'], gyro_data['y'], gyro_data['z']
                    pipe.send((x, y, z))
                    #time.sleep(0.7)
                except OSError as e:
                    print("error occured: ", e)
        t += 1
