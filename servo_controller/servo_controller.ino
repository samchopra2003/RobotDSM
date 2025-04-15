#include "PCA9685servo.h"

// Servo identifiers
uint8_t BLK = 0, BLS = 1, BRS = 6, BRK = 7;
uint8_t FLS = 14, FLK = 15, FRK = 8, FRS = 9, Head = 12;

// Constants
const static int SERVOMIN = 150, SERVOMAX = 600, USMIN = 600, USMAX = 2400;
const static int DEFAULT_FREQ = 50, ANGLE_RANGE = 180;
#define FRAME_NUM 54 * 8

const static int8_t wkF[FRAME_NUM] = 
                    { 9, 49, 67, 38, 24, 20, 27, 14,
  8, 50, 68, 39, 28, 21, 22, 14,
 10, 51, 70, 41, 26, 22, 18, 14,
 12, 52, 69, 42, 24, 24, 11, 14,
 14, 52, 63, 44, 22, 26,  1, 15,
 16, 53, 53, 45, 21, 27, -2, 15,
 18, 53, 41, 46, 20, 29, -1, 15,
 21, 54, 26, 47, 18, 30,  6, 16,
 22, 54, 23, 48, 18, 32,  9, 17,
 25, 54, 20, 49, 16, 34, 12, 18,
 26, 54, 17, 51, 16, 37, 17, 18,
 28, 54, 14, 52, 15, 39, 22, 19,
 30, 52, 11, 54, 14, 45, 27, 19,
 32, 54, 11, 54, 14, 44, 29, 20,
 33, 58, 13, 55, 15, 36, 27, 21,
 34, 61, 16, 56, 15, 31, 24, 23,
 36, 64, 18, 56, 14, 24, 23, 24,
 38, 66, 20, 57, 14, 20, 22, 26,
 39, 67, 22, 57, 14, 16, 21, 28,
 41, 64, 24, 57, 14,  5, 20, 30,
 42, 55, 26, 57, 14, -1, 19, 32,
 44, 44, 28, 57, 15, -3, 18, 35,
 45, 30, 29, 57, 15,  1, 18, 38,
 46, 21, 31, 57, 15,  5, 17, 40,
 47, 19, 32, 56, 16,  9, 17, 43,
 48, 16, 35, 57, 17, 12, 16, 44,
 49, 12, 37, 62, 18, 17, 14, 35,
 49,  9, 39, 66, 20, 24, 14, 29,
 50,  8, 40, 68, 21, 28, 14, 23,
 51, 10, 42, 70, 22, 26, 14, 19,
 52, 12, 43, 70, 24, 24, 15, 17,
 52, 14, 44, 67, 26, 22, 15,  5,
 53, 16, 46, 59, 27, 21, 15, -2,
 53, 18, 47, 47, 29, 20, 16, -2,
 54, 21, 48, 34, 30, 18, 16,  1,
 54, 22, 49, 24, 32, 18, 17,  6,
 54, 25, 50, 21, 34, 16, 18, 10,
 54, 26, 51, 19, 37, 16, 19, 12,
 54, 28, 52, 15, 39, 15, 20, 19,
 52, 30, 54, 12, 45, 14, 19, 24,
 54, 32, 55, 12, 44, 14, 20, 27,
 58, 33, 55, 11, 36, 15, 22, 29,
 61, 34, 56, 14, 31, 15, 24, 26,
 64, 36, 56, 17, 24, 14, 25, 24,
 66, 38, 57, 18, 20, 14, 27, 23,
 67, 39, 57, 21, 16, 14, 29, 21,
 64, 41, 57, 23,  5, 14, 31, 20,
 55, 42, 57, 24, -1, 14, 33, 20,
 44, 44, 57, 26, -3, 15, 36, 19,
 30, 45, 57, 28,  1, 15, 39, 18,
 21, 46, 56, 30,  5, 15, 42, 17,
 19, 47, 56, 32,  9, 16, 45, 17,
 16, 48, 59, 33, 12, 17, 41, 17,
 12, 49, 64, 35, 17, 18, 33, 16};


const static int8_t crF[34*8] = {42,  73,  83,  75, -43, -42, -49, -41,
  37,  75,  78,  77, -41, -41, -50, -41,
  36,  78,  73,  80, -38, -41, -50, -41,
  37,  81,  68,  82, -36, -41, -50, -40,
  41,  83,  62,  85, -36, -40, -48, -40,
  42,  88,  57,  85, -36, -40, -47, -39,
  45,  92,  51,  88, -37, -44, -44, -38,
  48,  91,  50,  90, -38, -45, -41, -37,
  51,  89,  55,  91, -39, -48, -40, -36,
  53,  84,  55,  93, -40, -49, -40, -35,
  56,  80,  58,  99, -40, -50, -41, -37,
  59,  75,  61, 101, -41, -50, -41, -40,
  62,  69,  64, 101, -41, -50, -41, -42,
  64,  64,  65,  99, -41, -49, -41, -44,
  67,  58,  68,  95, -42, -48, -42, -46,
  70,  53,  71,  92, -42, -47, -42, -47,
  73,  48,  74,  88, -42, -45, -41, -48,
  74,  42,  75,  83, -41, -43, -41, -49,
  77,  37,  78,  78, -41, -41, -41, -50,
  80,  36,  81,  73, -41, -38, -41, -50,
  82,  37,  83,  68, -40, -36, -40, -50,
  85,  41,  85,  62, -40, -36, -40, -48,
  89,  42,  87,  57, -41, -36, -39, -47,
  93,  45,  89,  51, -45, -37, -38, -44,
  92,  48,  91,  50, -47, -38, -37, -41,
  88,  51,  93,  55, -48, -39, -36, -40,
  83,  53,  96,  55, -49, -40, -35, -40,
  78,  56, 100,  58, -50, -40, -38, -41,
  73,  59, 102,  61, -50, -41, -42, -41,
  68,  62,  99,  64, -50, -41, -43, -41,
  62,  64,  96,  65, -48, -41, -45, -41,
  57,  67,  93,  68, -47, -42, -47, -42,
  51,  70,  89,  71, -46, -42, -48, -42,
  46,  73,  84,  74, -45, -42, -49, -41,};


int currentAng[DOF] = { -30, -80, -45, 0,
                        0, 0, 0, 0,
                        75, 75, 75, 75,
                        -55, -55, -55, -55 };

int speedRatio = 0;


// 8-11: Shoulders (8: FLS, 9: FRS, 10: BRS, 11: BLS)
// 12-15: Knees (12: FLK, 13: FRK, 14: BRK, 15: BLK)
int8_t walkCalib[DOF] = { 0, 0, 0, 0,
                        0, 0, 0, 0,
                        0,  -18,  20, -21,
                        30, 20, 25, 35};

//int8_t crawlCalib[DOF] = { 0, 0, 0, 0,
//                        0, 0, 0, 0,
//                        -10,  -30,  10, 10, 
//                        38, 38, 20, 30 };

int8_t crawlCalib[DOF] = { 0, 0, 0, 0,
                        0, 0, 0, 0,
                        0,  -20,  35, 0, 
                        39, 37, 20, 28 };


char gait = 'w';
//char gait = 'c';
uint8_t cur_step[8] = {0, 0, 0, 0, 0, 0, 0, 0};
//uint8_t frames_per_burst = 8;
uint8_t frames_per_burst = 16;
//uint8_t frames_per_burst = 4;

void setup() {
  Serial.begin(9600);
//  saveCalib(walkCalib);
//  servoSetup();
//  changeGait('w', walkCalib);
  stand();
  delay(2000);
}


// 8-11: Shoulders (8: FLS, 9: FRS, 10: BRS, 11: BLS)
// 12-15: Knees (12: FLK, 13: FRK, 14: BRK, 15: BLK)
int trans_to_walk_angs[8] = {30,  30,  30,  45,
                                            10,  20,  15,  10};
int init_angs[8] = {0,  0,  0,  0,
                    0,  0,  0,  0};
uint8_t trans_servo_seq[8] = {8, 12, 9, 13, 11, 10, 15, 14};

int start_time, end_time;

void loop() {
  start_time = millis();
  
  int spike[9] = {0}; // Initialize array with zeros
  if (Serial.available() > 0) {
    // Deserialize the data back into an array of integers
    for (int i = 0; i < 9; i++) {
        spike[i] = Serial.parseInt(); // Parse each integer value from the serial input
        if (Serial.peek() == ' ') {
            Serial.read(); // Read the space character separator
        }
    

    // Gait selection
    if (spike[8] == 3 && gait != 'w') {
      changeGait('w', walkCalib);
    } 
    else if (spike[8] == 2 && gait != 'c') {
      changeGait('c', crawlCalib);
    } 

    }
  }

  // Handle burst logic
   for (uint8_t j = 0; j < frames_per_burst; j++) {
      if (spike[0] == 1) {
      for (uint8_t i = 0; i < 8; i++) {
//      if (spike[i] == 1) {
          writeAngleToPetoi(cur_step[i], 8 + i, gait);
//          delay(1.0);
          cur_step[i]++;
          if ((cur_step[i] >= 54 && gait == 'w') || (cur_step[i] >= 34 && gait == 'c')) {
            cur_step[i] = 0;
          }
        }
//        delay(6.0);
          if (gait == 'w')
            delay(4.0);
          else
//            delay(8.0);
              delay(7.0);
      }
    }


    end_time = millis(); // Capture end time
  int duration = end_time - start_time;
  byte data[2];
  data[0] = highByte(duration); // Split into high byte
  data[1] = lowByte(duration);  // Split into low byte
  Serial.write(data, 2); // Send the integer as 2 bytes
//  Serial.println(duration);
}

void changeGait(char newGait, int8_t *calib) {
//  saveCalib(calib);
//  servoSetup();
  gait = newGait;

  if (gait == 'w') {
    saveCalib(calib);
    servoSetup();
    for (uint8_t i = 0; i < 8; i++) {
      writeExactAngle(init_angs[i], i+8);
      delay(500);
    }
    for (uint8_t i = 0; i < 8; i++) {
      writeExactAngle(trans_to_walk_angs[trans_servo_seq[i]-8], trans_servo_seq[i]);
      if (i == 3)
        delay(1000);
      else
        delay(20);
    }
  }
  else {
    for (uint8_t i = 0; i < 8; i++) {
      writeExactAngle(init_angs[trans_servo_seq[i]-8], trans_servo_seq[i]);
      if (i == 3)
      delay(1000);
    }
    saveCalib(calib);
    servoSetup();
//    delay(500);
//    for (uint8_t i = 8; i < 15; i++) {
//    for (uint8_t i = 0; i < 8; i++) {
////      writeAngleToPetoi(0, i, gait);
//      writeAngleToPetoi(0, trans_servo_seq[i], gait);
//      delay(200);
//    }
    delay(100);
  }

  for (uint8_t i = 0; i < 8; i++) {
    cur_step[i] = 0;
  }
  delay(500.0);
}


void writeAngleToPetoi(uint8_t cur_step, uint8_t servoNum, char gait) {
    int cur_angle = 0;
    if (gait == 'w') {
        cur_angle = wkF[(servoNum % 8) + cur_step * 8];
    } else if (gait == 'c') {
        cur_angle = crF[(servoNum % 8) + cur_step * 8];
    } else {
        return;
    }
    int i = servoNum;
    int angle = max(float(loadAngleLimit(i, 0)), min(float(loadAngleLimit(i, 1)), cur_angle));
    int duty0 = EEPROMReadInt(CALIBRATED_ZERO_POSITIONS + i * 2) + currentAng[i] * eeprom(ROTATION_DIRECTION, i);
    currentAng[i] = angle;

    int duty = EEPROMReadInt(CALIBRATED_ZERO_POSITIONS + i * 2) + angle * eeprom(ROTATION_DIRECTION, i);
    int steps = speedRatio > 0 ? int(round(abs(duty - duty0) / 1.0 / speedRatio)) : 0;
    for (int s = 0; s <= steps; s++) {
        pwm.writeAngle(servoNum, duty + (steps == 0 ? 0 : (1 + cos(M_PI * s / steps)) / 2 * (duty0 - duty)));
    }
}

void writeExactAngle(int cur_angle, uint8_t servoNum) {
    int i = servoNum;
    int angle = max(float(loadAngleLimit(i, 0)), min(float(loadAngleLimit(i, 1)), cur_angle));
    int duty0 = EEPROMReadInt(CALIBRATED_ZERO_POSITIONS + i * 2) + currentAng[i] * eeprom(ROTATION_DIRECTION, i);
    currentAng[i] = angle;

    int duty = EEPROMReadInt(CALIBRATED_ZERO_POSITIONS + i * 2) + angle * eeprom(ROTATION_DIRECTION, i);
    int steps = speedRatio > 0 ? int(round(abs(duty - duty0) / 1.0 /*degreeStep*/ / speedRatio)) : 0;
    //if default speed is 0, no interpolation will be used
    //otherwise the speed ratio is compared to 1 degree per second.
    for (int s = 0; s <= steps; s++) {
      // pwm.writeAngle(i, duty + (steps == 0 ? 0 : (1 + cos(M_PI * s / steps)) / 2 * (duty0 - duty)));
      pwm.writeAngle(servoNum, duty + (steps == 0 ? 0 : (1 + cos(M_PI * s / steps)) / 2 * (duty0 - duty)));
    }

}


void stand() {
  saveCalib(walkCalib);
  servoSetup();

  if (gait == 'w') {
    for (uint8_t i = 0; i < 8; i++) {
      writeExactAngle(trans_to_walk_angs[trans_servo_seq[i]-8], trans_servo_seq[i]);
    }
  }
}
