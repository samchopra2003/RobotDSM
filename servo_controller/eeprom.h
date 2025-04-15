#define DOF 16

#define PWM_PIN 0                     // 16 byte array
#define CALIB 16                      // 16 byte array
#define MID_SHIFT 32                  // 16 byte array
#define ROTATION_DIRECTION 48         // 16 byte array
#define ZERO_POSITIONS 64             // 16*2 = 32   64+32 = 96
#define CALIBRATED_ZERO_POSITIONS 96  // 16*2 = 32   96+32 = 128
#define ADAPT_PARAM 128               // 16 x 2 byte array 128+32 = 160
#define ANGLE2PULSE_FACTOR 160        // 16*2 = 32  160+32 = 192
#define ANGLE_LIMIT 192               // 16*2*2 = 64   192+64 = 256
#define MPUCALIB 256                  // 9 int byte array 9x2 =18  256+18=274
#define B_OFFSET 274                  // 1 bytes
#define PCA9685_FREQ 275              // 2 bytes
#define NUM_SKILLS 277                // 1 bytes


void EEPROMWriteInt(int p_address, int p_value)
{
  byte lowByte = ((p_value >> 0) & 0xFF);
  byte highByte = ((p_value >> 8) & 0xFF);
  EEPROM.update(p_address, lowByte);
  EEPROM.update(p_address + 1, highByte);
  delay(6);
}
//This function will read a 2-byte integer from the EEPROM at the specified address and address + 1
int EEPROMReadInt(int p_address)
{
  byte lowByte = EEPROM.read(p_address);
  byte highByte = EEPROM.read(p_address + 1);
  return ((lowByte << 0) & 0xFF) + ((highByte << 8) & 0xFF00);
}

inline int8_t eeprom(int address, byte x = 0, byte y = 0, byte yDim = 1) {
  return EEPROM.read(address + x * yDim + y);
}

void flushEEPROM(char b = -1) {
  for (int j = 0; j < 1024; j++) {
    EEPROM.update(j, b);
  }
}

// void printEEPROM() {
//   PTLF("\nEEPROM contents:");
//   for (int i = 0; i < 1024; i++) {
//     if (!(i % 16)) {
//       PT(i / 16);
//       PTF(":\t");
//       delay(6);
//     }
//     PT(eeprom(i));
//     PT('\t');
//     if (i % 16 == 15)
//       PTL();
//   }
// }

// template <typename T> void printEEPROMList(int EEaddress, byte len = DOF) {
//   for (byte i = 0; i < len; i++) {
//     PT((T)(EEPROM.read(EEaddress + i)));
//     PT('\t');
//   }
//   PTL();
// }

void saveCalib(int8_t *var) {
  for (byte i = 0; i < DOF; i++) {
    EEPROM.update(CALIB + i, var[i]);
    //    calibratedZeroPosition[i] = EEPROMReadInt(ZERO_POSITIONS + i * 2) + float(var[i]) * eeprom(ROTATION_DIRECTION, i);
    EEPROMWriteInt(CALIBRATED_ZERO_POSITIONS + i * 2, EEPROMReadInt(ZERO_POSITIONS + i * 2) + float(var[i]) * eeprom(ROTATION_DIRECTION, i));
  }
}

void saveMelody(int &address, byte melody[], int len ) {
  EEPROM.update(address--, len);
  for (byte i = 0; i < len; i++)
    EEPROM.update(address --, melody[i]);
  // PTL(address);
}

inline int loadAngleLimit(byte idx, byte dim) {
  return EEPROMReadInt(ANGLE_LIMIT + idx * 4 + dim * 2);
}
