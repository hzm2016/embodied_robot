import serial
import threading

class JoyReader:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.data = None
        self.running = False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

    def read_serial_data(self):
        if not self.ser:
            return

        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line:
                    data = line.split(',')
                    if len(data) != 6:
                        continue
                    
                    data = [int(x) for x in data]
                    data[2] = data[2] - 512
                    data[3] = data[3] - 512

                    if abs(data[2]) < 50:
                        data[2] = 0
                    if abs(data[3]) < 50:
                        data[3] = 0
                    if data[4]<50:
                        data[4] = 0
                    else:
                        data[4] = 1
                    self.data = data

            except ValueError as e:
                print(f"Error parsing line: {line} - {e}")

    def start(self):
        if self.ser:
            self.running = True
            self.thread = threading.Thread(target=self.read_serial_data)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.ser:
            self.thread.join()
            self.ser.close()

if __name__ == '__main__':
    # Usage
    serial_reader = JoyReader()
    serial_reader.start()

    try:
        while True:
            if serial_reader.data:
                print(serial_reader.data)
    except KeyboardInterrupt:
        serial_reader.stop()