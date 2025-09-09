import serial
import time
import glob

class DynamixelController:
    def __init__(self, port=None, baudrate=115200, init_left=0, init_right=0):
        self.baudrate = baudrate
        self.left_pos = init_left
        self.right_pos = init_right
        self.ser = None
        self.port = port or self.find_port()
        self.connect()

    def find_port(self):
        """Busca automáticamente un puerto ttyUSB o ttyACM."""
        candidates = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        return candidates[0] if candidates else None

    def connect(self):
        """Intenta conectar al ESP32."""
        if not self.port:
            raise RuntimeError("No se encontró ningún puerto USB")

        try:
            print(f"[GripperController] Conectando a {self.port}...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2.5)  # Deja que el ESP32 arranque
            if self.check_alive():
                print("[GripperController] Conectado exitosamente ✅")
                self.send_positions(self.left_pos, self.right_pos)
            else:
                raise RuntimeError("ESP32 no responde a ping")
        except Exception as e:
            print(f"[GripperController] Error conectando: {e}")
            self.ser = None

    def reconnect(self):
        """Intenta reconectar si la conexión se perdió."""
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
        self.port = self.find_port()
        self.connect()

    def check_alive(self):
        """Envía [ping] y espera pong."""
        try:
            self.ser.reset_input_buffer()
            self.ser.write(b"[ping]\n")
            response = self.ser.readline().decode().strip()
            return response == "pong"
        except Exception as e:
            print(f"[GripperController] Error check_alive: {e}")
            return False

    def send_positions(self, left, right):
        """Envia posiciones de ambos grippers."""
        if not self.ser or not self.check_alive():
            print("[GripperController] ❌ No hay conexión. Reintentando...")
            self.reconnect()

        try:
            command = f"[{right};{left}]\n"
            self.ser.write(command.encode())
            print(f"[GripperController] Enviado: {command.strip()}")
        except Exception as e:
            print(f"[GripperController] Error al enviar posiciones: {e}")

    def set_left_gripper(self, pos):
        if 0 <= pos <= 1023:
            self.left_pos = pos
            self.send_positions(self.left_pos, self.right_pos)
        else:
            print("[GripperController] Valor inválido para el gripper izquierdo.")

    def set_right_gripper(self, pos):
        if 0 <= pos <= 1023:
            self.right_pos = pos
            self.send_positions(self.left_pos, self.right_pos)
        else:
            print("[GripperController] Valor inválido para el gripper derecho.")

    def set_grippers(self, left, right):
        if 0 <= left <= 1023 and 0 <= right <= 1023:
            self.left_pos = left
            self.right_pos = right
            self.send_positions(left, right)
        else:
            print("[GripperController] Valores fuera de rango (0–1023).")

    def get_positions(self):
        return self.left_pos, self.right_pos

    def close(self):
        if self.ser:
            try:
                self.ser.write(b"[reset]\n")
                time.sleep(0.5)
                self.ser.close()
                print("[GripperController] Conexión cerrada.")
            except:
                pass
