import time
import multiprocessing as mp
import zerorpc
import numpy as np

class FrankaHandController(mp.Process):
    def __init__(self, host="129.97.71.27", port=4242, speed=0.05, force=20.0, update_rate=30):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.speed = speed
        self.force = force
        self.update_dt = 1.0 / update_rate
        self.command_queue = mp.Queue()
        self.ready_event = mp.Event()
        self._stop_event = mp.Event()
        self._client = None
        self._connected = False

    def connect(self):
        if not self._connected:
            try:
                self._client = zerorpc.Client()
                self._client.connect(f"tcp://{self.host}:{self.port}")
                self._connected = True
                print("[FrankaHandController] Connected to gripper RPC.")
            except Exception as e:
                print(f"[FrankaHandController] Failed to connect: {e}")
                self._connected = False

    def get_state(self):
        self.connect()
        try:
            return self._client.get_gripper_state()
        except Exception as e:
            print(f"[FrankaHandController] Error getting gripper state: {e}")
            return {'width': 0.0}

    def get_all_state(self):
        state = self.get_state()
        try:
            timestamp = state['timestamp']['seconds'] + state['timestamp']['nanoseconds'] * 1e-9
        except KeyError:
            timestamp = time.time()

        return {
            'gripper_position': [state['width']],
            'gripper_timestamp': [timestamp],
        }
    
    def schedule_waypoint(self, pos):
        if isinstance(pos, np.ndarray):
            pos = float(pos)
        self.send_target(pos)

    def send_target(self, width):
        self.command_queue.put(width)

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            time.sleep(0.5)

    def stop(self, wait=True):
        self._stop_event.set()
        if wait:
            self.join()
        print("[FrankaHandController] Stopped.")

    def start_wait(self):
        pass

    def stop_wait(self):
        pass

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self):
        self.stop()

    def run(self):
        self.connect()
        if not self._connected:
            print("[FrankaHandController] Failed to start gripper RPC.")
            return

        self.ready_event.set()
        print("[FrankaHandController] Ready and running.")

        last_width = None

        while not self._stop_event.is_set():
            try:
                # Flush queue to get latest command
                while True:
                    width = self.command_queue.get_nowait()
                    last_width = width
            except mp.queues.Empty:
                pass  # No new command, proceed with last one

            if last_width is not None:
                try:
                    # print(f"[FrankaHandController] Streaming width: {last_width:.3f}")
                    self._client.gripper_goto(last_width, self.speed, self.force)
                    # self._client.gripper_grasp(self.speed, self.force, last_width)

                except Exception as e:
                    # print(f"[FrankaHandController] RPC error during gripper_goto: {e}")
                    self._connected = False
                    self.ready_event.clear()
                    break  # Exit loop or try reconnecting optionally

            time.sleep(self.update_dt)

        print("[FrankaHandController] Exiting run loop.")