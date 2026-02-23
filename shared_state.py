import threading

class SharedCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.in_count = 0
            self.out_count = 0
            self.running = False

    def add_in(self):
        with self.lock:
            self.in_count += 1

    def add_out(self):
        with self.lock:
            self.out_count += 1

    def get(self):
        with self.lock:
            return {
                "in": self.in_count,
                "out": self.out_count,
                "net": self.in_count - self.out_count,
                "running": self.running
            }

counter_state = SharedCounter()
def get_shared_counter():
    return counter_state