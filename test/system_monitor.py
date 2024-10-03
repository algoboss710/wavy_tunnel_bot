import psutil
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class SystemMonitor:
    def __init__(self, duration=43200):  # 12 hours in seconds
        self.duration = duration
        self.cpu_percent = []
        self.memory_percent = []
        self.disk_io = []
        self.network_io = []
        self.time_points = []
        self.start_time = time.time()

        self.prev_disk_read = psutil.disk_io_counters().read_bytes
        self.prev_disk_write = psutil.disk_io_counters().write_bytes
        self.prev_net_sent = psutil.net_io_counters().bytes_sent
        self.prev_net_recv = psutil.net_io_counters().bytes_recv

    def collect_data(self):
        while time.time() - self.start_time < self.duration:
            current_time = time.time() - self.start_time

            # CPU usage
            self.cpu_percent.append(psutil.cpu_percent())

            # Memory usage
            self.memory_percent.append(psutil.virtual_memory().percent)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes - self.prev_disk_read
            disk_write = disk_io.write_bytes - self.prev_disk_write
            self.disk_io.append((disk_read + disk_write) / 1024 / 1024)  # MB
            self.prev_disk_read = disk_io.read_bytes
            self.prev_disk_write = disk_io.write_bytes

            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent - self.prev_net_sent
            net_recv = net_io.bytes_recv - self.prev_net_recv
            self.network_io.append((net_sent + net_recv) / 1024 / 1024)  # MB
            self.prev_net_sent = net_io.bytes_sent
            self.prev_net_recv = net_io.bytes_recv

            self.time_points.append(current_time / 60)  # Convert to minutes

            time.sleep(1)  # Collect data every second

    def plot_data(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

        def update(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.plot(self.time_points, self.cpu_percent)
            ax1.set_title('CPU Usage')
            ax1.set_ylabel('Percentage')

            ax2.plot(self.time_points, self.memory_percent)
            ax2.set_title('Memory Usage')
            ax2.set_ylabel('Percentage')

            ax3.plot(self.time_points, self.disk_io)
            ax3.set_title('Disk I/O')
            ax3.set_ylabel('MB/s')

            ax4.plot(self.time_points, self.network_io)
            ax4.set_title('Network I/O')
            ax4.set_ylabel('MB/s')
            ax4.set_xlabel('Time (minutes)')

        ani = FuncAnimation(fig, update, interval=1000)
        plt.tight_layout()
        plt.show()

    def run(self):
        data_thread = threading.Thread(target=self.collect_data)
        data_thread.start()

        self.plot_data()

        data_thread.join()

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.run()