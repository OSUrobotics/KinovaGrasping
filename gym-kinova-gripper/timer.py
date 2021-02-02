import time

class TimerError(Exception):
    """Reports for error in timer usage"""
    print("Uh oh! Error with timer")

class Timer:
    def __init__(self):
        self._wall_start_time = None
        self._process_start_time = None

    def start(self):
        """Start a new timer"""
        if self._wall_start_time is not None:
            raise TimerError(f"Wall timer is running. Use .stop() to stop it")

        if self._process_start_time is not None:
            raise TimerError(f"Process timer is running. Use .stop() to stop it")

        self._wall_start_time = time.perf_counter()
        self._process_start_time = time.process_time()

    def stop(self):
        """Stop the timer and report the elapsed time"""
        if self._wall_start_time is None:
            raise TimerError(f"Wall Timer is not running. Use .start() to start it")

        if self._process_start_time is None:
            raise TimerError(f"Wall Timer is not running. Use .start() to start it")

        wall_elapsed_time = time.perf_counter() - self._wall_start_time
        process_elapsed_time = time.perf_counter() - self._process_start_time

        self._wall_start_time = None
        self._process_start_time = None

        print(f"\nWall clock elapsed time: {wall_elapsed_time:0.4f} seconds")
        print(f"Process elapsed time: {process_elapsed_time:0.4f} seconds")