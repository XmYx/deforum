import threading


class SingletonBorg:
    _shared_state = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not hasattr(cls, "_instance"):
                cls._instance = super(SingletonBorg, cls).__new__(cls, *args, **kwargs)
                cls._instance.__dict__ = cls._shared_state
        return cls._instance


class ModelStorage(SingletonBorg):
    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            # Add any other initialization code here.
            self.model = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

