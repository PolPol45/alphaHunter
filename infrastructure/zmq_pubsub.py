import zmq
import json
import logging

class EventBus:
    """Wrapper per interfacciarsi con il broker ZeroMQ (Publisher o Subscriber)"""
    def __init__(self, role="PUB"):
        self.context = zmq.Context()
        self.role = role
        if role == "PUB":
            self.socket = self.context.socket(zmq.PUB)
            self.socket.connect("tcp://127.0.0.1:5555")  # Connect to broker frontend
        elif role == "SUB":
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect("tcp://127.0.0.1:5556")  # Connect to broker backend
        
        self.logger = logging.getLogger(f"zmq_{role}")

    def publish(self, topic: str, payload: dict):
        if self.role != "PUB":
            raise ValueError("Solo un publisher pu\u00f2 pubblicare")
        msg = f"{topic} {json.dumps(payload)}"
        self.socket.send_string(msg)
        self.logger.debug(f"Pubblicato evento su [{topic}]")

    def subscribe(self, topic: str):
        if self.role != "SUB":
            raise ValueError("Solo un subscriber pu\u00f2 sottoscrivere")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    def receive(self):
        msg = self.socket.recv_string()
        topic, json_payload = msg.split(" ", 1)
        return topic, json.loads(json_payload)
