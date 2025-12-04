import flwr as fl
import numpy as np
import time

# Dummy model (akan diganti classifier head nanti)
def init_weights():
    w = np.random.randn(100)
    b = np.random.randn(1)
    return [w, b]

class DummyClient(fl.client.NumPyClient):
    def __init__(self):
        self.weights = init_weights()

    def get_parameters(self, config):
        print("[CLIENT] Sending initial parameters")
        return self.weights

    def fit(self, parameters, config):
        print(f"[CLIENT] Round {config.get('round')} - training dummy")
        
        # Simulate update
        updated = []
        for p in parameters:
            noise = np.random.normal(0, 0.01, p.shape)
            updated.append(p + noise)

        self.weights = updated
        return updated, len(updated), {}

    def evaluate(self, parameters, config):
        loss = np.random.random()  # dummy accuracy metric
        print(f"[CLIENT] Evaluation loss {loss}")
        return loss, len(parameters), {}

        
def start_flower_client():
    while True:
        try:
            print("[CLIENT] Trying to connect FL server...")
            fl.client.start_numpy_client(
                server_address="server:8085",
                client=DummyClient()
            )
            break
        except Exception as e:
            print("[CLIENT] Cannot connect, retrying in 3s...")
            time.sleep(3)
