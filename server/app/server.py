import threading
import flwr as fl

class FLServerManager:
    def __init__(self):
        self.running = False
        self.current_round = 0
        self.total_rounds = 0
        self.server_thread = None

    def start_training(self, rounds: int = 10):
        if self.running:
            print("[FL SERVER] Training sudah berjalan")
            return

        self.running = True
        self.total_rounds = rounds
        print(f"[FL SERVER] Mulai Federated Learning: {rounds} rounds")

        self.server_thread = threading.Thread(
            target=self._run_flower_server,
            args=(rounds,),
            daemon=True
        )
        self.server_thread.start()

    def _run_flower_server(self, rounds: int):
        print("[FL SERVER] Preparing FedAvg strategy")

        def fit_config(rnd):
            print(f"[FL SERVER] --- Round {rnd} started ---")
            self.current_round = rnd
            return {"round": rnd}

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=2,
            min_available_clients=2,
            on_fit_config_fn=fit_config
        )

        fl.server.start_server(
            server_address="0.0.0.0:8085",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=rounds)
        )

        self.running = False
        print("[FL SERVER] Federated learning selesai")

    def stop(self):
        self.running = False
        print("[FL SERVER] Stop request diterima.")

    def status(self):
        return {
            "running": self.running,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds
        }
