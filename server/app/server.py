import threading
import flwr as fl
import torch
import io
import time
from app.db.db import SessionLocal
from app.db.models import ModelVersion
from flwr.common import ndarrays_to_parameters

class FLServerManager:
    def __init__(self):
        self.running = False
        self.current_round = 0
        self.total_rounds = 0
        self.server_thread = None
        
        # Metrics
        self.start_time = 0
        self.model_size_bytes = 0 # Ukuran payload model head
        self.clients_per_round = 2 

    def _get_initial_parameters(self):
        db = SessionLocal()
        try:
            latest_model = db.query(ModelVersion).order_by(ModelVersion.version_id.desc()).first()
            
            if latest_model:
                # Simpan ukuran model untuk perhitungan bandwidth
                self.model_size_bytes = len(latest_model.head_blob)
                print(f"[FL SERVER] Model Size Loaded: {self.model_size_bytes / 1024:.2f} KB")

                buffer = io.BytesIO(latest_model.head_blob)
                state_dict = torch.load(buffer)
                params = [val.cpu().numpy() for _, val in state_dict.items()]
                return ndarrays_to_parameters(params)
            else:
                return None
        except Exception as e:
            print(f"[FL SERVER] Error loading model: {e}")
            return None
        finally:
            db.close()

    def start_training(self, rounds: int = 10):
        if self.running:
            return

        self.running = True
        self.total_rounds = rounds
        self.start_time = time.time() # Mulai stopwatch
        
        print(f"[FL SERVER] Mulai Federated Learning: {rounds} rounds")

        self.server_thread = threading.Thread(
            target=self._run_flower_server,
            args=(rounds,),
            daemon=True
        )
        self.server_thread.start()

    def _run_flower_server(self, rounds: int):
        initial_params = self._get_initial_parameters()

        def fit_config(rnd):
            self.current_round = rnd
            return {"round": rnd}

        # Konfigurasi Strategi
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.clients_per_round,
            min_available_clients=self.clients_per_round,
            on_fit_config_fn=fit_config,
            initial_parameters=initial_params
        )

        fl.server.start_server(
            server_address="0.0.0.0:8085",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=rounds)
        )

        self.running = False
        print("[FL SERVER] Training Selesai.")

    def stop(self):
        pass 

    def status(self):
        # Hitung Durasi
        if self.running:
            elapsed = time.time() - self.start_time
        elif self.start_time > 0:
            elapsed = time.time() - self.start_time
        else:
            elapsed = 0
            
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins:02d}:{secs:02d}"

        total_transfer_bytes = self.model_size_bytes * self.clients_per_round * 2 * self.current_round * 1.1
        bandwidth_mb = total_transfer_bytes / (1024 * 1024)

        return {
            "running": self.running,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "elapsed_time": time_str,
            "bandwidth_mb": f"{bandwidth_mb:.2f} MB"
        }