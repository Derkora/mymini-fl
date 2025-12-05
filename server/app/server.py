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
        self.end_time = 0
        self.model_size_bytes = 0 # Ukuran payload model head
        self.clients_per_round = 2 

    def _get_initial_parameters(self):
        db = SessionLocal()
        try:
            # Ambil versi terakhir
            latest_model = db.query(ModelVersion).order_by(ModelVersion.version_id.desc()).first()
            
            if latest_model:
                self.model_size_bytes = len(latest_model.head_blob)
                print(f"[FL SERVER] Model Size Loaded: {self.model_size_bytes} bytes")

                buffer = io.BytesIO(latest_model.head_blob)
                state_dict = torch.load(buffer)
                params = [val.cpu().numpy() for _, val in state_dict.items()]
                return ndarrays_to_parameters(params)
            else:
                print("[FL SERVER] Belum ada model di DB. Jalankan init.py dulu!")
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
        self.current_round = 0
        self.total_rounds = rounds
        self.start_time = time.time()
        self.end_time = 0 
        
        if self.model_size_bytes == 0:
             self._get_initial_parameters()

        print(f"[FL SERVER] Mulai Federated Learning: {rounds} rounds")

        self.server_thread = threading.Thread(
            target=self._run_flower_server,
            args=(rounds,),
            daemon=True
        )
        self.server_thread.start()

    def _run_flower_server(self, rounds: int):
        try:
            initial_params = self._get_initial_parameters()

            def fit_config(rnd):
                self.current_round = rnd
                return {"round": rnd}

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
            print("[FL SERVER] Training Selesai.")

        except Exception as e:
            print(f"[FL SERVER] Error: {e}")
        finally:

            self.running = False
            self.end_time = time.time() 
            print(f"[FL SERVER] Stopped at {self.end_time}")

    def stop(self):
        self.running = False

    def status(self):
        if self.running:
            elapsed = time.time() - self.start_time
        elif self.end_time > 0:
            elapsed = self.end_time - self.start_time
        else:
            elapsed = 0
            
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        size_kb = self.model_size_bytes / 1024
        
        if self.current_round > 0:
            total_kb = size_kb * self.clients_per_round * 2 * self.current_round * 1.1
        else:
            total_kb = 0
        
        bandwidth_str = f"{total_kb:,.2f} KB"

        is_finished = (self.current_round >= self.total_rounds) and (self.total_rounds > 0)
        
        return {
            "running": self.running and not is_finished,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "elapsed_time": time_str, 
            "bandwidth_kb": bandwidth_str 
        }