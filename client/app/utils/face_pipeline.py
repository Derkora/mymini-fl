import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet
import io

class FaceAnalysisPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[PIPELINE] Loading MTCNN on {self.device}...")
        self.mtcnn = MTCNN(
            image_size=112, # Ukuran input standar MobileFaceNet
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        print("[PIPELINE] Loading MobileFaceNet Backbone...")
        self.backbone = MobileFaceNet(embedding_size=128).to(self.device)
        self.backbone.eval() # Set ke mode evaluasi (Frozen Backbone)

    def process_image(self, image_bytes):
        """
        Menerima bytes gambar -> Return embedding (numpy)
        """
        try:
            # Load Image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # MTCNN: Detect & Crop
            # return_prob=True mengembalikan probabilitas deteksi juga
            face_tensor, prob = self.mtcnn(img, return_prob=True)
            
            if face_tensor is None:
                return None, "No face detected"
            
            if prob < 0.90: # Filter jika confidence rendah
                 return None, f"Face confidence too low ({prob:.2f})"

            # Preprocess untuk MobileFaceNet (Normalize)
            # MTCNN facenet-pytorch outputnya sudah ditensor dan discale, 
            # tapi biasanya MobileFaceNet butuh normalisasi ( - 127.5 / 128 )
            # Disini kita asumsikan output MTCNN sudah range standar.
            
            # Tambah dimensi batch: (1, 3, 112, 112)
            face_batch = face_tensor.unsqueeze(0).to(self.device)

            # Forward Pass Backbone
            with torch.no_grad():
                embedding = self.backbone(face_batch)
            
            # Convert to Numpy & Normalize (L2 Norm)
            emb_numpy = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(emb_numpy)
            if norm != 0:
                emb_numpy = emb_numpy / norm
                
            return emb_numpy, "Success"

        except Exception as e:
            return None, str(e)

# Singleton instance
face_pipeline = FaceAnalysisPipeline()