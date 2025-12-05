import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet
import io
import torchvision.transforms as T # <--- TAMBAHAN

class FaceAnalysisPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[PIPELINE] Loading MTCNN on {self.device}...")
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        print("[PIPELINE] Loading MobileFaceNet Backbone...")
        self.backbone = MobileFaceNet(embedding_size=128).to(self.device)
        self.backbone.eval()

        # AUGMENTASI
        self.transforms = [
            # Flip Horizontal (Efek Cermin)
            T.RandomHorizontalFlip(p=1.0),
            
            # Perubahan Pencahayaan (Brightness & Contrast)
            T.ColorJitter(brightness=0.3, contrast=0.3),
            
            # Rotasi Sedikit (+/- 10 derajat) untuk toleransi miring kepala
            T.RandomRotation(degrees=10),
            
            # Kombinasi Flip + Color
            T.Compose([
                T.RandomHorizontalFlip(p=1.0),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        ]

    def _get_embedding(self, img_pil):
        """Helper internal untuk ekstrak embedding dari objek PIL Image"""
        try:
            face_tensor, prob = self.mtcnn(img_pil, return_prob=True)
            
            if face_tensor is None:
                return None, "No face"
            
            if prob < 0.85: 
                 return None, f"Low confidence ({prob:.2f})"

            face_batch = face_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.backbone(face_batch)
            
            emb_numpy = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(emb_numpy)
            if norm != 0:
                emb_numpy = emb_numpy / norm
                
            return emb_numpy, "Success"
        except Exception as e:
            return None, str(e)

    def process_image(self, image_bytes):
        """Metode Legacy (Single Image)"""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self._get_embedding(img)

    def process_with_augmentation(self, image_bytes):
        """
        Metode Baru: Menghasilkan BANYAK embedding dari 1 foto.
        Return: List of valid embeddings (numpy arrays)
        """
        valid_embeddings = []
        
        original_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        emb, msg = self._get_embedding(original_img)
        if emb is not None:
            valid_embeddings.append(emb)
        else:
            return [], f"Original Fail: {msg}"

        for i, transform in enumerate(self.transforms):
            try:
                aug_img = transform(original_img)
                
                aug_emb, aug_msg = self._get_embedding(aug_img)
                
                if aug_emb is not None:
                    valid_embeddings.append(aug_emb)
            except Exception as e:
                print(f"[AUGMENT ERROR] Transform {i}: {e}")

        print(f"[PIPELINE] Generated {len(valid_embeddings)} embeddings from 1 image.")
        return valid_embeddings, "Success"

# Singleton instance
face_pipeline = FaceAnalysisPipeline()