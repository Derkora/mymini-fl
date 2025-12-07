import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet
import io
import torchvision.transforms as T
import cv2

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
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomRotation(degrees=10),
            T.Compose([
                T.RandomHorizontalFlip(p=1.0),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        ]

    def _check_quality(self, img_pil):
        """Cek apakah gambar buram atau terlalu gelap/terang"""
        try:
            # Konversi PIL ke OpenCV format (numpy)
            img_np = np.array(img_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 1. Blur Detection (Variance of Laplacian)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100: # Threshold umum, bisa disesuaikan
                return False, f"Blurry (Score: {blur_score:.1f})"
                
            # 2. Brightness Check
            mean_brightness = np.mean(gray)
            if mean_brightness < 40:
                return False, "Too Dark"
            if mean_brightness > 220:
                return False, "Too Bright"
                
            return True, "OK"
        except Exception as e:
            return True, "Quality Check Skip" # Lanjut jika error library

    def _get_embedding(self, img_pil):
        try:
            # [BARU] Cek Kualitas Sebelum Proses
            is_good, msg = self._check_quality(img_pil)
            if not is_good:
                return None, f"Quality Reject: {msg}"

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
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self._get_embedding(img)
        except Exception as e:
            return None, str(e)

    def process_with_augmentation(self, image_bytes):
        valid_embeddings = []
        try:
            original_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return [], f"File Error: {e}"
        
        # Proses Asli
        emb, msg = self._get_embedding(original_img)
        if emb is not None:
            valid_embeddings.append(emb)
        else:
            return [], f"Original Fail: {msg}"

        # Proses Augmentasi
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