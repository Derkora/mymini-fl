import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet
import io
import torchvision.transforms as T
import cv2
import random
import base64

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
            # Simulasi Random Crop dengan resize kembali ke 112x112 
            T.Compose([
                T.Resize(int(112 * 1.1)), # Zoom in
                T.RandomCrop(112),
            ]),
            T.Compose([ # Kombinasi untuk variasi
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        ]
        
    def _pil_to_base64(self, img_pil):
        """Konversi PIL Image ke string Base64"""
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _draw_bounding_box(self, img_pil, box):
        """Menggambar bounding box di gambar PIL dan mengembalikan base64"""
        if box is None:
            return self._pil_to_base64(img_pil)
            
        img_np = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Koordinat MTCNN adalah [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2) # Warna Hijau
        
        img_result_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        return self._pil_to_base64(img_result_pil)

    def _add_random_noise(self, img_pil):
        """Menambahkan Gaussian Noise (simulasi random noise)"""
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mean = 0.0
        stddev = random.uniform(0.01, 0.05) # Noise level
        noise = np.random.normal(mean, stddev, img_np.shape)
        noisy_img_np = img_np + noise
        noisy_img_np = np.clip(noisy_img_np, 0.0, 1.0)
        return Image.fromarray((noisy_img_np * 255).astype(np.uint8))


    def _check_quality(self, img_pil):
        """Cek apakah gambar buram atau terlalu gelap/terang"""
        try:
            img_np = np.array(img_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 70: 
                return False, f"Blurry (Score: {blur_score:.1f})"
            mean_brightness = np.mean(gray)
            if mean_brightness < 30:
                return False, "Too Dark"
            if mean_brightness > 230:
                return False, "Too Bright"
            return True, "OK"
        except Exception as e:
            return True, "Quality Check Skip" 

    def _get_embedding(self, img_pil):
        """
        Melakukan deteksi, ekstraksi embedding, dan mengembalikan 
        embedding numpy, gambar wajah yang di-crop (PIL), dan bounding box (kalau ada)
        """
        try:
            is_good, msg = self._check_quality(img_pil)
            if not is_good:
                return None, None, None, f"Quality Reject: {msg}"

            boxes, probs = self.mtcnn.detect(img_pil)

            if boxes is None or len(boxes) == 0:
                return None, None, None, "No face"
            
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            prob = probs[best_idx]
            
            if prob < 0.75: 
                 return None, None, None, f"Low confidence ({prob:.2f})"

            face_tensor = self.mtcnn.extract(img_pil, [box], save_path=None)
            
            if face_tensor is None:
                return None, None, None, "Face extraction failed"

            # Crop wajah 112x112 (untuk visualisasi crop)
            face_img_pil = img_pil.crop(box.astype(int))
            face_img_pil = face_img_pil.resize((112, 112))

            face_batch = face_tensor.to(self.device)

            with torch.no_grad():
                embedding = self.backbone(face_batch)
            
            emb_numpy = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(emb_numpy)
            if norm != 0:
                emb_numpy = emb_numpy / norm
                
            return emb_numpy, face_img_pil, box, "Success"
        except Exception as e:
            return None, None, None, str(e)

    def process_image(self, image_bytes):
        """
        Untuk endpoint /attendance. 
        Mengembalikan embedding, gambar asli dengan bounding box (b64), dan pesan
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            original_img = img.copy() 
            
            emb, face_img_pil, box, msg = self._get_embedding(original_img)
            
            drawn_img_b64 = self._draw_bounding_box(original_img, box)
            
            return emb, drawn_img_b64, msg
        except Exception as e:
            return None, None, str(e)

    def process_with_augmentation(self, image_bytes, filename="original"):
        """
        Untuk endpoint /register. 
        Mengembalikan list of dict: [{'emb': array, 'cropped_face_b64': str, 'original_img_with_bb_b64': str, 'source': str, 'status': str}, ...]
        """
        valid_data = []
        
        try:
            original_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return [], f"File Error: {e}", []
        
        # Proses Asli
        emb, face_img_pil, box, msg = self._get_embedding(original_img.copy()) 
        
        original_img_bb_b64 = self._draw_bounding_box(original_img.copy(), box)
        
        if emb is not None:
            valid_data.append({
                'emb': emb,
                'cropped_face_b64': self._pil_to_base64(face_img_pil) if face_img_pil else None,
                'original_img_with_bb_b64': original_img_bb_b64,
                'source': f"{filename} (Original)",
                'status': 'Success'
            })
        else:
            valid_data.append({
                'emb': None,
                'cropped_face_b64': None,
                'original_img_with_bb_b64': original_img_bb_b64,
                'source': f"{filename} (Original)",
                'status': f"Fail: {msg}"
            })

        # Proses Augmentasi
        for i, transform in enumerate(self.transforms):
            aug_source = f"{filename} (Aug #{i+1})"
            try:
                temp_img = original_img.copy() 
                aug_img = transform(temp_img)
                
                is_noise_transform = (i == 3)
                if is_noise_transform: 
                    aug_img = self._add_random_noise(aug_img)
                    aug_source = f"{filename} (Aug #{i+1} - Noise)"
                    
                aug_emb, aug_face_img_pil, aug_box, aug_msg = self._get_embedding(aug_img)
                
                aug_img_bb_b64 = self._draw_bounding_box(aug_img.copy(), aug_box)
                
                if aug_emb is not None:
                    valid_data.append({
                        'emb': aug_emb,
                        'cropped_face_b64': self._pil_to_base64(aug_face_img_pil),
                        'original_img_with_bb_b64': aug_img_bb_b64,
                        'source': aug_source,
                        'status': 'Success'
                    })
                else:
                    valid_data.append({
                        'emb': None,
                        'cropped_face_b64': None,
                        'original_img_with_bb_b64': aug_img_bb_b64,
                        'source': aug_source,
                        'status': f"Fail: {aug_msg}"
                    })
            except Exception as e:
                print(f"[AUGMENT ERROR] Transform {i}: {e}")
                valid_data.append({
                    'emb': None,
                    'cropped_face_b64': None,
                    'original_img_with_bb_b64': None,
                    'source': aug_source,
                    'status': f"Error: {e}"
                })
        
        successful_embeddings = [d['emb'] for d in valid_data if d['emb'] is not None]
        
        # Cek apakah setidaknya original_img berhasil, jika tidak, kembalikan error umum
        if not [d for d in valid_data if d['source'].endswith('(Original)') and d['emb'] is not None]:
             return [], "Original Fail: No face detected in original image or poor quality.", valid_data
             
        print(f"[PIPELINE] Generated {len(successful_embeddings)} successful embeddings from {filename}.")
        return successful_embeddings, "Success", valid_data

face_pipeline = FaceAnalysisPipeline()