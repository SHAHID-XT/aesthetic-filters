import cv2
import numpy as np
import os
from typing import Tuple


class ImageFilter:
    
    def __init__(self):
        
        self.FILTERS = {
            "warm": self.filter_warm,
            "cool": self.filter_cool,
            "cinematic": self.filter_cinematic,
            "faded": self.filter_faded,
            "vintage": self.filter_vintage_film,
            "low_key": self.filter_low_key,
            "punch": self.filter_punch,
            "soft_blue": self.filter_soft_blue,
            "punchy_cinematic": self.filter_punchy_cinematic,
            "emboss": self.filter_emboss,
            "vintage2": self.filter_vintage2,
        }
        
    def apply_filter(self, filter_name, image_path, outpath="output.png", intensity=0.7):
        img = self.read_image(image_path)
        img = self.ensure_float(img)
        fn = self.FILTERS.get(filter_name)
        if fn is None:
            raise ValueError(f"Unknown filter: {filter_name}")
            
        out = fn(img, intensity=intensity)
        out_u8 = self.ensure_uint8(out)
        outpath = os.path.join(os.getcwd(), f"{outpath}")
        self.save_image(outpath, out_u8)
        return outpath
    
    def read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        return img

    def save_image(self, path: str, img: np.ndarray):
        cv2.imwrite(path, img)
        return path

    def ensure_float(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0

    def ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return out
    
    
    def adjust_contrast_brightness(self, img: np.ndarray, contrast: float=1.0, bright: float=0.0):
        # img float [0..1]
        return np.clip(img * contrast + bright, 0, 1)
    
    def blend(self, a: np.ndarray, b: np.ndarray, alpha: float):
        return np.clip(a * (1-alpha) + b * alpha, 0, 1)

    def vignette_mask(self, shape: Tuple[int,int], strength=0.5):
        rows, cols = shape
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols*strength)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows*strength)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        return mask
    
    
    def bilateral_smooth(self,img: np.ndarray, d=9, sigmaColor=75, sigmaSpace=75):
        bgr = self.ensure_uint8(img)
        sm = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return self.ensure_float(sm)

    def unsharp_mask(self,img: np.ndarray, strength=1.0, radius=1):
        bgr = self.ensure_uint8(img)
        blur = cv2.GaussianBlur(bgr, (0,0), radius)
        sharp = cv2.addWeighted(bgr, 1.0 + strength, blur, -strength, 0)
        return self.ensure_float(sharp)

    def gaussian_glow(self,img: np.ndarray, sigma=10, glow_strength=0.25):
        blur = cv2.GaussianBlur(self.ensure_uint8(img), (0,0), sigma)
        return self.blend(img, self.ensure_float(blur), glow_strength)

    def sepia_tone(self,img: np.ndarray, intensity=0.5):
        t = np.array([[0.393,0.769,0.189],
                    [0.349,0.686,0.168],
                    [0.272,0.534,0.131]], dtype=np.float32)
        out = img.copy()
        out = np.tensordot(out, t.T, axes=([2],[0]))
        out = self.blend(img, out, intensity)
        return np.clip(out, 0, 1)

    def rgb_curve(self,img: np.ndarray, r_curve=1.0, g_curve=1.0, b_curve=1.0):
        # simple per-channel gamma-like scaling
        out = img.copy()
        out[:,:,2] = np.power(out[:,:,2], 1.0 / r_curve)  # R
        out[:,:,1] = np.power(out[:,:,1], 1.0 / g_curve)  # G
        out[:,:,0] = np.power(out[:,:,0], 1.0 / b_curve)  # B
        return np.clip(out, 0, 1)

    def color_shift(self,img: np.ndarray, shift=(0.0,0.0,0.0)):
        # shift BGR channels by percentages
        out = img.copy()
        out[:,:,0] = np.clip(img[:,:,0] + shift[0], 0, 1)
        out[:,:,1] = np.clip(img[:,:,1] + shift[1], 0, 1)
        out[:,:,2] = np.clip(img[:,:,2] + shift[2], 0, 1)
        return out

    def contrast_s_curve(self,img: np.ndarray, amount=0.5):
        # soft S curve via LUT approximation
        x = np.linspace(0,1,256)
        s = 1/(1 + np.exp(-12*(x-0.5)))  # sigmoid centered
        lut = ((1-amount)*x + amount*s)
        # apply per channel
        out = img.copy()
        for c in range(3):
            out[:,:,c] = np.interp(out[:,:,c], x, lut)
        return np.clip(out, 0, 1)

    def posterize(self,img: np.ndarray, bits=4):
        levels = 2 ** bits
        out = np.floor(img * levels) / (levels-1)
        return np.clip(out, 0, 1)
    
    
    def filter_warm(self,img: np.ndarray, intensity=0.7):
        s = self.bilateral_smooth(img, d=9, sigmaColor=70, sigmaSpace=70)
        t = self.rgb_curve(s, r_curve=0.9 - 0.05*intensity, g_curve=1.0, b_curve=1.1 + 0.05*intensity)
        out = self.adjust_contrast_brightness(t, contrast=1.05 + 0.15*intensity, bright=0.02*intensity)
        return out
    

    def filter_cool(self,img: np.ndarray, intensity=0.7):
        s = self.bilateral_smooth(img, d=9, sigmaColor=70, sigmaSpace=70)
        t = self.rgb_curve(s, r_curve=1.05, g_curve=1.0, b_curve=0.9 - 0.05*intensity)
        out = self.adjust_contrast_brightness(t, contrast=1.02 + 0.12*intensity, bright=0.0)
        return out
    
    def filter_cinematic(self, img: np.ndarray, intensity=0.8):
        s = self.bilateral_smooth(img, d=7, sigmaColor=60, sigmaSpace=60)
        teal = self.color_shift(s, shift=(0.03*intensity, 0.0, -0.02*intensity))
        matt = self.contrast_s_curve(teal, amount=0.35*intensity)
        # add subtle letterbox matte (top-bottom)
        rows, cols, _ = matt.shape
        mask = self.vignette_mask((rows, cols), strength=0.5)
        mask = cv2.merge([mask, mask, mask])
        out = self.blend(matt * mask, matt, 0.25*intensity)
        return out
    def filter_faded(self,img: np.ndarray, intensity=0.9):
        s = self.bilateral_smooth(img, d=9, sigmaColor=60, sigmaSpace=60)
        faded = self.adjust_contrast_brightness(s, contrast=0.85, bright=0.04*intensity)
        out = self.rgb_curve(faded, r_curve=1.05, g_curve=1.02, b_curve=1.02)
        return out
    
    def filter_vintage_film(self,img: np.ndarray, intensity=0.8):
        s = self.bilateral_smooth(img, d=7, sigmaColor=60, sigmaSpace=60)
        sep = self.sepia_tone(s, intensity=0.45*intensity)
        grain = (np.random.randn(*sep.shape) * 0.03 * (1-intensity) + sep)
        out = self.adjust_contrast_brightness(grain, contrast=1.05, bright=0.02)
        return np.clip(out,0,1)
    
    
    def filter_low_key(self,img: np.ndarray, intensity=0.7):
        out = self.adjust_contrast_brightness(img, contrast=1.1 + 0.2*intensity, bright=-0.05*intensity)
        mask = self.vignette_mask((img.shape[0], img.shape[1]), strength=0.6)
        mask = cv2.merge([mask,mask,mask])
        return out * mask

    def filter_punch(self,img: np.ndarray, intensity=0.5):
        # strong saturation + contrast
        hsv = cv2.cvtColor(self.ensure_uint8(img), cv2.COLOR_BGR2HSV).astype(np.float32)/255.0
        h,s,v = cv2.split(hsv)
        s = np.clip(s * (1.0 + 0.7*intensity), 0, 1)
        v = np.clip(v * (1.0 + 0.2*intensity), 0, 1)
        out = cv2.cvtColor((cv2.merge([h,s,v])*255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self.ensure_float(out)
    
    
    def filter_soft_blue(self,img: np.ndarray, intensity=0.6):
        return self.blend(img, self.color_shift(img, shift=(0.06*intensity, 0.0, -0.02*intensity)), 0.35*intensity)
    
    def filter_punchy_cinematic(self,img: np.ndarray, intensity=0.9):
        c = self.filter_cinematic(img, intensity=intensity)
        return self.unsharp_mask(c, strength=0.8*intensity, radius=1)
    
    def filter_emboss(self,img: np.ndarray, intensity=0.6):
        k = np.array([[ -2,-1,0],[-1,1,1],[0,1,2]])
        bgr = self.ensure_uint8(img)
        emb = cv2.filter2D(bgr, -1, k).astype(np.float32)/255.0
        return self.blend(img, emb, intensity)

    def filter_vintage2(self,img: np.ndarray, intensity=0.8):
        s = self.filter_vintage_film(img, intensity=intensity)
        return self.blend(img, s, intensity)

    
if __name__ == "__main__":
    f = ImageFilter()
    f.apply_filter("warm", "image-utils/input.jpg", outpath="output.png", intensity=0.8)
