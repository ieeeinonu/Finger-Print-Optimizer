import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1) Görüntüyü yüklemek:
img = cv2.imread("fingerprint2.png", 0)   # 0 -> grayscale

# 2) fotoğrafın yüklenip yüklenmediği kontrol edilir:
if img is None:
    print("Hata: 'fingerprint.png' dosyası bulunamadı veya yüklenemedi. Dosya yolunu ve adını kontrol edin.")
else:
    plt.figure(figsize=(12,8))
    plt.subplot(2,4,1)
    plt.title("Original")
    plt.imshow(img, cmap="gray") #orijinal görüntü gösterilir
    plt.axis("off")

    # 2) Gürültüsü azaltılmış görüntüyü sunma:
    denoised = cv2.medianBlur(img, 3) # Gürültü azaltma burada yapılıyor (Median blur fiktered)
    plt.subplot(2,4,2) # yeni görüntünün gösterimi
    plt.title("Median Filtered")
    plt.imshow(denoised, cmap="gray")
    plt.axis("off")

    # 3) Morfolojik açma:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Önce görüntü matrisi ile çarpılacak kernel hazırlanır, 3x3 boyutunda olmak zorunda: 
    opening = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel) # Çarpma iler birlikte 
    plt.subplot(2,4,3) # yeni görüntünün gösterimi
    plt.title("Opening")
    plt.imshow(opening, cmap="gray")
    plt.axis("off")

    # 4) MORPHOLOGICAL CLOSING 
    closing = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel) # önceden olulturulan kernel bu sefer closing işlemi için kullanılmakta
    plt.subplot(2,4,4) # yeni görüntünün gösterimi
    plt.title("Closing")
    plt.imshow(closing, cmap="gray")
    plt.axis("off")

    # 5) Morfolojik Gradient:
    gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel) # önceden olulturulan kernel bu sefer gradient işlemi için kullanılmakta
    plt.subplot(2,4,5) # yeni görüntünün gösterimi
    plt.title("Gradient")
    plt.imshow(gradient, cmap="gray")
    plt.axis("off")

    # 6) BLACK-HAT (Ridge kontrastını artırır) 
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    plt.subplot(2,4,6) # yeni görüntünün gösterimi
    plt.title("Black-hat")
    plt.imshow(blackhat, cmap="gray")
    plt.axis("off")

    # 7) Kontrast Artırma:
    enhanced = cv2.add(denoised, blackhat)
    plt.subplot(2,4,7) # yeni görüntünün gösterimi
    plt.title("Enhanced FP")
    plt.imshow(enhanced, cmap="gray")
    plt.axis("off")

    # 8) ADAPTIVE THRESHOLD (Binary fingerprint maskesi): En önemli filtreleme, burada mask oluşturuluyor.
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    plt.subplot(2,4,8) # yeni görüntünün gösterimi
    plt.title("Binary Output")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
