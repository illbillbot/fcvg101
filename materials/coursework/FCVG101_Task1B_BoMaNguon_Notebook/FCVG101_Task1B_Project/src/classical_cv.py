"""Task 1B: filtering, Fourier, edge and feature matching helpers.

Includes transparent, reusable functions for an FCVG101 coursework project.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def load_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Không đọc được ảnh: {path}')
    return image


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def create_demo_scene(height=320, width=480, seed=42):
    """Generate clearly labeled SYNTHETIC objects and reproducible texture."""
    rng = np.random.default_rng(seed)
    im = np.full((height, width, 3), (224, 219, 211), np.uint8)
    # textured workspace and dark margins
    cv2.rectangle(im, (12, 14), (width-12, height-14), (242, 238, 231), -1)
    cv2.rectangle(im, (12, 14), (width-12, height-14), (84, 79, 73), 2)
    cv2.rectangle(im, (38, 55), (208, 248), (238, 246, 254), -1)
    cv2.rectangle(im, (38, 55), (208, 248), (41, 67, 98), 3)
    cv2.putText(im, 'APPLE 01', (49, 82), cv2.FONT_HERSHEY_SIMPLEX, .46, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.circle(im, (123, 151), 52, (43, 59, 218), -1)
    cv2.circle(im, (104, 134), 13, (91, 112, 249), -1)
    cv2.ellipse(im, (125, 93), (21, 9), -30, 0, 360, (56, 137, 60), -1)
    cv2.putText(im, 'QA-27', (88, 227), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 25, 25), 2, cv2.LINE_AA)
    cv2.rectangle(im, (252, 57), (440, 247), (239, 251, 236), -1)
    cv2.rectangle(im, (252, 57), (440, 247), (49, 99, 71), 3)
    cv2.putText(im, 'ORANGE 02', (267, 83), cv2.FONT_HERSHEY_SIMPLEX, .46, (45, 55, 45), 1, cv2.LINE_AA)
    cv2.circle(im, (341, 151), 53, (26, 145, 239), -1)
    cv2.circle(im, (324, 133), 13, (95, 184, 246), -1)
    cv2.ellipse(im, (351, 96), (21, 8), 23, 0, 360, (55, 133, 42), -1)
    for k in range(8):
        xx=285+18*k
        cv2.line(im, (xx, 215), (xx, 231), (55, 55, 55), 1 if k%2 else 2)
    cv2.putText(im, 'FCVG101', (170, 285), cv2.FONT_HERSHEY_SIMPLEX, .78, (41, 60, 70), 2, cv2.LINE_AA)
    # texture to allow keypoint detection, reproducible but not random noise everywhere
    for _ in range(95):
        x=int(rng.integers(42, width-42)); y=int(rng.integers(53, height-55))
        col=(int(rng.integers(50,200)),)*3
        cv2.circle(im,(x,y),int(rng.integers(1,3)),col,-1)
    for x in range(16, width-15, 22):
        cv2.line(im, (x, 20), (x+5, 20), (66, 71, 76), 1)
    return im


def make_same_scene_variant(bgr):
    h,w=bgr.shape[:2]
    A=cv2.getRotationMatrix2D((w/2,h/2), 12, 0.87)
    A[0,2] += 7
    A[1,2] += 5
    return cv2.warpAffine(bgr,A,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(224,219,211))


def make_unrelated_scene(h,w):
    rng=np.random.default_rng(114)
    out=np.full((h,w,3),(244,231,220),np.uint8)
    for idx in range(20):
        x=int(rng.integers(15,w-15)); y=int(rng.integers(15,h-15))
        color=tuple(int(z) for z in rng.integers(30,220,size=3))
        cv2.line(out,(x,0),(max(0,x-60),h-1),color,2)
        cv2.circle(out,(x,y),int(rng.integers(4,10)),color,-1)
    cv2.putText(out,'UNRELATED', (78,h//2),cv2.FONT_HERSHEY_SIMPLEX,1.5,(10,10,10),4,cv2.LINE_AA)
    return out


def add_gaussian_noise(gray, sigma=20.0, seed=42):
    rng=np.random.default_rng(seed)
    noise=rng.normal(0,sigma,gray.shape)
    return np.clip(gray.astype(np.float32)+noise,0,255).astype(np.uint8)


def add_salt_pepper_noise(gray, proportion=0.05, seed=43):
    rng=np.random.default_rng(seed)
    out=gray.copy()
    r=rng.random(gray.shape)
    out[r < proportion/2]=0
    out[(r >= proportion/2) & (r < proportion)]=255
    return out


def filtering_outputs(noisy, kernel=5, sigma=1.2):
    if kernel<3 or kernel%2==0:
        raise ValueError('kernel must be odd and >= 3')
    return {
       'Box':cv2.blur(noisy,(kernel,kernel)),
       'Gaussian':cv2.GaussianBlur(noisy,(kernel,kernel),sigma),
       'Median':cv2.medianBlur(noisy,kernel)
    }


def mse(ref,pred):
    if ref.shape != pred.shape:
        raise ValueError('Images must have matching shapes')
    d=ref.astype(np.float64)-pred.astype(np.float64)
    return float(np.mean(d*d))


def psnr(ref,pred):
    e=mse(ref,pred)
    return float('inf') if e==0 else float(10*np.log10(255*255/e))


def frequency_outputs(gray, cutoff=26, gauss_sigma=25):
    src=gray.astype(np.float64)
    shifted=np.fft.fftshift(np.fft.fft2(src))
    yy,xx=np.ogrid[:gray.shape[0],:gray.shape[1]]
    r2=(yy-gray.shape[0]//2)**2+(xx-gray.shape[1]//2)**2
    ideal_low=(r2<=cutoff**2).astype(np.float64)
    ideal_high=1-ideal_low
    soft_low=np.exp(-r2/(2*gauss_sigma**2))
    def inv(mask):
        return np.fft.ifft2(np.fft.ifftshift(shifted*mask)).real
    return {
        'spectrum_log':np.log1p(np.abs(shifted)),
        'ideal_low_mask':ideal_low,
        'ideal_high_mask':ideal_high,
        'gaussian_low_mask':soft_low,
        'ideal_low_img':inv(ideal_low),
        'ideal_high_img':inv(ideal_high),
        'gaussian_low_img':inv(soft_low),
    }


def edges_outputs(gray, blur_kernel=5, sigma=1.2, low=80, high=160):
    smoothed=cv2.GaussianBlur(gray,(blur_kernel,blur_kernel),sigma)
    gx=cv2.Sobel(smoothed,cv2.CV_64F,1,0,ksize=3)
    gy=cv2.Sobel(smoothed,cv2.CV_64F,0,1,ksize=3)
    mag=np.hypot(gx,gy)
    orient=np.arctan2(gy,gx)
    canny=cv2.Canny(smoothed,low,high)
    return {'smoothed':smoothed,'gx':gx,'gy':gy,'magnitude':mag,
            'orientation':orient,'canny':canny}


def harris_points(gray, max_points=80, quality=0.035):
    g=np.float32(gray)/255
    r=cv2.cornerHarris(g,blockSize=3,ksize=3,k=0.04)
    dilated=cv2.dilate(r,None)
    cand=(r==dilated)&(r>quality*max(float(r.max()),0))
    yy,xx=np.where(cand)
    order=np.argsort(r[yy,xx])[::-1][:max_points]
    return r, [(int(xx[i]),int(yy[i])) for i in order]


def sift_match(bgr_a,bgr_b,ratio=0.75):
    s=cv2.SIFT_create()
    ga=to_gray(bgr_a);gb=to_gray(bgr_b)
    k1,d1=s.detectAndCompute(ga,None)
    k2,d2=s.detectAndCompute(gb,None)
    matches=[]
    if d1 is not None and d2 is not None and len(d2)>=2:
        knn=cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1,d2,k=2)
        matches=[p[0] for p in knn if len(p)==2 and p[0].distance < ratio*p[1].distance]
    return k1,k2,matches


def homography_inliers(kp1,kp2,matches):
    if len(matches)<4:
        return None,0
    a=np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    b=np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H,mask=cv2.findHomography(a,b,cv2.RANSAC,5.0)
    return H,(int(mask.sum()) if mask is not None else 0)


def draw_harris(bgr,pts):
    out=bgr.copy()
    for x,y in pts:
        cv2.circle(out,(x,y),4,(0,0,255),1,cv2.LINE_AA)
    return out
