"""Run an end-to-end SYNTHETIC demonstration of FCVG101 Task 1B.

Run from the project root: python demo_task1b.py
Replace data/demo_scene.png with permitted own images for coursework evidence.
"""
from pathlib import Path
import json
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.classical_cv import (
  create_demo_scene, make_same_scene_variant, make_unrelated_scene,
  add_gaussian_noise,add_salt_pepper_noise,filtering_outputs,
  mse,psnr,frequency_outputs,edges_outputs,harris_points,
  sift_match, homography_inliers, draw_harris, to_gray,bgr_to_rgb)

OUT=Path('results'); DATA=Path('data'); OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
base=create_demo_scene(); h,w=base.shape[:2]
variant=make_same_scene_variant(base)
unrelated=make_unrelated_scene(h,w)
for name,arr in [('demo_scene.png',base),('same_scene_transformed.png',variant),('unrelated_scene.png',unrelated)]:
  cv2.imwrite(str(DATA/name),arr)
clean=to_gray(base)
gauss=add_gaussian_noise(clean, sigma=20, seed=42)
sp=add_salt_pepper_noise(clean, proportion=.05, seed=43)
rows=[]
fig,axs=plt.subplots(2,5,figsize=(13.5,5.1),constrained_layout=True)
for row,(typ,noisy) in enumerate([('Gaussian noise (sigma=20)',gauss),('Salt & pepper (5%)',sp)]):
  results=filtering_outputs(noisy,5,1.2)
  imgs=[clean,noisy,*results.values()]
  titles=['Original','Noisy',*results.keys()]
  for ax,img,title in zip(axs[row],imgs,titles):
    ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=9);ax.axis('off')
  for method,img in [('Noisy',noisy),*results.items()]:
    rows.append({'noise':typ,'method':method,'MSE':round(mse(clean,img),3),'PSNR_dB':round(psnr(clean,img),3)})
fig.suptitle('Task 1B - Spatial filtering on synthetic controlled noise',fontsize=12)
fig.savefig(OUT/'01_filter_comparison.png',dpi=150);plt.close(fig)

freq=frequency_outputs(clean,cutoff=26,gauss_sigma=25)
fig,axs=plt.subplots(2,4,figsize=(13.5,6.4),constrained_layout=True)
examples=[('Original',clean,'gray',None),('Log FFT spectrum',freq['spectrum_log'],'magma',None),
          ('Ideal low-pass mask',freq['ideal_low_mask'],'gray',None),('Ideal high-pass mask',freq['ideal_high_mask'],'gray',None),
          ('Ideal low-pass result',freq['ideal_low_img'],'gray',(0,255)),
          ('Ideal high-pass (signed)',freq['ideal_high_img'],'gray',(-100,100)),
          ('Gaussian low-pass mask',freq['gaussian_low_mask'],'gray',None),
          ('Gaussian low-pass result',freq['gaussian_low_img'],'gray',(0,255))]
for ax,(title,img,cmap,lims) in zip(axs.flat,examples):
  kw={'cmap':cmap}
  if lims is not None: kw.update(vmin=lims[0],vmax=lims[1])
  ax.imshow(img,**kw);ax.set_title(title,fontsize=9);ax.axis('off')
fig.suptitle('Task 1B - Frequency-domain filtering (FFT)',fontsize=12)
fig.savefig(OUT/'02_frequency_domain.png',dpi=150);plt.close(fig)

edges=edges_outputs(clean)
fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for ax,(title,im,cmap) in zip(axs.flat,[('Original',clean,'gray'),('Smoothed',edges['smoothed'],'gray'),
             ('Sobel Gx (signed)',edges['gx'],'coolwarm'),('Sobel Gy (signed)',edges['gy'],'coolwarm'),
             ('Gradient magnitude',edges['magnitude'],'gray'),('Canny (80,160)',edges['canny'],'gray')]):
  ax.imshow(im,cmap=cmap);ax.set_title(title,fontsize=10);ax.axis('off')
fig.suptitle('Task 1B - Gradients and edge detection',fontsize=12)
fig.savefig(OUT/'03_edges.png',dpi=150);plt.close(fig)

response,pts=harris_points(clean)
fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
axs[0].imshow(bgr_to_rgb(draw_harris(base,pts)));axs[0].set_title(f'Harris corner markers (n={len(pts)})')
axs[1].imshow(np.maximum(response,0),cmap='hot');axs[1].set_title('Harris positive response')
for a in axs:a.axis('off')
fig.savefig(OUT/'04_harris.png',dpi=150);plt.close(fig)

k1,k2,match12=sift_match(base,variant)
_,k3,match13=sift_match(base,unrelated)
H,inliers=homography_inliers(k1,k2,match12)
vis=cv2.drawMatches(base,k1,variant,k2,match12[:35],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
ax.imshow(bgr_to_rgb(vis));ax.set_title(f'SIFT same scene: {len(match12)} ratio-test matches; RANSAC inliers {inliers}')
ax.axis('off');fig.savefig(OUT/'05_sift_matches.png',dpi=150);plt.close(fig)
with (OUT/'metrics_filtering.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=['noise','method','MSE','PSNR_dB']);writer.writeheader();writer.writerows(rows)
summary={'type':'SYNTHETIC DEMO; replace with own data for coursework',
         'image_size':[w,h],'harris_candidates':len(pts),
         'sift_kp_original':len(k1),'sift_kp_transformed':len(k2),
         'sift_kp_unrelated':len(k3),'good_matches_same':len(match12),
         'good_matches_unrelated':len(match13),'ransac_inliers_same':inliers,
         'filtering_metrics':rows}
(OUT/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != 'filtering_metrics'},ensure_ascii=False,indent=2))
print('Saved figures and metrics to:',OUT.resolve())
