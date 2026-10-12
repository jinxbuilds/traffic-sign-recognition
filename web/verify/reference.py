import numpy as np, cv2, json, sys
from skimage import exposure
rng=np.random.default_rng(0)
cases=[]
sizes=[(32,32),(64,64),(50,47),(120,90),(31,33),(200,300),(17,40),(640,480)]
for k,(w,h) in enumerate(sizes*3):
    if k%3==0: img=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    elif k%3==1:
        yy,xx=np.mgrid[0:h,0:w]; img=np.stack([(xx*255//max(w-1,1)),(yy*255//max(h-1,1)),((xx+yy)%256)],-1).astype(np.uint8)
    else:
        img=np.full((h,w,3),rng.integers(0,256,3),dtype=np.uint8); cv2.circle(img,(w//2,h//2),min(w,h)//3,(200,30,30),-1)
    small=cv2.resize(img,(32,32))
    g=(0.299*small[...,0]+0.587*small[...,1]+0.114*small[...,2])
    g=(g/255.).astype(np.float32)
    eq=exposure.equalize_adapthist(g)
    cases.append({"w":w,"h":h,"rgb":img.flatten().tolist(),"small":small.flatten().tolist(),"eq":eq.astype(np.float64).flatten().tolist()})
json.dump(cases,open('cases.json','w'))
print(len(cases))
