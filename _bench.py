import importlib.util, os, shutil, time, warnings, multiprocessing as mp
import cv2, lmdb, numpy as np
import fmg, fmg.core

BASE = os.path.abspath('_bench_data')
shutil.rmtree(BASE, ignore_errors=True); os.makedirs(BASE)

real_open = lmdb.open
def ropen(path, **kw):
    # The sandbox's virtiofs mount sporadically returns EIO while setting up
    # an mmap.  py-lmdb defaults to create=True, so treat every non-readonly
    # open as a creating one and clear half-created files before retrying.
    creating = not bool(kw.get('readonly', False))
    for _ in range(200):
        try: return real_open(path, **kw)
        except lmdb.Error as e:
            if 'Input/output' not in str(e): raise
            if creating:
                for f in ('data.mdb','lock.mdb'):
                    try: os.remove(os.path.join(path,f))
                    except OSError: pass
            time.sleep(0.02)
    raise RuntimeError('EIO')
fmg.core.lmdb.open = ropen

spec = importlib.util.spec_from_file_location('legacy_core', '/data/inputs/core.py')
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)

h = w = 1024
yy = np.arange(h)[:,None]; xx = np.arange(w)[None,:]
img = np.empty((h,w,3), np.uint8)
img[...,0] = ((xx//8)*11+(yy//16)*7)%256
img[...,1] = ((yy//8)*9+(xx//32)*5)%256
img[...,2] = (((xx//16)+(yy//16))*17)%256
img = np.ascontiguousarray(img)

# ---- write benchmark ----
legacy_writer = legacy.FastMemoryGraphics(os.path.join(BASE,'old.fmg'),'w')
t=time.time(); legacy_writer.save_from_numpy(
    img, ext='.png', params=[], mpp=0.25, pyramid=False)
t_old_w=time.time()-t
legacy_writer.close()
t=time.time(); fmg.create_slide(os.path.join(BASE,'new.fmg')).save_from_numpy(
    img, ext='.png', pyramid=False, mpp=0.25)
t_new_w=time.time()-t
print('WRITE  old=%.2fs new=%.2fs'%(t_old_w,t_new_w))

# ---- read benchmark (single process, 200 random 512x512 regions) ----
pts=[(int(x),int(y)) for x,y in zip(np.random.default_rng(0).integers(0,512,200),
                                    np.random.default_rng(1).integers(0,512,200))]
old = legacy.FastMemoryGraphics(os.path.join(BASE,'old.fmg'),'r')
old.read_region((0,0),0,(64,64))
t=time.time()
for p in pts: old.read_region(p,0,(512,512))
t_old_r=time.time()-t
new0 = fmg.open_slide(os.path.join(BASE,'new.fmg'), patch_cache=0)
new0.read_region((0,0),0,(64,64))
t=time.time()
for p in pts: new0.read_region(p,0,(512,512))
t_new_r0=time.time()-t
new1 = fmg.open_slide(os.path.join(BASE,'new.fmg'), patch_cache=256)
new1.read_region((0,0),0,(64,64))
t=time.time()
for p in pts: new1.read_region(p,0,(512,512))
t_new_r1=time.time()-t
print('READ   old=%.2fs new(cache off)=%.2fs new(cache on)=%.2fs'%(t_old_r,t_new_r0,t_new_r1))

# ---- legacy fork reproducer ----
def worker(slide, q, n, seed):
    try:
        rng=np.random.default_rng(seed)
        for _ in range(n):
            x=int(rng.integers(0,512)); y=int(rng.integers(0,512))
            a=np.asarray(slide.read_region((x,y),0,(512,512)))
            if a.shape!=(512,512,3): raise AssertionError('bad shape')
        q.put(('ok',os.getpid()))
    except BaseException as e:
        q.put(('fail',os.getpid(),repr(e)))

ctx=mp.get_context('fork'); q=ctx.Queue()
ps=[ctx.Process(target=worker,args=(old,q,300,i)) for i in range(4)]
for p in ps: p.start()
res=[]
for _ in ps:
    try: res.append(q.get(timeout=90))
    except Exception as e: res.append(('timeout','',repr(e)))
for p in ps: p.join(timeout=10)
print('LEGACY FORK exitcodes:', [p.exitcode for p in ps])
print('LEGACY FORK results:', res)
shutil.rmtree(BASE, ignore_errors=True)
