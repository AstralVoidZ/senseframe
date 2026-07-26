import json, os
runs_dir = '<DEPLOY_ROOT>/runs'
for d in sorted(os.listdir(runs_dir)):
    if not d.startswith('ResNet18_UT_HAR'): continue
    path = os.path.join(runs_dir, d)
    meta_path = os.path.join(path, 'metadata.json')
    if not os.path.exists(meta_path): continue
    m = json.load(open(meta_path))
    r = m.get('resource', {})
    vp = r.get('vram_probe', {})
    has_probe = vp is not None and vp.get('measured_vram_mb') is not None
    c = m.get('config', {})
    seed = c.get('seed')
    prec = c.get('precision')
    csv_path = os.path.join(path, 'metrics/metrics.csv')
    if not os.path.exists(csv_path): continue
    lines = open(csv_path).read().strip().split('\n')[1:]
    ep0 = None
    for line in lines:
        parts = line.split(',')
        if len(parts) > 4 and parts[4]:
            try:
                ep0 = float(parts[4])
            except:
                pass
            break
    timestamp = d[20:35]
    print(f'{timestamp:20s} probe={str(has_probe):5s} seed={seed} prec={prec} ep0={ep0}')
