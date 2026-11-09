#!/usr/bin/env python3
"""Reproduce the report: .venv/bin/python docs/hsa_pipeline_report.py bench|pdf."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/hsa_pipeline_benchmark.json'
REVISIONS = {'initial': '8b9b27a', 'graph': 'a708c7c', 'final': 'dbf4743'}
MODES = [('cpu_hsa', 'initial', 'hsa', 'cpu-reference'),
         ('metal_initial', 'initial', 'hsa', 'metal'),
         ('metal_graph', 'graph', 'hsa', 'metal'),
         ('metal_final', 'final', 'hsa', 'metal'),
         ('native', 'final', 'off', 'cpu-reference'),
         ('stage1', 'final', 'stage1', 'cpu-reference')]

def benchmark():
    with tempfile.TemporaryDirectory(prefix='rocket-hsa-report-') as temp:
        work = Path(temp)
        for name, rev in REVISIONS.items():
            target = work / name
            paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', rev, 'include', 'src'], cwd=ROOT, text=True).splitlines()
            for path in paths:
                out = target / path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=ROOT))
            runner = target / 'src/runner.cpp'
            text = runner.read_text()
            assert text.count('!trace_logits || forced_tokens.size()') == 1
            runner.write_text(text.replace('!trace_logits || forced_tokens.size()', 'forced_tokens.size()'))
            shader = (target / 'src/metal/rocket_metal.metal').read_text()
            (target / 'rocket_metal_source.h').write_text('static const char* rocket_metal_source = R"MSL(' + shader + ')MSL";\n')
            flags = ['clang++', '-std=c++17', '-O3', '-Wno-unknown-pragmas', '-I'+str(target/'include'), '-I'+str(target), '-Illama.cpp/include', '-Illama.cpp/ggml/include', '-Illama.cpp/vendor']
            subprocess.run(flags + ['-fobjc-arc', '-c', str(target/'src/metal/rocket_metal.mm'), '-o', str(target/'metal.o')], cwd=ROOT, check=True)
            subprocess.run(flags + [str(target/'src'/f) for f in ['runner.cpp','rocket.cpp','rocket_adapter.cpp']] + [str(target/'metal.o'), '-o', str(target/'runner'), '-Lllama.cpp/build/bin', '-Wl,-rpath,'+str(ROOT/'llama.cpp/build/bin'), '-lllama','-lggml','-lggml-base','-lggml-cpu','-framework','Metal','-framework','Foundation'], cwd=ROOT, check=True)
            print('Built', name, rev, flush=True)
        source = ROOT / 'results/runtime_probe/samples.json'
        prompt = next(x['prompt'] for x in json.loads(source.read_text()) if x['sample_id'] == 201)
        inp = work / 'input.json'
        inp.write_text(json.dumps([dict(sample_id=201, prompt='\n\n'.join([prompt]*8), max_tokens=33, forced_tokens=list(range(1000,1033)))]))
        report = dict(date=time.strftime('%Y-%m-%d'), device='Apple M2 Ultra', model='Llama-3.2-3B-Instruct-Q8_0.gguf', revisions=REVISIONS,
                      input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      config=dict(gpu_layers=99, kv_type='f16', budget=1024, threads=16, forwards=32, trace=False, rounds=1, lengths=[2048,4096,8192,16384]),
                      pipeline_definition='ttft_ms + ms_per_token * decode_tokens; excludes model loading, tokenization, process startup and final cleanup; fixed tokens, no sampling',
                      build_note='Historical project sources linked against the same current patched llama.cpp build; CPU HSA uses initial revision, model layers remain on GPU.',
                      runs=[])
        if DATA.exists():
            previous = json.loads(DATA.read_text())
            assert previous['input_sha256'] == report['input_sha256']
            report['runs'] = [r for r in previous['runs'] if r['trial'] == 0]
            if 'historical_quality' in previous:
                report['historical_quality'] = previous['historical_quality']
        for length in report['config']['lengths']:
            for trial in range(1):
                # One pass in pipeline order; no repeatability claim.
                order = MODES
                for name, revision, mode, backend in order:
                    if any(r['context']==length and r['variant']==name and r['trial']==trial for r in report['runs']):
                        continue
                    out = work / 'output.json'
                    cmd = [str(work/revision/'runner'), '--model', str(ROOT/'models'/report['model']), '--batch', str(inp), '--output', str(out), '--rocket-mode', mode, '--stage2-backend', backend, '--gpu-layers','99','--budget','1024','--kv-type','f16','--n-ctx',str(length+1024),'--context-len',str(length),'--threads','16']
                    start = time.monotonic()
                    with (work/'run.log').open('w') as log:
                        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log, timeout=600)
                    if proc.returncode:
                        raise RuntimeError((work/'run.log').read_text()[-6000:])
                    result = json.loads(out.read_text())[0]
                    m = result['metrics']
                    assert result['status']=='ok' and m['prompt_tokens']==length and m['decode_forward_calls']==32
                    if mode=='hsa':
                        assert m['stage2_backend']==backend and m['stage2_layer_calls']==28*32
                    total = m['ttft_ms'] + m['ms_per_token']*m['decode_tokens']
                    report['runs'].append(dict(context=length, trial=trial, variant=name, metrics=m, pipeline_ms=total, process_wall_s=time.monotonic()-start))
                    DATA.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
                    print(f'{length:5} round={trial+1} {name:14} decode={m["decode_ms_per_forward"]:.3f}ms pipeline={total/1000:.3f}s', flush=True)
        report['status']='complete'
        DATA.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

if __name__ == '__main__' and sys.argv[1:] == ['bench']:
    benchmark()

def render_pdf():
    """Render two English vector infographics; no new measurements are run."""
    import math
    from reportlab.pdfgen import canvas
    from reportlab import rl_config
    from reportlab.lib.colors import HexColor, white

    d=json.loads(DATA.read_text())
    assert d.get('status')=='complete' and len(d['runs'])==24
    q=d['historical_quality']
    lengths=d['config']['lengths']
    rows={(r['context'],r['variant']):r for r in d['runs']}
    def value(n,v,pipeline=False):
        r=rows[n,v]
        return r['pipeline_ms']/1000 if pipeline else r['metrics']['decode_ms_per_forward']
    W,H=842,595
    out=ROOT/'docs/hsa_pipeline_report.pdf'
    rl_config.useA85 = False
    c=canvas.Canvas(str(out),pagesize=(W,H),pageCompression=1)
    c.setTitle('HSA: CPU to Metal | Optimization and baseline comparison')
    c.setAuthor('RocketKV project')
    navy='#17364B';gray='#536879';teal='#087F8C';orange='#B67D32';blue='#517CAC';red='#AD5961';grid='#DCE5EB'
    colors={'cpu_hsa':red,'metal_initial':orange,'metal_graph':blue,'metal_final':teal,'native':navy}
    names={'cpu_hsa':'CPU HSA','metal_initial':'Initial Metal','metal_graph':'KV / graph','metal_final':'Final Metal','native':'Native baseline'}
    phases=['cpu_hsa','metal_initial','metal_graph','metal_final']
    def text(x,y,s,size=10,color=navy,bold=False,align='left'):
        c.setFillColor(HexColor(color));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        {'left':c.drawString,'right':c.drawRightString,'center':c.drawCentredString}[align](x,y,str(s))
    def lines(x,y,items,size=9,leading=12,color=gray):
        for i,s in enumerate(items):text(x,y-i*leading,s,size,color)
    def rect(x,y,w,h,fill='#F0F5F8',stroke=None,r=8):
        c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke or fill))
        c.roundRect(x,y,w,h,r,fill=1,stroke=bool(stroke))
    def line(x1,y1,x2,y2,color=grid,width=.6):
        c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x1,y1,x2,y2)
    def dot(x,y,color,r=2.7):
        c.setFillColor(HexColor(color));c.circle(x,y,r,fill=1,stroke=0)
    def legend(x,y,items,spacing=118):
        for i,(label,color) in enumerate(items):
            line(x+i*spacing,y+3,x+i*spacing+14,y+3,color,2.5)
            text(x+i*spacing+20,y,label,8.5,gray)
    def header(number,title,subtitle):
        text(36,568,'ROCKETKV  /  ENGINEERING RESULTS',8,teal,True)
        text(806,568,f'{number} / 2',8,gray,align='right')
        text(36,538,title,24,navy,True)
        text(36,520,subtitle,9,gray)
    def footer(items):
        line(36,39,806,39)
        for i,s in enumerate(items):text(36,26-i*10,s,7.1,gray)

    header(1,'HSA: from CPU to Metal','One model, four implementation stages. Decode latency is measured over the entire model, not an isolated HSA kernel.')
    algorithms=[['CPU scoring, Top-K and','exact sparse attention.'],['Move HSA kernels to GPU;','retain per-layer staging / waits.'],['Direct F16 KV + cached chunks;','threadgroup sort + GGML encoder.'],['Chunk sorting, 64-thread groups,','four-V loads and SIMD shuffle.']]
    commits=['8b9b27a / CPU backend','8b9b27a / Metal backend','a708c7c','dbf4743']
    xs=[36,233,430,627];cardw=179
    for i,v in enumerate(phases):
        x=xs[i];rect(x,349,cardw,154)
        rect(x,499,cardw,4,colors[v],r=0)
        text(x+12,481,names[v],13,colors[v],True)
        text(x+12,468,commits[i],7.5,gray)
        lines(x+12,451,algorithms[i],8,11)
        for j,n in enumerate(lengths):
            y=418-j*16
            text(x+12,y,f'{n//1024}K',9,gray)
            text(x+cardw-12,y,f'{value(n,v):.2f} ms',10,navy,True,align='right')
        if i<3:text(x+cardw+9,428,'>',15,gray,True,align='center')

    text(36,326,'WHOLE PIPELINE LATENCY',11,navy,True)
    text(806,326,'Prefill + Stage 1 + initialization + 32 decode steps | seconds, lower is better',8,gray,align='right')
    left,right,bottom,top=73,586,178,298
    maxy=50
    for t in range(0,51,10):
        y=bottom+t/maxy*(top-bottom)
        line(left,y,right,y);text(left-9,y-3,t,8,gray,align='right')
    plotxs=[left+i*(right-left)/3 for i in range(4)]
    for x,n in zip(plotxs,lengths):text(x,bottom-14,f'{n//1024}K',9,navy,align='center')
    for v in phases:
        ys=[bottom+value(n,v,True)/maxy*(top-bottom) for n in lengths]
        for i in range(3):line(plotxs[i],ys[i],plotxs[i+1],ys[i+1],colors[v],1.8)
        for x,y in zip(plotxs,ys):dot(x,y,colors[v])
    legend(83,146,[(names[v],colors[v]) for v in phases],126)
    text(627,294,'CPU -> FINAL',10,teal,True)
    text(627,279,'Pipeline time (seconds)',8,gray)
    for i,n in enumerate(lengths):
        a,b=value(n,'cpu_hsa',True),value(n,'metal_final',True)
        text(627,261-i*23,f'{n//1024}K',9,gray)
        text(658,261-i*23,f'{a:.2f} -> {b:.2f}',9,navy,True)
        text(806,261-i*23,f'{a/b:.2f}x',9,teal,True,align='right')

    text(36,120,'FINAL KERNEL REFINEMENTS',10,navy,True)
    text(806,120,'Incremental decode-time reduction | prior 3-round A/B | 8K / 16K',8,gray,align='right')
    refinements=[('Sort chunks, expand IDs',(32.46559,26.69697,17.77),(50.28675,26.78578,46.73)),
                 ('64-thread weighted sum',(26.60859,26.12062,1.83),(26.84113,26.25769,2.17)),
                 ('Preload four V values',(26.16825,21.476125,17.93),(26.24678,21.75716,17.11)),
                 ('SIMD shuffle sorting',(21.57825,21.31903,1.20),(21.72072,21.35416,1.69))]
    for x,(title,eight,sixteen) in zip(xs,refinements):
        rect(x,49,cardw,59)
        text(x+10,92,title,9,navy,True)
        for y,label,(before,after,gain) in [(76,'8K',eight),(61,'16K',sixteen)]:
            text(x+10,y,f'{label}: {before:.2f} -> {after:.2f} ms',8,navy)
            text(x+cardw-9,y,f'-{gain:.2f}%',8,teal,True,align='right')
    footer(['Latency: one run per stage / length; identical forced tokens; M2 Ultra, Llama-3.2-3B Q8_0, F16 KV, GPU layers=99, budget=1024, threads=16.',
            'Model stays on GPU for CPU HSA. Pipeline excludes loading / tokenization / sampling. Each kernel card is a separate A/B. Sources: benchmark JSON + HSA_OPTIMIZATION_LOG.md.'])
    c.showPage()

    header(2,'Final HSA vs native baseline','Latency: final Metal dbf4743. Quality: historical HSA 8b9b27a only; final-version task accuracy has not been measured.')
    def barpanel(x,w,pipeline):
        rect(x,288,w,214)
        text(x+14,483,'Pipeline latency (s)' if pipeline else 'Decode latency (ms / forward)',12,navy,True)
        legend(x+14,466,[('Native baseline',navy),('Final HSA',teal)],143)
        l,r,b,t=x+37,x+w-14,333,446
        ymax=14 if pipeline else 25
        ticks=[0,5,10] if pipeline else [0,10,20]
        for tick in ticks:
            y=b+tick/ymax*(t-b);line(l,y,r,y);text(l-7,y-3,tick,7.5,gray,align='right')
        for j,n in enumerate(lengths):
            center=l+(j+.5)*(r-l)/4
            a,z=value(n,'native',pipeline),value(n,'metal_final',pipeline)
            for xx,v,col in [(center-20,a,navy),(center+2,z,teal)]:
                h=v/ymax*(t-b);rect(xx,b,18,h,col,r=0)
                text(xx+9,b+h+5,f'{v:.2f}',8,col,True,align='center')
            text(center,b-14,f'{n//1024}K',9,navy,align='center')
            text(center,b-29,f'{z/a:.2f}x',8.5,teal,True,align='center')
        text(x+14,294,'Ratio below each group = final / baseline latency (>1 means slower).',7.5,gray)
    barpanel(36,376,False);barpanel(430,376,True)

    text(36,266,'QUALITY / TASK SCORES (0-100)',11,navy,True)
    text(806,266,'30 samples per task | 120 total | context capped at 8K | higher is better',8,gray,align='right')
    rect(36,72,541,183)
    text(50,237,'Historical comparison - NOT final-version accuracy',10,orange,True)
    legend(50,219,[('Native baseline',navy),('Historical HSA',orange)],161)
    l,r,b,t=73,560,122,205
    for tick in [0,40,80]:
        y=b+tick/80*(t-b);line(l,y,r,y);text(l-8,y-3,tick,7.5,gray,align='right')
    tasks=['multi_news','trec','passage_retrieval_en','repobench-p']
    labels=[('Multi-News','ROUGE-L'),('TREC','Classification'),('Retrieval','Retrieval score'),('RepoBench-P','Code similarity')]
    for j,(task,(label,metric)) in enumerate(zip(tasks,labels)):
        center=l+(j+.5)*(r-l)/4
        a,z=q['scores']['off'][task],q['scores']['hsa'][task]
        for xx,v,col in [(center-23,a,navy),(center+3,z,orange)]:
            h=v/80*(t-b);rect(xx,b,21,h,col,r=0)
            text(xx+10.5,b+h+5,f'{v:.2f}',8,col,True,align='center')
        text(center,b-15,label,8.5,navy,True,align='center')
        text(center,b-27,metric,7.5,gray,align='center')
        text(center,b-40,f'{z-a:+.2f} points',8,orange,align='center')
    rect(595,72,211,183,fill='#FFF5E9')
    text(609,235,'FINAL-VERSION ACCURACY',10,orange,True)
    text(609,211,'Not yet measured',17,navy,True)
    line(609,197,791,197,'#E5CEAB')
    text(609,180,'Existing numerical check',10,navy,True)
    lines(609,164,['8K / 16K / 32K, 4 fixed forwards:','attention + logits match the','pre-optimization Metal version.'],9,14)
    text(609,108,'Numerical parity is not task accuracy.',8.5,orange,True)
    text(36,53,'Result: GPU migration accelerates HSA, but final HSA still has higher latency than native full attention.',10,navy,True)
    footer(['Latency and quality use different workloads and versions; they are not a paired accuracy-latency trade-off experiment. No new quality evaluation was run.',
            'Sources: docs/hsa_pipeline_benchmark.json (latency + attributed historical scores); HSA_OPTIMIZATION_LOG.md (kernel A/B and parity).'])
    c.save()
    print(out)

if __name__ == '__main__' and sys.argv[1:] == ['pdf']:
    render_pdf()
