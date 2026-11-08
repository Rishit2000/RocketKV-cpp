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
    from html import escape
    import math
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.graphics.shapes import Drawing, Line, String, Circle

    d = json.loads(DATA.read_text())
    assert d.get('status') == 'complete' and len(d['runs']) == 24
    lengths = d['config']['lengths']
    rows = {(r['context'],r['variant']):r for r in d['runs']}
    names = {'cpu_hsa':'CPU HSA', 'metal_initial':'初版 Metal', 'metal_graph':'KV / graph 优化', 'metal_final':'当前 Metal', 'native':'原生 baseline', 'stage1':'仅 Stage 1'}
    order = [x[0] for x in MODES]
    phase = order[:4]
    def metric(n, v, kind='decode'):
        r = rows[n,v]
        return r['pipeline_ms']/1000 if kind=='pipeline' else r['metrics']['decode_ms_per_forward']
    def ratio(n, a, b, kind='decode'):
        return metric(n,a,kind)/metric(n,b,kind)
    def reduction(n,a,b,kind='decode'):
        return 100*(1-1/ratio(n,a,b,kind))

    font_path = '/Library/Fonts/Arial Unicode.ttf'
    pdfmetrics.registerFont(TTFont('Unicode',font_path))
    pdfmetrics.registerFontFamily('Unicode',normal='Unicode',bold='Unicode',italic='Unicode',boldItalic='Unicode')
    navy=colors.HexColor('#14324A'); teal=colors.HexColor('#087F8C'); gray=colors.HexColor('#536674')
    light=colors.HexColor('#EEF4F7')
    styles={
      'body':ParagraphStyle('body',fontName='Unicode',fontSize=9.5,leading=15,wordWrap='CJK',textColor=navy,spaceAfter=8),
      'small':ParagraphStyle('small',fontName='Unicode',fontSize=8,leading=12,wordWrap='CJK',textColor=gray,spaceAfter=6),
      'h1':ParagraphStyle('h1',fontName='Unicode',fontSize=22,leading=29,wordWrap='CJK',textColor=navy,spaceAfter=12),
      'h2':ParagraphStyle('h2',fontName='Unicode',fontSize=13,leading=19,wordWrap='CJK',textColor=teal,spaceBefore=10,spaceAfter=7),
      'cell':ParagraphStyle('cell',fontName='Unicode',fontSize=8.4,leading=12,wordWrap='CJK',textColor=navy),
      'head':ParagraphStyle('head',fontName='Unicode',fontSize=8.3,leading=12,wordWrap='CJK',textColor=colors.white),
    }
    def P(s,style='body'):
        return Paragraph(s,styles[style])
    story=[]
    def add(s,style='body'):story.append(P(s,style))
    def table(data,widths,small=False):
        cells=[[P(str(c),'head' if i==0 else 'cell') for c in row] for i,row in enumerate(data)]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5 if small else 7),('BOTTOMPADDING',(0,0),(-1,-1),5 if small else 7),('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor('#D4E0E6'))]))
        story.extend([t,Spacer(1,9)])
    def page(k,title,subtitle):
        if story:story.append(PageBreak())
        add(f'ROCKETKV   /   {k:02d}', 'small');add(title,'h1');add(subtitle,'small')
    palette={'cpu_hsa':'#AF5961','metal_initial':'#BA843D','metal_graph':'#517CAC','metal_final':'#087F8C','native':'#353C46','stage1':'#8B79A9'}
    def chart(kind, variants, log=False):
        w,h=510,226
        g=Drawing(w,h);left,right,bottom,top=48,492,49,195
        vals=[metric(n,v,kind) for n in lengths for v in variants]
        if log:
            low,high=math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))
            ticks=[10**i for i in range(low,high+1)]
            yy=lambda x:bottom+(math.log10(x)-low)/(high-low)*(top-bottom)
        else:
            low=0;high=math.ceil(max(vals)/10)*10
            ticks=list(range(0,int(high)+1,10))
            yy=lambda x:bottom+x/high*(top-bottom)
        for t in ticks:
            y=yy(t);g.add(Line(left,y,right,y,strokeColor=colors.HexColor('#DDE5EA'),strokeWidth=.5))
            g.add(String(left-7,y-3,str(t),fontName='Unicode',fontSize=8,textAnchor='end',fillColor=gray))
        g.add(String(left,top+17,'Pipeline seconds' if kind=='pipeline' else 'Decode ms / forward  (log scale)',fontName='Unicode',fontSize=9,fillColor=gray))
        xs=[left+i*(right-left)/3 for i in range(4)]
        for x,n in zip(xs,lengths):g.add(String(x,bottom-17,f'{n//1024}K',fontName='Unicode',fontSize=9,textAnchor='middle',fillColor=navy))
        for idx,v in enumerate(variants):
            color=colors.HexColor(palette[v]);ys=[yy(metric(n,v,kind)) for n in lengths]
            for i in range(3):g.add(Line(xs[i],ys[i],xs[i+1],ys[i+1],strokeColor=color,strokeWidth=1.7))
            for x,y in zip(xs,ys):g.add(Circle(x,y,2.5,fillColor=color,strokeColor=color))
            x=left+(idx%3)*155;y=16-(idx//3)*14
            g.add(Line(x,y+3,x+13,y+3,strokeColor=color,strokeWidth=2));g.add(String(x+18,y,names[v],fontName='Unicode',fontSize=8,fillColor=navy))
        return g

    page(1,'HSA 从 CPU 到 Metal','2K / 4K / 8K / 16K  ·  pipeline 性能与算法演进  ·  '+d['date'])
    speeds=[ratio(n,'cpu_hsa','metal_final','pipeline') for n in lengths]
    dec=[ratio(n,'cpu_hsa','metal_final') for n in lengths]
    add(f'在统一的 GPU 模型配置下，CPU HSA 到当前 Metal HSA 的整段推理加速为 {min(speeds):.2f}–{max(speeds):.2f} 倍；decode 加速为 {min(dec):.2f}–{max(dec):.2f} 倍。当前 HSA 仍慢于原生 Metal 全量 attention，下面同时给出该参照。')
    add('整段 pipeline：prefill + Stage 1 + HSA 初始化 + 32 步 decode','h2')
    header=['输入','CPU HSA\n秒','初版 Metal\n秒','KV / graph\n秒','当前 Metal\n秒','CPU→当前\n加速']
    table([header]+[[f'{n//1024}K']+[f'{metric(n,v,"pipeline"):.3f}' for v in phase]+[f'{ratio(n,"cpu_hsa","metal_final","pipeline"):.2f}×'] for n in lengths],[39,92,94,94,94,98])
    story.append(chart('pipeline',phase))
    add('读图说明','h2')
    add('每个输入长度、每个版本只补测一次，所有版本使用同一段文本、同一组固定 token 和 32 次 decode forward。每步耗时是该单次运行内的平均值，不是多轮中位数；小幅差异不应作统计显著性结论。')
    add('pipeline 不含模型加载、进程启动、分词和末尾清理。固定 token 绕过采样，所以这是受控的请求计算耗时，不是用户实际等待的完整墙钟时间。原始数据见 [S1]。','small')

    page(2,'Decode：速度提升发生在哪里','单位：ms / forward，越低越好。模型主体始终在 GPU，CPU HSA 仅指 HSA 在 CPU 执行。')
    table([['输入']+[names[v] for v in order]]+[[f'{n//1024}K']+[f'{metric(n,v):.2f}' for v in order] for n in lengths],[35,80,79,79,79,79,80])
    story.append(chart('decode',order,True))
    add('分阶段 decode 加速倍数','h2')
    table([['输入','CPU → 初版 Metal','初版 → KV / graph','KV / graph → 当前']]+[[f'{n//1024}K',f'{ratio(n,"cpu_hsa","metal_initial"):.2f}×',f'{ratio(n,"metal_initial","metal_graph"):.2f}×',f'{ratio(n,"metal_graph","metal_final"):.2f}×'] for n in lengths],[43,156,156,156])
    add('CPU → GPU 的大幅收益来自把近似评分、选择和精确稀疏 attention 移到 GPU；之后的收益主要来自减少数据搬运、同步和排序开销。以上计时覆盖整个模型的 decode forward，不能解释成单个 HSA kernel 的耗时。')
    add('原生 baseline 与仅 Stage 1 都使用原生 attention；HSA 还需要近似评分、Top-K 和自定义精确 attention，因此压缩 KV 或减少被选 token 并不保证 decode 更快。','small')

    page(3,'整个 pipeline 的耗时组成','各组件是同一次运行的实测值，单位：秒。Decode 包含 32 次 forward 与循环管理。')
    data=[['输入 / 版本','Prefill','Stage 1','HSA init','Decode','合计']]
    for n in lengths:
        for v in phase:
            r=rows[n,v];m=r['metrics']
            data.append([f'{n//1024}K / {names[v]}',f'{m["prefill_ms"]/1000:.3f}',f'{m["evict_ms"]/1000:.3f}',f'{m["stage2_init_ms"]/1000:.4f}',f'{m["ms_per_token"]*m["decode_tokens"]/1000:.3f}',f'{r["pipeline_ms"]/1000:.3f}'])
    table(data,[151,72,72,72,72,72],True)
    add('整段 pipeline 的分阶段耗时下降','h2')
    table([['输入','CPU → 初版 Metal','初版 → KV / graph','KV / graph → 当前']]+[[f'{n//1024}K']+[f'{reduction(n,a,b,"pipeline"):.1f}%' for a,b in [('cpu_hsa','metal_initial'),('metal_initial','metal_graph'),('metal_graph','metal_final')]] for n in lengths],[43,156,156,156],True)
    share=rows[16384,'metal_final']['metrics']['prefill_ms']/rows[16384,'metal_final']['pipeline_ms']*100
    add(f'当前 16K 测试中，prefill 占整段时间约 {share:.1f}%。因此即使 decode 明显加速，短输出请求的整体收益也会被 prefill 稀释；生成更多 token 时，decode 的权重才会更大。不同版本的 prefill 小差异来自运行波动，不能归因于 HSA kernel 优化。')

    baseline_total='；'.join(f'{n//1024}K：{metric(n,"native","pipeline"):.3f} 秒' for n in lengths)
    add('原生 baseline 的整段 pipeline 耗时为 '+baseline_total+'。它不执行 Stage 1 或 HSA，因此需要同时考虑 prefill 与额外压缩开销。','small')

    page(4,'每一步的算法与实现变化','按提交演进解释：CPU reference → 初版 Metal → KV / graph 优化 → 当前 Metal。')
    add('1  将 HSA 搬到 GPU  ·  8b9b27a [S2]','h2')
    add('HSA 先挑选 query 的重要维度，用 chunk 内 K 的 min/max 代表值估计注意力分数，再选 Top-K token，最后只对选中的 KV 做精确 QK、softmax 和加权求和。初版把这些步骤实现为 Metal kernels，利用 GPU 并行；但每层仍经 CPU callback 搬入数据、打包 KV，并提交和等待独立 command buffer。CPU reference 的算法代码未因此被替换。')
    add('2  减少重复工作与逐层同步  ·  a708c7c [S3]','h2')
    add('直接读取原始 F16 KV，避免每步完整转换/复制为 F32 工作数组；逐层缓存 chunk 的 min/max，追加 token 时只更新尾部。可容纳的 Top-K 排序改用 threadgroup 内存。兼容的共享缓冲区路径把 HSA 编码进 GGML 现有 Metal encoder，直接读 Q/K/V、写输出，把 CPU 错误检查留到 decode 同步后，减少逐层提交与等待。该阶段是组合优化，本次未逐项拆开重测。')
    add('3  缩小排序并提高 kernel 执行效率  ·  dbf4743 [S4]','h2')
    add('同一 chunk 的 token 近似分数相同，因此先排序 chunk，再展开 token ID 并按时间顺序排列，减少主排序的规模。加权求和改用 64 线程组；每次提前加载四个 V，但仍按原顺序累加。排序中 SIMD 组内交换使用 shuffle，减少共享内存访问和线程组屏障。')
    add('当前阶段的逐项收益：此前三轮交替 A/B [S5]','h2')
    table([['保留改动','8K decode 耗时下降','16K decode 耗时下降'],['chunk 排序并展开 token','17.77%','46.73%'],['加权求和：64 线程组','1.83%','2.17%'],['提前加载四个 V，保持累加顺序','17.93%','17.11%'],['SIMD shuffle 排序','1.20%','1.69%']],[251,130,130],True)
    add('上述百分比是各自相对上一保留版的增量，不能直接相加，也不是整段 pipeline 的下降比例。此前 8K / 16K 最终累计 decode 耗时下降分别为 34.29% / 57.44%；2K / 4K 没有逐项历史实验，本文只补测阶段组合。改变浮点归约顺序而导致模型数值漂移的方案已撤回。','small')

    page(5,'实验口径、验证与复现','单轮补测用于补齐四档输入；历史三轮结果用于解释已经验证过的细分优化。')
    table([['项目','设置'],['硬件 / 模型','Apple M2 Ultra；Llama-3.2-3B-Instruct-Q8_0'],['计算配置','模型全 GPU offload（99）；KV F16；16 CPU threads；budget=1024'],['输入','sample 201 文本重复 8 次；中间截断到 2048 / 4096 / 8192 / 16384 token；n_ctx=输入长度+1024'],['HSA 实际规模','四档 top-k 均为 512；chunk 大小依次为 1 / 2 / 2 / 3。输入经 Stage 1 缩减后，HSA 候选数小于原始输入长度。'],['输出','固定 token IDs 1000..1032；max_tokens=33；首 token 来自 prefill，后续 32 次 decode forward；trace 关闭'],['版本','CPU HSA 与初版 Metal：8b9b27a；KV / graph：a708c7c；当前 Metal：dbf4743'],['构建','历史项目源码均链接同一套当前 patched llama.cpp 库；仅临时 runner 解除 forced_tokens 对 trace 的要求'],['统计','每版本每长度 1 次，共 24 个有效测试；不外推为 LongBench 质量结果']],[93,418],True)
    add('验证范围','h2')
    add('本次检查每次运行均为准确的输入长度、32 次 decode forward，HSA 执行 28 层 × 32 步并使用指定后端。当前版本此前通过完整回归、UBSan、Metal shader validation；8K / 16K / 32K 固定 4 步模型比较中，attention 和 logits 与优化前 Metal 完全一致 [S5]。这不表示 CPU 与 GPU 全模型逐位一致，也不替代生成质量评测。')
    add('数据与源码索引','h2')
    for s in [
      '[S1] docs/hsa_pipeline_benchmark.json：本次 24 次运行的完整指标、配置、输入哈希；本 PDF 所有新表格由该文件计算。',
      '[S2] git 8b9b27a：Implement Metal HSA backend and organize project layout。',
      '[S3] git a708c7c：Optimize Stage 2 Metal KV access and graph execution。',
      '[S4] git dbf4743：Optimize Metal HSA selection and weighted attention。',
      '[S5] HSA_OPTIMIZATION_LOG.md：逐项 A/B、撤回原因、正确性记录与源代码差异。',
    ]:add(escape(s),'small')
    add('复现','h2')
    add('在仓库根目录执行：<br/>.venv/bin/python docs/hsa_pipeline_report.py bench<br/>.venv/bin/python docs/hsa_pipeline_report.py pdf','small')
    add('PDF 生成依赖 reportlab==4.4.3 和本机 Arial Unicode 字体。bench 需要已有模型、samples.json 和编译好的 llama.cpp；已有单轮记录会复用，要全新重测可先移走 JSON。正式推理源码不变，临时构建会自动清理。','small')

    out=ROOT/'docs/hsa_pipeline_report.pdf'
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#D4E0E6'));canvas.line(42,39,553,39)
        canvas.setFont('Unicode',8);canvas.setFillColor(gray)
        canvas.drawString(42,26,'ROCKETKV  /  HSA PERFORMANCE  /  '+d['date'])
        canvas.drawRightString(553,26,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=37,bottomMargin=51,title='HSA 从 CPU 到 Metal：Pipeline 性能与算法演进',author='RocketKV project',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)

if __name__ == '__main__' and sys.argv[1:] == ['pdf']:
    render_pdf()
