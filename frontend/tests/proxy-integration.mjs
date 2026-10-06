import assert from 'node:assert/strict';
import { createServer, request as httpRequest } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { setTimeout as sleep } from 'node:timers/promises';
const artifacts = process.env.ARTIFACT_DIR || '/tmp/invoice-proxy-artifacts';
await mkdir(artifacts, {recursive:true});
const report = {checks:[], started:new Date().toISOString()};
const password = 'integration-only-password-32-characters';
const authorization = `Basic ${Buffer.from(`demo:${password}`).toString('base64')}`;
const headers = {authorization};
const result = {invoice_number:'INV-1042',extraction_method:'heuristic',total_words:1,predictions:[{word:'INV-1042',label:'HEURISTIC_MATCH',is_invoice_number:true}]};
let mode='success', submitted=0, cancelled=0, lastInput, polls=0;
let workers={idle:1,running:0,initializing:0,ready:1};
const upstream=createServer(async(req,res)=>{
  const chunks=[];for await(const c of req) chunks.push(c);
  const send=(status,data)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(data));};
  if(req.url==='/predict') {
    if(mode==='local-timeout') { await sleep(2000); return send(200,result); }
    if(mode==='local-invalid-image') return send(400,{detail:'Invalid image file: cannot identify image file'});
    if(mode==='local-validation') return send(422,{detail:[{msg:'Field required'}]});
    return send(500,{detail:'Internal server error: private configuration'});
  }
  if(req.url.endsWith('/health')) return send(200,{workers});
  if(req.url.endsWith('/run')) {
    submitted++; const payload=JSON.parse(Buffer.concat(chunks)); lastInput=payload.input; assert(payload.policy.executionTimeout>=5000); assert(payload.policy.ttl>=10000); polls=0;
    assert.equal(req.headers.authorization,'Bearer integration-only-key');
    if(mode==='submit-failure') return send(503,{error:'failed'});
    if(mode==='bad-submit') return send(200,{status:'IN_QUEUE'});
    return send(200,{id:'job-1',status:'IN_QUEUE'});
  }
  if(req.url.includes('/cancel/')) {cancelled++;return send(mode==='cancel-failure'?500:200,{status:'CANCELLED'});}
  if(req.url.includes('/status/')) {
    polls++;
    if(mode==='poll-failure') return send(503,{error:'failed'});
    if(mode==='bad-json'){res.writeHead(200);return res.end('not json');}
    if(['timeout','cancel-failure','abort'].includes(mode)) return send(200,{status:'IN_PROGRESS'});
    if(['FAILED','CANCELLED','TIMED_OUT'].includes(mode)) return send(200,{status:mode});
    if(mode==='bad-output') return send(200,{status:'COMPLETED',output:{invoice_number:'oops'}});
    if(polls===1) return send(200,{status:'IN_PROGRESS'});
    return send(200,{status:'COMPLETED',output:result});
  }
  send(404,{});
});
await new Promise(resolve=>upstream.listen(0,'127.0.0.1',resolve));
const upstreamPort=upstream.address().port;
const reserve=createServer();await new Promise(resolve=>reserve.listen(0,'127.0.0.1',resolve));const port=reserve.address().port;await new Promise(resolve=>reserve.close(resolve));
const base=`http://127.0.0.1:${port}`;
let logs='';
const app=spawn(process.execPath,['node_modules/next/dist/bin/next','start','--port',String(port)],{env:{...process.env,DEMO_PASSWORD:password,RUNPOD_ENDPOINT_ID:'test',RUNPOD_API_KEY:'integration-only-key',RUNPOD_INVOKE_BASE_URL:`http://127.0.0.1:${upstreamPort}`,INVOICE_PROCESSING_TIMEOUT_MS:'3500'},stdio:['ignore','pipe','pipe']});
app.stdout.on('data',c=>logs+=c);app.stderr.on('data',c=>logs+=c);
const image=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j1xkAAAAASUVORK5CYII=','base64');
function form(text=JSON.stringify({words:['INV-1042'],bboxes:[[0,0,10,10]]}),name='receipt.json') {const f=new FormData();f.set('image',new Blob([image],{type:'image/png'}),'receipt.png');f.set('ocr_file',new Blob([text]),name);return f;}
const post=(body=form(),extra={})=>fetch(`${base}/api/predict`,{method:'POST',headers,body,...extra});
async function check(name,fn){await fn();report.checks.push(name);}
try {
 for(let i=0;i<100;i++){try{await fetch(base);break;}catch{await sleep(100);}}
 await check('Page and API reject unauthenticated requests before paid work',async()=>{assert.equal((await fetch(base)).status,401);assert.equal((await post(form(),{headers:{}})).status,401);assert.equal(submitted,0);});
 await check('Incorrect password and cross-origin requests rejected',async()=>{assert.equal((await post(form(),{headers:{authorization:'Basic ZGVtbzpiYWQ='}})).status,401);assert.equal((await post(form(),{headers:{...headers,origin:'https://other.example'}})).status,403);assert.equal(submitted,0);});
 const invalid=[['bad JSON','{'],['unequal arrays',JSON.stringify({words:['x'],bboxes:[]})],['invalid box',JSON.stringify({words:['x'],bboxes:[[9,0,1,5]]})],['non-numeric box',JSON.stringify({words:['x'],bboxes:[[0,0,'10',10]]})],['empty word',JSON.stringify({words:[''],bboxes:[[0,0,10,10]]})],['invalid lines',JSON.stringify({words:['x'],bboxes:[[0,0,10,10]],ocr_lines:[3]})],['bad TXT','bad line','receipt.txt'],['invalid numeric TXT','0,0,10,0,10,10,0,10,Hello\nno,0,10,0,10,10,0,10,broken','receipt.txt'],['empty-text TXT','0,0,10,0,10,10,0,10,','receipt.txt'],['invalid UTF-8',Buffer.from([0xff])]];
 for(const [name,text,file] of invalid) await check(`Reject ${name} before submission`,async()=>{const n=submitted;assert.equal((await post(form(text,file))).status,400);assert.equal(submitted,n);});
 await check('Streamed request size is bounded before parsing',async()=>{const n=submitted;const response=await new Promise((resolve,reject)=>{const req=httpRequest(`${base}/api/predict`,{method:'POST',headers:{...headers,'content-type':'multipart/form-data; boundary=test','transfer-encoding':'chunked'}},res=>{res.resume();res.on('end',()=>resolve(res.statusCode));});req.on('error',reject);req.end(Buffer.alloc(4_100_000));});assert.equal(response,413);assert.equal(submitted,n);});
 await check('Reject malformed multipart',async()=>{assert.equal((await post('not multipart')).status,400);});
 await check('Reject oversized OCR',async()=>{assert.equal((await post(form('x'.repeat(2_000_001)))).status,400);});
 for(const [name,text,file] of [['JSON',JSON.stringify({words:['INV-1042'],bboxes:[[0,0,10,10]]})],['fractional JSON',JSON.stringify({words:['INV-1042'],bboxes:[[0.5,1.5,10.25,10.75]]})],['TXT with short lines','Header\n0,0,10,0,10,10,0,10,INV-1042\nfooter','receipt.txt'],['TXT with empty text','0,0,10,0,10,10,0,10,\n0,0,10,0,10,10,0,10,INV-1042','receipt.txt'],['boxes alias',JSON.stringify({words:['INV-1042'],boxes:[[0,0,10,10]]})],['TXT','0,0,10,0,10,10,0,10,INV-1042','receipt.txt']]) await check(`Full submission and polling with ${name}`,async()=>{mode='success';const response=await post(form(text,file));assert.equal(response.status,200);assert.deepEqual(await response.json(),result);assert.equal(Buffer.from(lastInput.ocr_base64,'base64').toString(),text);assert.equal(lastInput.image_base64,image.toString('base64'));});
 for(const scenario of ['submit-failure','bad-submit','FAILED','CANCELLED','TIMED_OUT','bad-output']) await check(`Handles ${scenario}`,async()=>{mode=scenario;assert.equal((await post()).status,502);});
 for(const scenario of ['poll-failure','bad-json','timeout']) await check(`Cancels outstanding job on ${scenario}`,async()=>{mode=scenario;const before=cancelled;const response=await post();assert.equal(response.status,scenario==='timeout'?504:502);assert.equal(cancelled,before+1);});
 await check('Cancellation failure warns against duplicate retry',async()=>{mode='cancel-failure';const response=await post();assert.equal(response.status,504);assert.match((await response.json()).detail,/could not confirm cancellation/i);});
 await check('Disconnect cancels known job',async()=>{mode='abort';const before=cancelled;const n=submitted;const controller=new AbortController();const pending=post(form(),{signal:controller.signal}).catch(()=>null);while(submitted===n)await sleep(20);await sleep(100);controller.abort();await pending;for(let i=0;i<60 && cancelled===before;i++)await sleep(100);assert.equal(cancelled,before+1);});
 for(const [state,counts] of [['ready',{idle:1,ready:1}],['busy',{running:1,ready:1}],['initializing',{initializing:1}],['idle',{}]]) await check(`Health reports ${state}`,async()=>{workers=counts;const r=await fetch(`${base}/api/health`,{headers});const data=await r.json();assert.equal(data.status,state);assert.equal(data.ready,state==='ready');});
 await check('Production fails closed without password',async()=>{
   const reserve=createServer();await new Promise(r=>reserve.listen(0,'127.0.0.1',r));const p=reserve.address().port;await new Promise(r=>reserve.close(r));
   const unconfigured=spawn(process.execPath,['node_modules/next/dist/bin/next','start','--port',String(p)],{env:{...process.env,DEMO_PASSWORD:''},stdio:'ignore'});
   try {let response;for(let i=0;i<100;i++){try{response=await fetch(`http://127.0.0.1:${p}`);break;}catch{await sleep(100);}}assert.equal(response.status,503);}finally{unconfigured.kill('SIGTERM');}
 });
 const localReserve=createServer();await new Promise(r=>localReserve.listen(0,'127.0.0.1',r));const localPort=localReserve.address().port;await new Promise(r=>localReserve.close(r));
 const localApp=spawn(process.execPath,['node_modules/next/dist/bin/next','start','--port',String(localPort)],{env:{...process.env,DEMO_PASSWORD:password,RUNPOD_ENDPOINT_ID:'',RUNPOD_API_KEY:'',INVOICE_NER_API_URL:`http://127.0.0.1:${upstreamPort}`,INVOICE_PROCESSING_TIMEOUT_MS:'1000'},stdio:'ignore'});
 try {
  for(let i=0;i<100;i++){try{await fetch(`http://127.0.0.1:${localPort}`);break;}catch{await sleep(100);}}
  for(const [scenario,status] of [['local-invalid-image',400],['local-validation',422],['local-internal-error',500]]) {
   await check(`Local proxy handles ${scenario}`,async()=>{mode=scenario;const r=await fetch(`http://127.0.0.1:${localPort}/api/predict`,{method:'POST',headers,body:form()});assert.equal(r.status,status);const data=await r.json();if(status===400)assert.equal(data.detail,'Invalid image file: cannot identify image file');else{assert.equal(typeof data.detail,'string');assert(data.detail.length>0);assert(!data.detail.includes('private configuration'));}});
  }
  await check('Local proxy honors the configured processing deadline',async()=>{mode='local-timeout';const started=Date.now();const response=await fetch(`http://127.0.0.1:${localPort}/api/predict`,{method:'POST',headers,body:form()});assert.equal(response.status,504);assert(Date.now()-started<1800);});
 } finally {localApp.kill('SIGTERM');}
 report.status='passed';
} catch(error){report.status='failed';report.error=String(error.stack);process.exitCode=1;}
finally {app.kill('SIGTERM');upstream.closeAllConnections();await new Promise(resolve=>upstream.close(resolve));await writeFile(`${artifacts}/report.json`,JSON.stringify(report,null,2));await writeFile(`${artifacts}/server.log`,logs);console.log(JSON.stringify(report,null,2));}
