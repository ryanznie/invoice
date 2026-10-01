import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const dir = process.env.ARTIFACT_DIR || '/tmp/receipt-review-artifacts';
await mkdir(dir, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH ?? (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : undefined), headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
const report = { started: new Date().toISOString(), checks: [], errors: [] };
page.on('pageerror', error => report.errors.push(error.message));
const check = async (name, action) => { await action(); report.checks.push(name); };
const visible = async locator => { await locator.waitFor({ state: 'visible' }); };
try {
  // A synthetic receipt is the fixture, never a user's document.
  await page.setContent(`<body style="margin:0;background:white;width:380px;font:18px monospace;color:#333"><div style="padding:40px"><h2 style="text-align:center">NORTHSIDE COFFEE</h2><p style="text-align:center">21 Market Street<br>New York, NY</p><hr><p>Receipt: INV-1042</p><p>30 Sep 2026 · 09:41</p><hr><p>Flat white     $4.50</p><p>Croissant      $3.75</p><br><hr><p>TOTAL          $8.25</p><br><p style="text-align:center">Thank you. See you soon!</p></div></body>`);
  const receipt = await page.locator('body').screenshot();
  const image = { name: 'receipt.png', mimeType: 'image/png', buffer: receipt };
  const ocr = { name: 'receipt.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({words:['INV-1042'],bboxes:[[10,10,100,100]]})) };
  let status = 200, raw = null;
  let data = { invoice_number: 'INV-1042', extraction_method: 'heuristic', total_words: 7, predictions: [{ word: 'Receipt:', label:'LABEL_0', is_invoice_number:false },{word:'INV-1042',label:'HEURISTIC_MATCH',is_invoice_number:true}] };
  await page.route('**/api/predict', async route => {
    await new Promise(resolve => setTimeout(resolve, 180));
    await route.fulfill({status, contentType:raw ? 'text/plain' : 'application/json',body:raw || JSON.stringify(data)});
  });
  await page.goto(process.env.BASE_URL || 'http://localhost:3100');
  const extract = page.getByRole('button', {name:'Extract invoice number'});
  await check('Empty state; extraction requires both files', async()=>{ await visible(page.getByText('No data extracted')); assert(await extract.isDisabled()); });
  await page.screenshot({path:`${dir}/desktop-empty.png`,fullPage:true});
  await check('Unsupported image gives actionable error', async()=>{await page.getByLabel('Receipt image',{exact:true}).setInputFiles({name:'receipt.pdf',mimeType:'application/pdf',buffer:Buffer.from('pdf')}); await visible(page.locator('.error-message'));});
  await check('Image preview and required OCR file', async()=>{await page.getByLabel('Receipt image',{exact:true}).setInputFiles(image); await visible(page.getByAltText('Your uploaded receipt')); assert(await extract.isDisabled()); await page.locator('#ocr-file').setInputFiles(ocr); assert(await extract.isEnabled()); });
  await check('Loading, successful extraction, and needs review',async()=>{await extract.click();await visible(page.getByText('Extracting invoice number…'));await visible(page.getByText('Needs review',{exact:true}));assert.equal(await page.locator('#invoice-number').inputValue(),'INV-1042');});
  await page.screenshot({path:`${dir}/desktop-result.png`,fullPage:true});
  await check('Zoom can be toggled',async()=>{await page.getByRole('button',{name:'Zoom',exact:true}).click();assert.equal(await page.locator('.receipt-preview.zoomed').count(),1);await page.getByRole('button',{name:'Fit',exact:true}).click();});
  await check('Review is explicit; editing clears review',async()=>{await page.getByRole('button',{name:'Confirm number',exact:true}).click();await visible(page.getByText('Reviewed',{exact:true}));await page.locator('#invoice-number').fill('INV-1043');await visible(page.getByText('Needs review',{exact:true}));await visible(page.getByText('Originally extracted:'));});
  await check('Copy uses corrected number',async()=>{await page.context().grantPermissions(['clipboard-read','clipboard-write']);await page.getByRole('button',{name:'Copy number'}).click();await visible(page.getByText('Copied to clipboard'));assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),'INV-1043');});
  await check('Extraction details contain matched tokens',async()=>{await page.getByText('Extraction details',{exact:false}).click();await visible(page.locator('.word-list .matched'));});
  await check('Mobile result has no horizontal overflow',async()=>{await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:`${dir}/mobile-result.png`,fullPage:true});});
  await check('Changing OCR clears stale results',async()=>{await page.locator('#ocr-file').setInputFiles(ocr);await visible(page.getByText('No data extracted'));});
  data={...data,invoice_number:'Not Found',predictions:[]};
  await check('No match requires manual correction',async()=>{await extract.click();await visible(page.getByText('We couldn’t find an invoice number.'));assert(await page.getByRole('button',{name:'Confirm number',exact:true}).isDisabled());await page.locator('#invoice-number').fill('MANUAL-1');await page.getByRole('button',{name:'Confirm number',exact:true}).click();await visible(page.getByText('Reviewed',{exact:true}));});
  data={...data,invoice_number:'INV 1042',predictions:[{word:'INV',label:'LABEL_1',is_invoice_number:true},{word:'1042',label:'LABEL_2',is_invoice_number:true}]};
  await check('Multiple matches prompt careful review',async()=>{await extract.click();await visible(page.getByText('More than one word was matched.',{exact:false}));});
  status=502;raw='Bad gateway';
  await check('Non-JSON server errors remain retryable',async()=>{await extract.click();await visible(page.locator('.error-message'));assert(await extract.isEnabled());});
  status=200;raw=null;data={invoice_number:'bad'};
  await check('Malformed responses do not show success',async()=>{await extract.click();await visible(page.getByText('We received an incomplete result. Please try again.'));});
  await check('OCR size validation prevents upload',async()=>{await page.locator('#ocr-file').setInputFiles({name:'large.json',mimeType:'application/json',buffer:Buffer.alloc(2_000_001)});await extract.click();await visible(page.getByText('The OCR file must be 2 MB or smaller.'));});
  await check('Reset clears image, files, and result',async()=>{await page.getByRole('button',{name:'Start over'}).click();await visible(page.getByText('Upload document'));assert(await extract.isDisabled());});
  await check('Unreadable image is rejected',async()=>{await page.getByLabel('Receipt image',{exact:true}).setInputFiles({name:'corrupt.png',mimeType:'image/png',buffer:Buffer.from('invalid')});await visible(page.getByText('This image could not be opened.',{exact:false}));});
  await check('Real API rejects missing files without inference',async()=>{const response=await page.request.post(`${process.env.BASE_URL || 'http://localhost:3100'}/api/predict`,{multipart:{}});assert.equal(response.status(),400);});
  await check('Real API validates unsupported uploads',async()=>{const response=await page.request.post(`${process.env.BASE_URL || 'http://localhost:3100'}/api/predict`,{multipart:{image:{name:'receipt.pdf',mimeType:'application/pdf',buffer:Buffer.from('fake')},ocr_file:ocr}});assert.equal(response.status(),400);});
  assert.deepEqual(report.errors,[]);
  report.status='passed';
} catch(error) { report.status='failed';report.failure=String(error.stack);await page.screenshot({path:`${dir}/failure.png`,fullPage:true});process.exitCode=1; }
finally {report.finished=new Date().toISOString();await writeFile(`${dir}/report.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();}
