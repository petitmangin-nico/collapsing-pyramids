// Render figures/template.html with figures/data.json and save each plate as a PNG in figures/png/.
// Run from the repository root:  node figures/export_figures.js
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const here = __dirname;
const NAMES = ['fig1_pyramid', 'fig2_fold', 'fig3_invention', 'fig4_polarisation', 'fig5_remedy', 'fig6_sales', 'fig8_delay'];

(async () => {
  const template = fs.readFileSync(path.join(here, 'template.html'), 'utf8');
  const data = fs.readFileSync(path.join(here, 'data.json'), 'utf8');
  const page_html = path.join(here, 'figures.html');
  fs.writeFileSync(page_html, template.replace('__DATA__', data));

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 }, deviceScaleFactor: 3, colorScheme: 'light' });
  await page.goto('file://' + page_html);
  await page.waitForTimeout(2000);
  await page.evaluate(() => { document.body.classList.add('export'); draw(); });
  await page.waitForTimeout(400);
  const plates = await page.$$('figure.plate');
  for (let i = 0; i < NAMES.length; i++) {
    await plates[i].screenshot({ path: path.join(here, 'png', NAMES[i] + '.png') });
  }
  await browser.close();
  console.log(`${NAMES.length} figures written to figures/png/`);
})();
