const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto('http://localhost:8000');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_csv.png', fullPage: true });

  await page.click('a[href="#linear"]');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_linear.png', fullPage: true });

  await page.click('a[href="#metrics"]');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_metrics.png', fullPage: true });

  await page.click('a[href="#splines"]');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshot_splines.png', fullPage: true });

  await browser.close();
})();
