import fs from 'fs';
import { resizeLinear, preprocess } from '../preprocess.js';
const cases = JSON.parse(fs.readFileSync('cases.json'));
for (const c of cases) {
  const rgb = Uint8Array.from(c.rgb);
  const s = resizeLinear(rgb, c.w, c.h);
  let rd = 0; s.forEach((v, i) => rd = Math.max(rd, Math.abs(v - c.small[i])));
  // CLAHE checked on the reference resize so the two stages are tested separately
  const eq = preprocess(rgb, c.w, c.h);
  let ed = 0; eq.forEach((v, i) => ed = Math.max(ed, Math.abs(v - c.eq[i])));
  console.log(`${c.w}x${c.h} resizeMaxDiff=${rd} eqMaxDiff=${ed.toExponential(2)}`);
}
