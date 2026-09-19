import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const { createTextSpiralGuard } = await import(pathToFileURL(path.resolve(root, '../localforge/src/mediated-agent-harness/text-spiral-guard.ts')));
const pasted = fs.readFileSync('/mnt/c/Users/dinga/.codex/attachments/fc4de0f0-afc0-4af9-b3e4-f926abaceb97/pasted-text.txt', 'utf8');
const completion = pasted.slice(pasted.indexOf('{\n"verdict"')).trim();
const results = [];
for (const chunkSize of [1, 4, 16, 64, 256]) {
  const guard = createTextSpiralGuard({maxChars:81920});
  let result = {detected:false,reason:''};
  let offset = 0;
  for (; offset < completion.length; offset += chunkSize) {
    result = guard.feed(completion.slice(offset,offset+chunkSize));
    if (result.detected) break;
  }
  if (!result.detected) result = guard.feed('',true);
  results.push({chunkSize,characters:completion.length,...result});
}
const out = path.join(root,'docs/research/deepseek-loop-investigation-2026-09-17/guard-replay.json');
fs.writeFileSync(out,JSON.stringify({scope:'Replay captured partial output through the existing production guard; no inference',results},null,2)+'\n');
console.log(JSON.stringify(results,null,2));
