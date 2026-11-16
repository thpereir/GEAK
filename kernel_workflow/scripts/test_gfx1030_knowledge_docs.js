#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// Doc contract for gfx1030 (RDNA2). The roles route gfx103* to knowledge/amd_rdna2.md; that routing
// is only worth anything if the file still states the facts it is read for, if its numbers agree
// with the machine-readable references, and if its pointer to the long form resolves.
// Text only: no GPU, no npm install.

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const repoRoot = path.resolve(wfDir, '..');

function mustInclude(hay, needle, msg) {
  if (!hay.includes(needle)) throw new Error(msg || `missing ${needle}`);
}

const rdna2Raw = fs.readFileSync(path.join(wfDir, 'knowledge', 'amd_rdna2.md'), 'utf8');
// Facts are checked on the prose, so line wrapping and **emphasis** cannot hide or fake one.
const rdna2 = rdna2Raw.replace(/\*\*/g, '').replace(/\s+/g, ' ');
for (const fact of ['gfx1030', 'no matrix core', 'Radeon PRO V620', '(72 CU)', 'v_dot2_f32_f16',
  'v_dot2_f32_bf16', 'TORCH_BLAS_PREFER_HIPBLASLT']) {
  mustInclude(rdna2, fact, `amd_rdna2.md no longer states: ${fact}`);
}

// The occupancy steps in the guide must be the ones hw_constants.json (and so amd_occupancy.py) uses.
const hw = JSON.parse(fs.readFileSync(path.join(repoRoot, 'perf_knowledge', 'expert_skills', 'skills',
  'gluon_authoring', 'references', 'hardware', 'hw_constants.json'), 'utf8')).arch.gfx1030;
if (!hw) throw new Error('hw_constants.json has no gfx1030 entry');
for (const [vgpr, waves] of hw.vgpr_wave_steps) {
  if (!new RegExp(`≤${vgpr} (VGPRs )?${waves}\\b`).test(rdna2)) {
    throw new Error(`amd_rdna2.md does not state the step ≤${vgpr} VGPRs -> ${waves} waves from hw_constants.json`);
  }
}
mustInclude(rdna2, `${hw.vgpr_file_per_simd} VGPR/SIMD`, 'amd_rdna2.md VGPR file size disagrees with hw_constants.json');
mustInclude(rdna2, `granule ${hw.vgpr_alloc_granule}`, 'amd_rdna2.md granule disagrees with hw_constants.json');

// The roofline row is CU-gated; the guide must say which CU count it holds for.
const peaks = fs.readFileSync(path.join(repoRoot, 'e2e_workflow', 'knowledge', 'analysis_skills', 'roofline',
  'peaks.md'), 'utf8');
const row = peaks.split('```yaml').find((b) => /^\s*gfx:\s*gfx1030\s*$/m.test(b));
if (!row) throw new Error('peaks.md has no gfx1030 yaml row');
const cu = (row.match(/^\s*cu:\s*(\d+)/m) || [])[1];
mustInclude(rdna2, `only at ${cu} CU`, `amd_rdna2.md does not say the peaks hold only at ${cu} CU`);

// Every role that routes gfx11* to amd_ryzen.md must also route gfx103* to amd_rdna2.md, or a gfx1030
// run falls through to the CDNA reference.
for (const role of fs.readdirSync(path.join(wfDir, 'roles')).filter((f) => f.endsWith('.md'))) {
  const text = fs.readFileSync(path.join(wfDir, 'roles', role), 'utf8');
  if (text.includes('amd_ryzen.md') && !text.includes('amd_rdna2.md')) {
    throw new Error(`roles/${role} routes gfx11* to amd_ryzen.md but has no gfx103* route to amd_rdna2.md`);
  }
}

// The long form is a separate directory so amd_rdna2.md can stay a summary.
const longFormRel = 'perf_knowledge/hardware/rdna2_gfx1030';
mustInclude(rdna2Raw, `$SKILL_DIR/../${longFormRel}`, 'amd_rdna2.md lost the SKILL_DIR pointer to the long form');
const longFormDir = path.join(repoRoot, longFormRel);
const pages = fs.readdirSync(longFormDir).filter((f) => f.endsWith('.md'));
if (!pages.includes('arch.md')) throw new Error(`${longFormRel} has no arch.md entry point`);
const reproDir = path.join(longFormDir, 'repro');
const docs = pages.map((f) => path.join(longFormDir, f)).concat([path.join(reproDir, 'README.md')]);

// Every relative link and every relative `sources:` entry must resolve.
for (const doc of docs) {
  const rel = path.relative(repoRoot, doc);
  const text = fs.readFileSync(doc, 'utf8');
  const targets = [...text.matchAll(/\]\((?!\w+:)([^)#\s]+)(?:#[^)]*)?\)/g)].map((m) => m[1]);
  const front = text.startsWith('---\n') ? text.slice(4, text.indexOf('\n---', 4)) : '';
  for (const m of front.matchAll(/^\s+- (\S+)\s*$/gm)) {
    if (!/^\w+:\/\//.test(m[1])) targets.push(m[1]);
  }
  for (const target of targets) {
    if (!fs.existsSync(path.resolve(path.dirname(doc), target))) {
      throw new Error(`${rel} references a missing file: ${target}`);
    }
  }
}

// arch.md links each canonical source rather than restating it.
const archLinks = [...fs.readFileSync(path.join(longFormDir, 'arch.md'), 'utf8').matchAll(/\]\((?!\w+:)([^)#\s]+)\)/g)]
  .map((m) => path.relative(repoRoot, path.resolve(longFormDir, m[1])));
for (const canonical of [
  'kernel_workflow/knowledge/amd_rdna2.md',
  'perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json',
  'e2e_workflow/knowledge/analysis_skills/roofline/peaks.md',
]) {
  if (!archLinks.includes(canonical)) throw new Error(`arch.md does not link the canonical ${canonical}`);
}

// Every [measured] item names where it comes from under repro/, and every script there is described.
const reproReadme = fs.readFileSync(path.join(reproDir, 'README.md'), 'utf8');
mustInclude(reproReadme, '## Claims with no script here', 'repro/README.md lost its list of claims with no script');
for (const page of pages) {
  const items = fs.readFileSync(path.join(longFormDir, page), 'utf8')
    .split(/\n\s*\n|\n(?=\s{0,3}(?:[-*]|\d+\.)\s)/);
  for (const item of items) {
    if (item.includes('[measured]') && !/\brepro\//.test(item)) {
      throw new Error(`${page}: a [measured] item cites nothing under repro/: ${item.trim().slice(0, 80)}`);
    }
  }
}
for (const script of fs.readdirSync(reproDir, { withFileTypes: true })
  .filter((e) => e.isFile() && e.name !== 'README.md' && !e.name.startsWith('.')).map((e) => e.name)) {
  mustInclude(reproReadme, `\`${script}\``, `repro/README.md does not describe ${script}`);
}
for (const doc of docs) {
  for (const m of fs.readFileSync(doc, 'utf8').matchAll(/repro\/([\w.]+\.(?:py|sh))/g)) {
    if (!fs.existsSync(path.join(reproDir, m[1]))) {
      throw new Error(`${path.relative(repoRoot, doc)} cites repro/${m[1]}, which does not exist`);
    }
  }
}

mustInclude(
  fs.readFileSync(path.join(repoRoot, 'perf_knowledge', 'index', 'taxonomy.md'), 'utf8'),
  '| `gfx1030` |', 'taxonomy.md does not list gfx1030 as a hardware generation');

console.log('PASS: amd_rdna2.md states the gfx1030 facts and agrees with hw_constants.json and peaks.md; every role ' +
  'routes gfx103*; the long form resolves, links its canonical sources, and every cited script exists.');
