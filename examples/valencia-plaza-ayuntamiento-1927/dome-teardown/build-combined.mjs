import { build } from 'esbuild';
import fs from 'fs';
await build({ entryPoints: ['src/main.js'], bundle: true, minify: true, format: 'iife', outfile: 'dist/bundle.js', target: 'es2020', legalComments: 'none' });
const js = fs.readFileSync('dist/bundle.js', 'utf8').replace(/<\/script/gi, '<\\/script');
const vid = fs.readFileSync('/tmp/intro_full.mp4').toString('base64');
let html = fs.readFileSync('template.html', 'utf8').replace('/*BUNDLE*/', () => js).replace('/*VIDEO*/', () => vid);
fs.writeFileSync('dist/combined.html', html);
console.log('combined', (html.length / 1048576).toFixed(2), 'MiB');
