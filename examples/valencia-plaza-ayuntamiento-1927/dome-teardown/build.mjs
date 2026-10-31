import { build } from 'esbuild';
import fs from 'fs';
await build({ entryPoints: ['src/main.js'], bundle: true, minify: true, format: 'iife', outfile: 'dist/bundle.js', target: 'es2020', legalComments: 'none' });
const js = fs.readFileSync('dist/bundle.js', 'utf8').replace(/<\/script/gi, '<\\/script');
const html = fs.readFileSync('template.html', 'utf8').replace('/*BUNDLE*/', () => js);
fs.mkdirSync('dist', { recursive: true }); fs.writeFileSync('index.html', html);
console.log('index.html', (html.length / 1024).toFixed(0), 'KB');
