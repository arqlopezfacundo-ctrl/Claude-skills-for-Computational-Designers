# Plaza de Emilio Castelar, València — 1927 · animación axonométrica en bucle

Animación axonométrica (cámara ortográfica) generada **íntegramente por código**: Blender/Cycles
(`bpy` como módulo de Python) para la geometría, los materiales, la luz y la animación, y un
post-proceso en NumPy/SciPy para el acabado pictórico. Salida: `valencia_plaza_1927_loop.mp4`.

La pieza reconstruye la obra de la **plataforma central de la Plaza de Emilio Castelar** (hoy
Plaza del Ayuntamiento) a partir de la ficha *1927 · Plaza del Ayuntamiento · Javier Goerlich
Lleó · Plano D2* de la *Guía de Arquitectura de València*:

- el "Plano de proyecto de ensanche y rectificación de la Bajada de San Francisco y plaza de
  Emilio Castelar" (Goerlich, 1927), que fija la alineación definitiva de la plaza;
- la isla central de la lámina "Urbanización de la Plaza Emilio Castelar": un andén alargado
  formado por dos extremos circulares de distinto radio unidos por tangentes, con parterres
  ovales a lo largo del eje;
- la *plataforma central ligeramente sobreelevada y ornamental que alojaba en semisótano un
  mercado de flores iluminado por un patio circular* (texto de la ficha);
- el contexto: las Casas Consistoriales (fachada de Mora y Carbonell, 1905) con su torre del
  reloj y torres de esquina, el Palacio de Comunicaciones (Correos) con su torre coronada de
  hierro y la cúpula de vidrio del patio de operaciones, las torres-mirador con cupulino de los
  edificios altos de finales de los años 20, la torre de ladrillo de San Agustín, el tranvía.

## Secuencia (20 s, 24 fps, bucle perfecto)

| tiempo | |
|---|---|
| 0 – 1 s | El plano de Goerlich en tinta sobre papel; la línea de la axonometría se dibuja |
| 1 – 3 s | La aguada de color se extiende desde la isla hacia los bordes |
| 1 – 15 s | Obra en *time-lapse*: excavación a mano y retirada de tierras en carros de mulas, solera, muros de ladrillo que suben hilada a hilada, anillo del patio, pilares, cabrias de madera con cuerdas de cáñamo cuyas cargas oscilan como péndulos amortiguados, andamios de rollizos atados, forjado colocado pieza a pieza, chapado de sillería, escalinatas, balaustradas, pavimento, parterres, floración, árboles, palmeras, farolas, quioscos de flores, fuente; se retira la valla de obra |
| 15 – 17.5 s | La plaza terminada, viva: paseantes en la plataforma, mercado en el patio, tranvías |
| 17.5 – 20 s | El color se retira, queda la línea y vuelve el plano: el último fotograma es el primero |

Todo el tiempo la ciudad funciona: viandantes con ciclo de marcha en las aceras y paseos
arbolados, mirones en la valla, tranvías con trole bajo la catenaria, coches de época, carros,
palomas, banderas (Senyera) y ropa tendida en los terrados, árboles que se mecen, sombras de
nubes que cruzan los tejados.

## Dirección de arte

- **Materiales sin "cuadrasitis"**: todos procedurales y con imperfección — dispersión de color,
  chorreones verticales de lluvia, suciedad a pie de fachada, persianas alicantinas bajadas a
  alturas aleatorias y descoloridas, toldos a rayas, ladrillo con aparejo y mortero irregular,
  sillería con juntas, madera con veta y gris de intemperie, tierra con guijarros, losas de
  terrado con manchas, teja con líquenes.
- **Luz**: sol de hora dorada (27°) con penumbra, cielo frío de relleno para sombras azuladas,
  sombras de nubes animadas; en post, bruma atmosférica por profundidad, haces de luz
  direccionales con polvo en suspensión iluminado, polvo de obra en los frentes de trabajo.
- **Acabado pictórico**: curva tonal fílmica, grading cálido/frío, Kuwahara ligero, oscurecido
  de pigmento en bordes, línea de tinta derivada de discontinuidades de profundidad y normal,
  nubes pintadas al óleo con trama de semitono, papel de acuarela y borde rasgado.

## Estructura

```
render.py          construye la escena y renderiza EXR multicapa + metadatos JSON por frame
post.py            composición pictórica de los pases → PNG
make_video.sh      pipeline completo → MP4 (H.264)
vlc/plan.py        trazado urbano: frentes de la plaza, isla de Goerlich, calles, manzanas
vlc/city.py        parcelario, edificios, Ayuntamiento, Correos, cupulinos, árboles, tranvía
vlc/site.py        la obra: fases, cabrias con péndulo, andamios, forjado, balaustradas…
vlc/agents.py      obreros por oficio, viandantes, mirones, tranvías, coches, carros, palomas
vlc/figures.py     figuras low-poly con bibliotecas de poses (marcha, trabajo, capacho…)
vlc/materials.py   biblioteca de materiales procedurales envejecidos
vlc/stage.py       cámara axonométrica, sol, cielo, ajustes de Cycles
```

## Reproducir

```bash
python3 -m venv venv && venv/bin/pip install bpy numpy scipy shapely mapbox_earcut OpenEXR pillow
./make_video.sh            # render + post + encode (CPU; varias horas en 4 núcleos)
```

## Notas de rigor

Las dimensiones se han restituido de la lámina y de la morfología conocida de la plaza; no son
un levantamiento. Los edificios de contexto se modelan como masas parceladas con su ritmo de
huecos y los hitos (Ayuntamiento, Correos, cupulinos, torre de San Agustín) de forma
simplificada y reconocible, no como réplica ornamental. La plataforma se representa según el
proyecto de 1927 descrito en la ficha (sustituida en los años sesenta).
