# València · animaciones axonométricas en bucle

Dos piezas generadas íntegramente por código (Blender/Cycles vía `bpy` + post-proceso pictórico
en NumPy/SciPy), con la ciudad viva alrededor de la obra:

| pieza | vídeo | vertical (móvil) |
|---|---|---|
| **Película de 40 s: Correos 1915–1922, cámara continua pájaro ↔ peatón** — versión principal | — | `correos_film_40s_vertical.mp4` |
| **Palacio de Comunicaciones (Correos), 1915–1922** — tema principal | `correos_1915_1922_loop.mp4` | `correos_1915_1922_loop_vertical.mp4` |
| Plataforma central de la Plaza de Emilio Castelar, Goerlich 1927 | `valencia_plaza_1927_loop.mp4` | `valencia_plaza_1927_loop_vertical.mp4` |

```bash
python render.py --subject correos --res 1600x900 --samples 5 --frames 0:50 --out render_correos/exr  # por tramos de 50
python post.py --src render_correos/exr --dst render_correos/frames              # 16:9
python post.py --src render_correos/exr --dst render_correos/frames_v --vertical # 9:16
```

## Película de 40 s (`film.py`, `film_map.py`, `film_post.py`)

Un único plano en perspectiva, sin cortes: el plano de emplazamiento, doblado y arrugado sobre una
mesa de dibujo (nogal, compás, tintero, plumilla, regla), se despliega y se convierte en el modelo
3D; vista de pájaro en órbita; descenso hasta la altura de los ojos en el taller de cantería
(bancos con sillares a medio labrar, mazas, cinceles, escuadras, artesas, cubos, criba, barricas,
caballetes con sierra, cartel de obra, mula con carro de ruedas de radios); mirada hacia arriba
por la esquina andamiada; travelling lateral por la fachada junto al torno de la cabria, cuyos
engranajes giran al izar; contrapicado de la torre y la cúpula; ascenso a la cenital y el plano se
vuelve a plegar (bucle perfecto). Foco que cambia de distancia y profundidad de campo real.
Figuras articuladas (metaballs, rodilla/codo, faja, alpargatas, boina) y mulas con collera y
arreos; fachada con huecos reales (jambas, alféizar, molduras, balcones de hierro, carpinterías y
persianas, arcos de medio punto en planta baja). Abajo, una lámina técnica dibuja planta, sección
A-A' y axonometría al ritmo de la obra, con la fase y la regla 1915–1922.

```bash
python film.py --frames 57:157 --res 900x1000 --samples 5 --out film/exr   # tramos de 100
python film_map.py --out film/map
python film_post.py --jobs 3   # -> film/frames, luego ffmpeg
```

## Palacio de Comunicaciones (Correos y Telégrafos)

Arquitecto Miguel Ángel Navarro Pérez; construido entre 1915 y 1922 e inaugurado en 1923;
eclecticismo de raíz francesa con toques del modernismo valenciano; esbelta torre metálica
rematada por una esfera armilar; patio de operaciones cubierto por una cúpula de vidriera de
370 piezas (Maumejean). Fuentes: Visit València, Ayuntamiento de València, ONCE, y la
monografía de D. Espada Torres (Universidad de Zaragoza).

Secuencia constructiva (`vlc/correos.py`): replanteo con estacas y cordeles; vaciado del sótano
a mano con salida de tierras en carros; muros de cimentación y sótano; forjado de planta baja;
seis plantas que suben una tras otra — pilares de acero colocados con cabria, muros con cara de
sillería levantados hilada a hilada en una onda que recorre el perímetro, paños de forjado —
con el andamio de rollizos trepando con los muros y un montacargas de madera subiendo y
bajando; columnas y cubierta del patio de operaciones; tambor, nervios de hierro de la cúpula
uno a uno y después el vidrio; ático y balaustrada, pabellones de esquina con mansarda de
pizarra, cuerpo central; torre de celosía por tramos y esfera armilar; acristalamiento del gran
arco, grupo escultórico, ornamentación; andamios desmontados de arriba abajo y retirada de la
valla. Cuadrillas por oficio: excavadores, albañiles sobre el andamio, peones de capacho,
canteros labrando sillares en el taller de la calle lateral (con polvo), gruistas en los
tornos, capataces con planos; mirones en la valla, paseantes, tranvía.

La plaza aparece con su trazado ajardinado anterior a la plataforma de 1927, por coherencia
cronológica. **Supuestos**: el sistema estructural (estructura de acero roblonado + muros de
carga con aplacado de sillería), la distribución en alas alrededor del patio y la forma de los
pabellones son una reconstrucción verosímil de la época, no un levantamiento; las fuentes
consultadas no describen la estructura ni la planta.

---

# Plataforma de la Plaza de Emilio Castelar, 1927 (primera pieza)

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

## Render

- Cycles CPU, 1600×900, 5 muestras + OpenImageDenoise, EXR multicapa (color, profundidad,
  normales) → post-proceso → escalado Lanczos a 1920×1080, H.264 CRF 19, 24 fps, 20 s.
- ~22 s/frame en 4 núcleos. Con *persistent data* Cycles degrada el shader de fachadas tras
  ~150 frames en un mismo proceso, por eso `make_video.sh` renderiza en tramos de 50 frames.
- `stills/` contiene fotogramas de referencia de cada fase.

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
