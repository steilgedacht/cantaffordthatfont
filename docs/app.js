const { createApp, ref, onMounted} = Vue;

const fontOrder = [
  "Thin",
  "ThinItalic",
  "ExtraLight",
  "ExtraLightItalic",
  "Light",
  "LightItalic",
  "Regular",
  "Italic",
  "Mono",
  "MonoItalic",
  "Medium",
  "MediumItalic",
  "Semibold",
  "SemiboldItalic",
  "SemiBold",
  "SemiBoldItalic",
  "Bold",
  "BoldItalic",
  "ExtraBold",
  "ExtraBoldItalic",
  "Black",
  "BlackItalic"
];

const getWeightFromFilename = (filename) => {
  // Remove extension
  const nameWithoutExt = filename.replace(/\.[^/.]+$/, "");

  // Extract the weight part (assumes format: name-WEIGHT or name-WEIGHT-STYLE)
  for (const weight of fontOrder) {
    if (nameWithoutExt.endsWith(`-${weight}`)) {
      return weight;
    }
  }

  return null; // fallback if not found
};

function getPreferredSubfont(subfonts) {
  const tryMatch = (weight) => 
    subfonts.find(s => s.replace(/\.[^/.]+$/, "").endsWith(`-${weight}`));

  return (
    tryMatch("Medium") ||
    tryMatch("Regular") ||
    tryMatch("Mono") ||
    subfonts[0]
  );
}


async function runModel(inputData) {
    if (!session) {
        console.error("Model session not initialized.");
        return [];
    }

    const feeds = { input: inputData };
    const results = await session.run(feeds);
    const result = results.output;

    // apply softmax to the output
    const softmax = (arr) => {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    };
    
    const softmaxResult = softmax(result.data);

    // get the top 4 predictions
    const arr = Array.from(softmaxResult); // turn it into a regular array

    const topIndices = arr
        .map((value, index) => ({ index, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 4)
        .map(item => item.index);
    
    // load json 
    const fonts = await fetch('fonts.json');
    const fontsJson = await fonts.json();

    const fonts_to_subfonts = await fetch('fonts_to_subfonts.json');
    const fonts_to_subfontsJson = await fonts_to_subfonts.json();


    const topPredictions = topIndices.map(i => {
        const rawSubfonts = fonts_to_subfontsJson[fontsJson[i]];
        // sort by weight using existing helper
        const sortedRaw = rawSubfonts.slice().sort((a, b) => {
            const weightA = getWeightFromFilename(a);
            const weightB = getWeightFromFilename(b);

            const indexA = fontOrder.indexOf(weightA);
            const indexB = fontOrder.indexOf(weightB);

            return indexA - indexB;
        });

        return {
            index: fontsJson[i].replace("[wght]", ""),
            probability: softmaxResult[i],
            // don't store a fixed font_path here; build from selectedSubfont when needed
            subfonts: sortedRaw.map(s => ({ name: s, display: s.replace(/\[.*?\]/g, "").split("-").slice(-1)[0].split(".")[0] })),
            selectedSubfont: getPreferredSubfont(rawSubfonts),
            selectedSubfont_selectbox: getPreferredSubfont(rawSubfonts).replace(/\[.*?\]/g, ""),
            link: "https://fonts.google.com/?query=" + fontsJson[i].replace(/(?<!\\d)([A-Z])/g, ' $1').trim().replace(" ", "+")
        };
    });

    console.log(topPredictions);

    return topPredictions;
}


async function process_image(img) {
    // Create canvas to manipulate the image
    const canvas = document.createElement('canvas');
    const aspectRatio = img.width / img.height;
    const targetHeight = 150;
    const targetWidth = Math.round(aspectRatio * targetHeight);
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

    // Get grayscale pixel data
    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const data = imageData.data;
    const grayData = [];

    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        grayData.push(gray);
    }

    // Check if background is white or black
    const hist = new Array(10).fill(0);
    for (let i = 0; i < grayData.length; i++) {
        const bin = Math.floor((grayData[i] / 256) * 10);
        hist[bin]++;
    }
    const sorted = [...hist].map((count, i) => ({ i, count }))
                            .sort((a, b) => b.count - a.count);
    if (sorted[0].i < sorted[1].i) {
        for (let i = 0; i < grayData.length; i++) {
            grayData[i] = 255 - grayData[i];
        }
    }

    // Normalize to [0, 255]
    const min = Math.min(...grayData);
    const max = Math.max(...grayData);
    const range = max - min || 1;
    const normData = grayData.map(v => ((v - min) / range) * 255);
    
    // Pad to 150x700
    const padded = new Float32Array(150 * 700).fill(255); // white background
    for (let y = 0; y < 150; y++) {
        for (let x = 0; x < Math.min(targetWidth, 700); x++) {
            padded[y * 700 + x] = normData[y * targetWidth + x];
        }
    }

    // Create ONNX tensor
    const inputTensor = new ort.Tensor('float32', padded, [1, 150, 700]);    
    
    return inputTensor;
}

function injectFont(fontName, fontUrl) {
    const styleId = `font-${fontName}`;
    if (document.getElementById(styleId)) return; // Prevent duplicate

    const style = document.createElement('style');
    style.id = styleId;
    style.innerText = `
        @font-face {
            font-family: '${fontName}';
            src: url('${fontUrl}');
        }
    `;
    console.log(`Injecting font: ${fontName} from ${fontUrl}`);
    document.head.appendChild(style);
}

// new: try to preload font via FontFace API, fallback to injectFont
async function preloadFont(rawFilename, fontName) {
    // build path using encodeURIComponent for the filename only
    const fontPath = 'all_fonts_filtered/' + encodeURIComponent(rawFilename);
    if (document.fonts && [...document.fonts].some(f => f.family === fontName)) {
        return; // already added
    }

    try {
        // try to load with FontFace API and wait until the font is usable
        const ff = new FontFace(fontName, `url("${fontPath}") format("truetype")`);
        await ff.load();
        document.fonts.add(ff);
        // ensure the font is available for rendering
        try {
            await document.fonts.load(`16px "${fontName}"`);
            await document.fonts.ready;
        } catch (e) {
            // non-fatal: proceed, font likely loaded
            console.warn('document.fonts.load/ready failed for', fontName, e);
        }
        console.log(`Preloaded font ${fontName} -> ${fontPath}`);
    } catch (e) {
        console.warn(`FontFace load failed for ${fontName}, falling back to @font-face injection:`, e);
        injectFont(fontName, fontPath);
    }
}

function getFontName(pred) {
    // compute font-family name but don't inject here
    const raw = pred.selectedSubfont || (pred.subfonts && pred.subfonts[0] && pred.subfonts[0].name) || '';
    const index = raw.split("-")[0].split(".")[0];
    const selected_subfont = raw.split("-").slice(-1)[0].split(".")[0].replace(/\[.*?\]/g, "");
    const fontName = `PredictedFont-${index}-${selected_subfont}`.replace("[wght]", "");
    return fontName;
}

async function initializeModel() {
    session = await ort.InferenceSession.create('./model_resnet_final_v1.onnx');
}

createApp({
    setup() {
        const image = ref(null);
        const result = ref(null);
        const isDarkMode = ref(window.matchMedia('(prefers-color-scheme: dark)').matches);

        onMounted(() => {
            document.body.classList.toggle('dark-mode', isDarkMode.value);
        });

        let input_data = null;
        let session = null;

        // when clicking on the upload field, trigger the file input
        document.querySelector('.upload_field').addEventListener('click', () => {
            const fileInput = document.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.click();
            }
        });

        async function handleFile(event) {
            const file = event.target.files[0];
            if (!file) return;
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = async () => {

                document.querySelector('.upload_field').style.backgroundImage = `url(${img.src})`;
                document.querySelector('.upload_field').style.backgroundSize = 'contain';
                document.querySelector('.upload_field').style.backgroundPosition = 'center';
                document.querySelector('.upload_field').style.backgroundRepeat = 'no-repeat';
                document.querySelector('.upload_field').textContent = '';
                document.querySelector('.upload_field').style.height = img.height;
                
                input_data = await process_image(img);
                
                const predictions = await runModel(input_data);

                // preload fonts into document.fonts (prevents layout jumps)
                await Promise.all(predictions.map(async (pred, i) => {
                    const raw = pred.selectedSubfont || (pred.subfonts && pred.subfonts[0] && pred.subfonts[0].name);
                    const fontName = getFontName(pred);
                    if (raw) await preloadFont(raw, fontName);
                    pred.fontName = fontName; // store it for later use
                }));

                result.value = predictions;
            };
        }


        onMounted(async () => {
            await initializeModel(); // load model once            

            // prevent multiple registrations if onMounted is called multiple times
            if (window.__pasteHandlerAdded) return;
            window.__pasteHandlerAdded = true;

            document.addEventListener('paste', async (event) => {
                try {
                    const clipboard = event.clipboardData || window.clipboardData;
                    if (!clipboard) return;

                    const items = clipboard.items || clipboard.files || [];
                    for (const item of items) {
                        if (!item) continue;

                        // handle DataTransferItem (has .type) or File (from clipboard.files)
                        const itemType = item.type || (item.name && item.type) || '';
                        if (!itemType) continue;

                        if (itemType.startsWith('image/')) {
                            // only prevent default when we handle an image so normal text paste still works
                            event.preventDefault();

                            const file = (typeof item.getAsFile === 'function') ? item.getAsFile() : (item instanceof File ? item : null);
                            const fallbackFile = (clipboard.files && clipboard.files.length > 0) ? clipboard.files[0] : null;
                            const imgFile = file || fallbackFile;
                            if (!imgFile) continue;

                            const objectUrl = URL.createObjectURL(imgFile);
                            const img = new Image();
                            img.src = objectUrl;
                            img.onload = async () => {
                                const uploadField = document.querySelector('.upload_field');
                                if (uploadField) {
                                    uploadField.style.backgroundImage = `url(${objectUrl})`;
                                    uploadField.style.backgroundSize = 'contain';
                                    uploadField.style.backgroundPosition = 'center';
                                    uploadField.style.backgroundRepeat = 'no-repeat';
                                    uploadField.textContent = '';
                                    uploadField.style.height = img.height + 'px';
                                }

                                const input_data = await process_image(img);
                                const predictions = await runModel(input_data);

                                // preload fonts
                                await Promise.all(predictions.map(async (pred, i) => {
                                    const raw = pred.selectedSubfont || (pred.subfonts && pred.subfonts[0] && pred.subfonts[0].name);
                                    const fontName = getFontName(pred);
                                    if (raw) await preloadFont(raw, fontName);
                                    pred.fontName = fontName;
                                }));

                                result.value = predictions;

                                // free object URL
                                URL.revokeObjectURL(objectUrl);
                            };

                            // we handled an image — stop processing further items
                            break;
                        }
                    }
                } catch (err) {
                    console.error('Paste handler error:', err);
                }
            }, false);
        });


        function toggleDarkMode() {
            isDarkMode.value = !isDarkMode.value;
            document.body.classList.toggle('dark-mode', isDarkMode.value);
        }

        // new: preload when user selects different subfont
        async function onSubfontChange(pred) {
            try {
                const raw = pred.selectedSubfont || (pred.subfonts && pred.subfonts[0] && pred.subfonts[0].name);
                if (!raw) return;
                const fontName = getFontName(pred);
                await preloadFont(raw, fontName);
                pred.fontName = fontName;
            } catch (e) {
                console.error('Subfont load failed', e);
            }
        }

        // show legal text in footer
        function showLegal(type) {
            const container = document.getElementById('legal_text');
            if (!container) return;
            if (type === 'imprint') {
                container.innerHTML = '<p><strong>Imprint</strong></p><p>Benjamin Bergmann<br>Altenberger Straße 9, 415<br>4040 Linz<br>Austria<br><p><a href="mailto:bergmannbenjamin@proton.me">bergmannbenjamin@proton.me</a></p>';
            } else if (type === 'privacy') {
                container.innerHTML = '<p><strong>Privacy Policy</strong></p><p>This website is hosted via GitHub Pages, a service of GitHub, Inc., 88 Colin P. Kelly Jr. Street, San Francisco, CA 94107, USA. When you access the website, technical information such as your IP address, browser type, operating system and time of access is automatically processed by GitHub to ensure operation and security. For details, see the <a href="https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement" target="_blank">GitHub Privacy Statement</a>. This website does not set cookies and does not use tracking or analytics tools. If you enter data into the provided tool, the data is processed locally in your browser only and is not stored or transmitted.</p>';
            }
            // scroll into view smoothly
            container.scrollIntoView({ behavior: 'smooth' });
        }

        onMounted(() => {
            // Initialer Zustand
            document.body.classList.toggle('dark-mode', isDarkMode.value);

            // register footer button handlers
            const btnImprint = document.getElementById('btn-imprint');
            const btnPrivacy = document.getElementById('btn-privacy');
            if (btnImprint) btnImprint.addEventListener('click', () => showLegal('imprint'));
            if (btnPrivacy) btnPrivacy.addEventListener('click', () => showLegal('privacy'));
        });

        return { image, result, handleFile, getFontName, isDarkMode, toggleDarkMode, onSubfontChange };
    },
    template: `
        <button id="dark-mode-toggle" @click="toggleDarkMode">
            {{ isDarkMode ? 'Light Mode' : 'Dark Mode' }}
        </button>
        <input type="file" accept="image/*" @change="handleFile" style="display:none;" />
        <div v-if="result" style="margin-bottom: 200px;">
            <p style="margin-top:20px">Top 4 Predictions</p>
            <div v-for="(pred, index) in result" :key="index" class="prediction">
                <div :style="{ fontFamily: pred.fontName || getFontName(pred) }" class="predicted_font">
                    {{ pred.index }}
                </div>
                <div>
                    <a :href="pred.link" target="_blank">→ Google Fonts</a> | Confidence: {{ (pred.probability * 100).toFixed(2) }}%
                </div>
                <div v-if="pred.subfonts.length > 1" class="subfonts">
                    <select v-model="pred.selectedSubfont" @change="onSubfontChange(pred)">
                        <option v-for="(subfont, subIndex) in pred.subfonts" :key="subIndex" :value="subfont.name">
                            {{ subfont.display }}
                        </option>
                    </select>
                </div>
            </div>
        </div>
    `
}).mount('#app');
