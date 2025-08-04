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


    const topPredictions = topIndices.map(i => ({
        index: fontsJson[i],
        probability: softmaxResult[i],
        font_path: 'all_fonts_filtered/' + fonts_to_subfontsJson[fontsJson[i]][0],
        subfonts: fonts_to_subfontsJson[fontsJson[i]].sort((a, b) => {
            const weightA = getWeightFromFilename(a);
            const weightB = getWeightFromFilename(b);

            const indexA = fontOrder.indexOf(weightA);
            const indexB = fontOrder.indexOf(weightB);

            return indexA - indexB;
        }),
        selectedSubfont:  fonts_to_subfontsJson[fontsJson[i]][0],
        link: "https://fonts.google.com/?query=" + fontsJson[i].replace(/(?<!\d)([A-Z])/g, ' $1').trim().replace(" ", "+")
    }));

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
    document.head.appendChild(style);
}

function getFontName(pred) {
    index = pred.selectedSubfont.split("-")[0];
    let selected_subfont = pred.selectedSubfont.split("-").slice(-1)[0].split(".")[0];
    const fontName = `PredictedFont-${index}-${selected_subfont}`;
    injectFont(fontName, 'all_fonts_filtered/' + pred.selectedSubfont);
    return fontName;
}

async function initializeModel() {
    session = await ort.InferenceSession.create('./model_resnet_final_v1.onnx');
}

createApp({
    setup() {
        const image = ref(null);
        const result = ref(null);

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

                // Inject fonts into <head>
                predictions.forEach((pred, i) => {
                    const fontName = `PredictedFont-${i}`;
                    injectFont(fontName, pred.font_path);
                    pred.fontName = fontName; // store it for later use
                });

                result.value = predictions;
            };
        }


        onMounted(async () => {
            await initializeModel(); // load model once            
            window.addEventListener('paste', async (event) => {
                const items = event.clipboardData.items;
                for (const item of items) {
                    if (item.type.startsWith('image/')) {
                        const file = item.getAsFile();
                        if (file) {
                            const img = new Image();
                            img.src = URL.createObjectURL(file);
                            img.onload = async () => {
                                document.querySelector('.upload_field').style.backgroundImage = `url(${img.src})`;
                                document.querySelector('.upload_field').style.backgroundSize = 'contain';
                                document.querySelector('.upload_field').style.backgroundPosition = 'center';
                                document.querySelector('.upload_field').style.backgroundRepeat = 'no-repeat';
                                document.querySelector('.upload_field').textContent = '';
                                document.querySelector('.upload_field').style.height = img.height;

                                const input_data = await process_image(img);
                                const predictions = await runModel(input_data);

                                // Inject fonts
                                predictions.forEach((pred, i) => {
                                    const fontName = `PredictedFont-${i}`;
                                    injectFont(fontName, pred.font_path);
                                    pred.fontName = fontName;
                                });

                                result.value = predictions;
                            };
                        }
                    }
                }
            });
        });


        return { image, result, handleFile, getFontName};
    },
    template: `
        <input type="file" accept="image/*" @change="handleFile" style="display:none;" />
        <div v-if="result" style="margin-bottom: 200px;">
            <p style="margin-top:20px">Top 4 Predictions</p>
            <div v-for="(pred, index) in result" :key="index" class="prediction">
                <div :style="{ fontFamily: getFontName(pred) }" class="predicted_font">
                    {{ pred.index }}
                </div>
                <div>
                    <a :href="pred.link" target="_blank">→ Google Fonts</a> | Confidence: {{ (pred.probability * 100).toFixed(2) }}%
                </div>
                <div v-if="pred.subfonts.length > 1" class="subfonts">
                    <select v-model="pred.selectedSubfont">
                        <option v-for="(subfont, subIndex) in pred.subfonts" :key="subIndex" :value="subfont">
                            {{ subfont.split("-").slice(-1)[0].split(".")[0] }}
                        </option>
                    </select>
                </div>
            </div>
        </div>
    `
}).mount('#app');
