const state = {
  imageFile: null,
  imageLoaded: false,
};

const fileInput       = document.getElementById('fileInput');
const imageWrapper    = document.getElementById('imageWrapper');
const imagePlaceholder= document.getElementById('imagePlaceholder');
const previewImg      = document.getElementById('previewImg');
const detectionLabel  = document.getElementById('detectionLabel');
const confLabel       = document.getElementById('confLabel');
const plateBox        = document.getElementById('plateBox');
const analyzeBtn      = document.getElementById('analyzeBtn');
const loadingOverlay  = document.getElementById('loadingOverlay');

const resultEmpty     = document.getElementById('resultEmpty');
const resultData      = document.getElementById('resultData');
const resultPlate     = document.getElementById('resultPlate');
const resultOCR       = document.getElementById('resultOCR');

function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  state.imageFile = file;
  state.imageLoaded = true;

  const url = URL.createObjectURL(file);
  previewImg.src = url;

  imagePlaceholder.style.display = 'none';
  previewImg.style.display = 'block';
  imageWrapper.classList.add('has-image');

  resetResults();

  fileInput.value = '';
}

function resetResults() {
  detectionLabel.style.display = 'none';
  plateBox.style.display = 'none';
  resultData.style.display = 'none';
  resultEmpty.style.display = 'flex';

  plateBox.style.left = '0%';
  plateBox.style.top = '0%';
  plateBox.style.width = '0%';
  plateBox.style.height = '0%';
}

async function analyzePlate() {
  if (!state.imageLoaded) {
    alert('Najpierw wczytaj zdjęcie pojazdu.');
    return;
  }

  showLoading(true);

  try {
    const formData = new FormData();
    formData.append('image', state.imageFile);

    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Błąd odpowiedzi serwera API');

    const data = await response.json();

    displayResults(data.plate, data.confidence, data.detection_confidence, data.bbox);

  } catch (err) {
    console.error('Błąd analizy:', err);
    alert('Wystąpił błąd podczas analizy. Upewnij się, że serwer app.py działa.');
  } finally {
    showLoading(false);
  }
}


function displayResults(plate, ocrConfidence, detectionConfidence, bbox) {
  confLabel.textContent = `${detectionConfidence}%`;
  detectionLabel.style.display = 'block';

  if (bbox) {
    plateBox.style.left = `${bbox.left * 100}%`;
    plateBox.style.top = `${bbox.top * 100}%`;
    plateBox.style.width = `${bbox.width * 100}%`;
    plateBox.style.height = `${bbox.height * 100}%`;
    plateBox.style.display = 'block';
  } else {
    plateBox.style.display = 'none';
  }

  resultEmpty.style.display = 'none';
  resultPlate.textContent = plate;
  resultOCR.textContent = `${ocrConfidence}%`;
  resultData.style.display = 'block';
}

function showLoading(show) {
  loadingOverlay.style.display = show ? 'flex' : 'none';
  analyzeBtn.disabled = show;
}

imageWrapper.addEventListener('dragover', (e) => {
  e.preventDefault();
  imageWrapper.classList.add('drag-over');
});

imageWrapper.addEventListener('dragleave', () => {
  imageWrapper.classList.remove('drag-over');
});

imageWrapper.addEventListener('drop', (e) => {
  e.preventDefault();
  imageWrapper.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    state.imageFile = file;
    state.imageLoaded = true;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    imagePlaceholder.style.display = 'none';
    previewImg.style.display = 'block';
    imageWrapper.classList.add('has-image');
    resetResults();
  }
});

imageWrapper.addEventListener('click', () => {
  if (!state.imageLoaded) {
    fileInput.click();
  }
});