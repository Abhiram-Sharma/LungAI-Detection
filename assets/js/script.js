document.getElementById('submitBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please upload an image!');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('https://lung-cancer-backend.onrender.com/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        document.getElementById('classification').textContent = `Classification: ${data.classification}`;
        document.getElementById('heatmap').src = `data:image/png;base64,${data.heatmap}`;
        document.getElementById('results').style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        alert('Something went wrong. Please try again.');
    }
});
