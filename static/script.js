document.getElementById('uploadForm').onsubmit = async function(event) {
    event.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput').files[0];
    const targetColumn = document.getElementById('targetColumn').value;

    if (!fileInput || !targetColumn) {
        alert('Please select a file and enter the target column name!');
        return;
    }
    
    formData.append('file', fileInput);
    formData.append('target_column', targetColumn);
    
    document.getElementById('loading').style.display = 'block';
    
    try {
        const response = await fetch('/', {
            method: 'POST',
            body: formData
        });

        const result = await response.text();
        document.getElementById('result').innerHTML = result;
    } catch (error) {
        console.error('Error:', error);
        alert('Error uploading file');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
};
