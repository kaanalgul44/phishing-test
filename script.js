async function checkEmail() {
    const emailText = document.getElementById('emailText').value;
    const resultDiv = document.getElementById('result');

    const response = await fetch('https://phishing-test-cbpk.onrender.com', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ email_text: emailText })
    });

    const data = await response.json();
    resultDiv.textContent = "Tahmin: " + (data.prediction || "Bilinmiyor").toUpperCase();
}
