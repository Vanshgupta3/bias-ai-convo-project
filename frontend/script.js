async function analyzeText() {

    const text = document.getElementById("userText").value;

    try {
        const response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error("HTTP Error " + response.status);
        }

        const data = await response.json();

        let html = "<h3>Bias Analysis Result</h3>";
        html += `<p><b>Detected Bias:</b> ${data.bias}</p>`;
        html += `<p><b>Severity:</b> ${data.severity}/5</p>`;
        html += `<p><b>Explanation:</b> ${data.explanation}</p>`;
        html += `<p><b>Correction:</b> ${data.correction}</p>`;

        document.getElementById("resultBox").innerHTML = html;

    } catch (err) {
        document.getElementById("resultBox").innerHTML =
            "<p style='color:red'>Request Failed: " + err + "</p>";
    }
}
