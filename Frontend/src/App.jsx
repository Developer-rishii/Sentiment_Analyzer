import React, { useState } from "react";
import axios from "axios";

const App = () => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);

  const analyzeSentiment = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/sentiment/analyze", {
        text: inputText
      });

      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Sentiment Analyzer</h2>

      <textarea
        rows="4"
        style={{ width: "300px" }}
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />

      <br />

      <button onClick={analyzeSentiment} style={{ marginTop: "10px" }}>
        Analyze
      </button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Result:</h3>
          <p><strong>Sentiment:</strong> {result.sentiment}</p>
          <p><strong>Polarity:</strong> {result.polarity}</p>
        </div>
      )}
    </div>
  );
};

export default App;
