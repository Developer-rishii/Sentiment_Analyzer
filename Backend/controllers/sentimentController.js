import axios from "axios";

export const analyzeSentiment = async (req, res) => {
  try {
    const { text } = req.body;
    const response = await axios.post(
      "http://localhost:8000/sentiment",
      new URLSearchParams({ text })
    );
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: "Python API error" });
  }
};
