import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/ping", (req, res) => {
  res.json({ ok: true, message: "API running" });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log("API on", PORT));
