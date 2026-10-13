# 🔍 HalluScan
**Dual-Layer Hallucination Detection for LLMs**

Detects whether an LLM's output is faithful to a source, using DeBERTa NLI + Llama 3.1 8B as judge.

🔗 Live Demo: [your-app.streamlit.app](https://streamlit.io)

---

## How It Works

```
Source + LLM Claim
      ↓
Layer 1: DeBERTa NLI  →  FAITHFUL / NEUTRAL (done, fast)
      ↓ (if ambiguous)
Layer 2: Llama 3.1 8B →  HALLUCINATED / PARTIALLY_HALLUCINATED + reason + span
```

80% of cases resolved by Layer 1 with no API cost.

---

## Setup

### 1. Clone and install
```bash
git clone https://github.com/Prachichi7/halluscan
cd halluscan
pip install -r requirements.txt
```

### 2. Get a Groq API key
Go to [console.groq.com](https://console.groq.com) → free account → create API key.

### 3. Set your API key
```bash
export GROQ_API_KEY="your_key_here"
```

### 4. (Optional) Finetune DeBERTa
Open `model/train.ipynb` in Google Colab with T4 GPU.
Run all cells (~45 min). Download the `finetuned_model/` folder into your repo root.

If you skip this step, the app uses the base pretrained DeBERTa model — still works, just less tuned for hallucinations.

### 5. Run locally
```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select this repo → set `app.py` as entry point
4. Add your Groq key: Settings → Secrets → add `GROQ_API_KEY = "your_key"`
5. Deploy

Done. Your app is live.

---

## Project Structure

```
halluscan/
├── app.py                  ← Streamlit UI
├── requirements.txt
├── src/
│   ├── detector.py         ← DeBERTa Layer 1
│   ├── judge.py            ← Groq/Llama Layer 2
│   └── pipeline.py         ← combines both
├── model/
│   └── train.ipynb         ← Colab finetuning notebook
└── data/
    └── sample_tests.json   ← benchmark test cases
```

---

## Built by
**Prachi Gautam** · MSc Data Science, SRMIST  
[github.com/Prachichi7](https://github.com/Prachichi7) · [LinkedIn](https://linkedin.com/in/prachi-gautam-5a3987261)
