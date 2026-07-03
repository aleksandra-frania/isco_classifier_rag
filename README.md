# ISCO Classification with RAG + OpenAI API

This project implements an **ISCO-08 job classification pipeline** using:

- 🔎 **RAG (Retrieval-Augmented Generation)** with ChromaDB  
- 🧠 **OpenAI GPT models** (`gpt-5-mini`)  
- 📚 Hybrid retrieval (semantic vector search + BM25)  
- ⚡ Parallel batch processing  

It classifies short job descriptions (e.g. written by children about their parents) into official **ISCO-08 occupation codes**, and returns:

- ISCO code  
- ISCO title  
- ISEI value  
- Certainty percentage  
- Seriousness percentage  
- Explanation  
- Token usage + cost  
- Tool usage  

---

# 📁 Project Structure

```
isco_classifier_rag/
│
├── data/                 
│   ├── processed/
│   ├── prompts/
│   │   └── prompt.md
│   └── raw/
│       ├── glossary.csv
│       ├── inputs.csv
│       ├── ISCO-08 EN Structure and definitions.csv
│       ├── ISCO-08 Luxembourg.csv
│       ├── isco08_de_fr.csv
│       └── l1_schools_2025.csv
│
├── output/               
│
├── src/
│   └── isco_classifier/
│       ├── document_stores.py
│       ├── functions.py
│       ├── main.py
│       └── test.py
│
├── README.md
├── requirements.txt
```

---

# ⚙️ Required Directory Placeholders

You **must replace** the following placeholders in `main.py`:

```python
DATA_DIR = Path("YOUR_DATA_DIR") 
PROCESSED_DIR = Path("YOUR_PROCESSED_DIR")
OUTPUT_DIR = Path("YOUR_OUTPUT_DIR")
PROMPTS_DIR = Path("YOUR_PROMPTS_DIR")
```

And modify this line in `main.py`:

```python
input_file = DATA_DIR / "YOUR_INPUT_FILE.csv"
```

Replace with your actual filename.

---


# 🔐 Environment Setup

## 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Set OpenAI API Key

### Mac / Linux
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Windows (PowerShell)
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

---

# 🧠 How It Works

## 1️⃣ RAG Stores

Two ChromaDB stores are created:

- `isco_collection`
- `glossary_collection`

They are stored inside:

```
PROCESSED_DIR/openai_isco.db
PROCESSED_DIR/openai_glossary.db
```

The first time you run the script:

- ISCO CSV files are loaded
- Duplicates are removed
- Translations and ISEI values are merged
- Chunks are created
- Everything is embedded using `text-embedding-3-small`
- Stored persistently

Subsequent runs reuse the stored database.

---

## 2️⃣ Classification Flow

For each job description:

1. System prompt is loaded from:
   ```
   PROMPTS_DIR/prompt.md
   ```

2. Model can call tools:
   - `rag_retrieve_isco_excerpts`
   - `rag_retrieve_glossary_excerpts`

3. Retrieved excerpts are injected into the conversation

4. Model returns a structured first line:

```
ISCO_CODE, PCT_CERTAIN, PCT_SERIOUS
```

5. The system:
   - Looks up title + ISEI
   - Calculates token cost
   - Stores metadata

---

# 🚀 Running Batch Classification

```bash
python main.py
```

If the input file exists, the script will:

- Process jobs in parallel (`max_workers=50`)
- Print timing statistics
- Print cost summary
- Save output to:

```
OUTPUT_DIR/last_batch_classification_results.csv
```

---

# 🧪 Running a Single Test

```bash
python test_single.py
```

Example test query:

```
"Prof op der uni"
```

You will see:

- ISCO Code
- Title
- ISEI
- Certainty
- Explanation
- Token usage
- Cost

---

# 💰 Cost Calculation

Pricing is defined inside `functions.py`:

```python
PRICING = {
    "gpt-5-mini": {
        "input": 0.25,
        "output": 0.025,
        "cached_input": 2.00
    }
}
```

Costs are calculated per 1M tokens.

Each result stores:

- prompt tokens
- completion tokens
- cached tokens
- total cost (10 decimal precision)

---

# ⚡ Performance

The system is optimized for:

- High-throughput parallel processing
- Efficient RAG retrieval
- Minimal tool-call iterations (1–3 typical)

Parallelism controlled by:

```python
max_workers=50
```

Adjust depending on:
- API rate limits
- Server capacity
- Available CPU cores

---

# 🧩 Output Columns

The final CSV contains:

| Column | Description |
|--------|-------------|
| isco | ISCO-08 code |
| isco_title | Official title |
| isei | ISEI value |
| pct_certain | Model certainty |
| pct_serious | Model seriousness rating |
| explanation | Model reasoning |
| elapsed_seconds | Runtime per query |
| any_tools_used | Whether RAG was used |
| total_input | Input tokens |
| total_output | Output tokens |
| total_cached_input | Cached tokens |
| cost | USD cost |
| model | Model used |

---

# 🛠 Customization

## Change model

In `main.py`:

```python
model="gpt-5-mini"
```

---

## Change number of parallel workers

```python
max_workers=50
```

---

## Modify retrieval depth

In `functions.py`:

```python
results = isco_store.search(function_args['text'], top_k=7)
results = glossary_store.search(function_args['text'], top_k=3)
```

---

# 🧱 Important Notes

- ISCO codes are padded to 4 digits automatically
- Duplicate ISCO rows are removed (keep last)
- ISEI values are converted to nullable integers
- Empty job descriptions return code `9999`
- Errors are caught and logged per query

---

# 📌 Summary

This project provides:

- Fully automated ISCO-08 classification
- RAG-powered retrieval
- Parallel processing
- Cost tracking
- Clean structured output
- Persistent vector storage

It is designed for **large-scale occupational classification tasks**.

---
