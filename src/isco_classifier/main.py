"""
Main script for ISCO classification using RAG with OpenAI API.
Fast, direct implementation matching the R version's performance.
"""
import os
from pathlib import Path
from openai import OpenAI
from document_stores import create_isco_store, create_glossary_store
import pandas as pd

# Set up absolute paths
DATA_DIR = Path("YOUR_DATA_DIR") 
PROCESSED_DIR = Path("YOUR_PROCESSED_DIR")
OUTPUT_DIR = Path("YOUR_OUTPUT_DIR")
PROMPTS_DIR = Path("YOUR_PROMPTS_DIR")

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load system prompt
prompt_path = PROMPTS_DIR / "prompt.md"
with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Create or connect to stores
isco_store_path = str(PROCESSED_DIR / "openai_isco.db")
glossary_store_path = str(PROCESSED_DIR / "openai_glossary.db")

print("Initializing stores...")
isco_store = create_isco_store(isco_store_path, client, data_dir=DATA_DIR)
glossary_store = create_glossary_store(glossary_store_path, client, data_dir=DATA_DIR)
print("Stores ready!\n")

# Define tools for OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve_isco_excerpts",
            "description": (
                "Use this tool to retrieve the most relevant excerpts from the ISCO "
                "knowledge store for a given text input. This function uses both "
                "vector (semantic) similarity and BM25 text search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to search for in the ISCO knowledge base"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve_glossary_excerpts",
            "description": (
                "Use this tool to retrieve glossary entries for a given text input. "
                "This function uses both vector (semantic) similarity and BM25 text search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to search for in the glossary"
                    }
                },
                "required": ["text"]
            }
        }
    }
]


if __name__ == "__main__":
    from functions import get_isco, parallel_chat
    import time
    
    # Load input data for batch classification
    input_file = DATA_DIR / "YOUR_INPUT_FILE.csv"  
    if input_file.exists():
        print(f"Loading input data from {input_file}...")
        inputs_df = pd.read_csv(input_file)
        
        inputs_df["job"] = inputs_df["job"].fillna("").astype(str)
        inputs_df["description"] = inputs_df["description"].fillna("").astype(str)

        # Create merged column
        inputs_df["job_description"] = (
            inputs_df["job"].str.strip() + " " + inputs_df["description"].str.strip()
        ).str.strip()
        
        # Assuming the CSV has a column with job descriptions
        job_column = 'job_description'  # Change this to match your CSV column name
        
        if job_column in inputs_df.columns:
            inputs = inputs_df[job_column].tolist()
            
            print(f"\nProcessing {len(inputs)} job descriptions in parallel...")
            start_time = time.time()
            
            results = parallel_chat(
                inputs,
                client,
                system_prompt,
                tools,
                isco_store,
                glossary_store,
                max_workers=50,
                model="gpt-5-mini"
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Convert to DataFrame
            results_data = [r.to_dict() for r in results if r is not None]
            results_df = pd.DataFrame(results_data)
            
            # Ensure isei is saved as integer (not float)
            if 'isei' in results_df.columns:
                results_df['isei'] = results_df['isei'].astype('Int64')  # pandas nullable integer
            
            # Combine with original data
            output_df = pd.concat([inputs_df, results_df], axis=1)
            
            # Save results with proper float formatting for cost
            output_file = OUTPUT_DIR / "last_batch_classification_results.csv"
            output_df.to_csv(output_file, index=False, float_format='%.10f')
            print(f"\nResults saved to {output_file}")
            print(f"\nFirst few results:")
            print(results_df.head())
            
            # Print timing and summary statistics
            print(f"\n{'='*60}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"Jobs processed: {len(results_df)}")
            print(f"Average time per job: {elapsed_time/len(results_df):.2f} seconds")
            print(f"Throughput: {len(results_df)/(elapsed_time/60):.2f} jobs/minute")
        
        
            if 'cost' in results_df.columns:
                total_cost = results_df['cost'].sum()
                print(f"\nTotal cost: ${total_cost:.10f}")
                print(f"Average cost per query: ${total_cost/len(results_df):.10f}")
        else:
            print(f"Warning: Column '{job_column}' not found in {input_file}")
            print(f"Available columns: {list(inputs_df.columns)}")
    else:
        print(f"\nNo input file found at {input_file}")
        print("Place your CSV file with job descriptions there to run batch classification.")