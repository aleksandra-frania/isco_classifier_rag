"""
Document stores for RAG retrieval using ChromaDB with hybrid search.
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import re


# Default data paths - can be overridden
DEFAULT_DATA_DIR = Path("YOUR_DATA_DIR")


class RAGStore:
    """Wrapper for ChromaDB with hybrid search (vector + BM25)."""
    
    def __init__(self, db_path: str, collection_name: str, openai_client):
        """Initialize the RAG store."""
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_client.api_key,
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def insert(self, chunks: List[Dict[str, Any]]):
        """Insert document chunks into the store."""
        if not chunks:
            return
        
        # Check if collection is already populated
        if self.collection.count() > 0:
            print(f"Collection already contains {self.collection.count()} items. Skipping insertion.")
            return
        
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{'context': chunk['context']} for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # ChromaDB handles embedding and indexing automatically
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid retrieval.
        ChromaDB uses cosine similarity by default (equivalent to vector search).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'text': doc,
                    'context': results['metadatas'][0][i].get('context', ''),
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results


def isco_to_chunks(isco_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert ISCO DataFrame to document chunks."""
    chunks = []
    
    for _, row in isco_df.iterrows():
        code = row["ISCO 08 Code"]
        title = row["Title EN"]
        context = f"ISCO {code}: {title}"
        
        # Build text from all relevant columns
        text_parts = []
        for col in row.index:
            if col not in ["ISCO 08 Code", "Title EN", "Level"]:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    text_parts.append(f"## {col}\n\n{value}")
        
        text = "\n\n".join(text_parts)
        
        if text.strip():
            chunks.append({
                'context': context,
                'text': text
            })
    
    return chunks


def glossary_to_chunks(glossary_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert glossary DataFrame to document chunks."""
    chunks = []
    
    for _, row in glossary_df.iterrows():
        chunks.append({
            'context': row['context'],
            'text': row['text']
        })
    
    return chunks


def pad_isco_codes(text: str) -> str:
    """Pad ISCO codes to 4 digits with trailing zeros."""
    return re.sub(r'\b(\d{1,3})\b', lambda m: m.group(1).ljust(4, '0'), text)


def create_isco_store(db_path: str, openai_client, data_dir: Path = DEFAULT_DATA_DIR) -> RAGStore:
    """Create or load ISCO document store."""
    store = RAGStore(db_path, "isco_collection", openai_client)
    
    # Only load and process data if the collection is empty
    if store.collection.count() == 0:
        print("Loading ISCO data...")
        
        # Load main ISCO data
        isco_df = pd.read_csv(
            data_dir / "ISCO-08 EN Structure and definitions.csv",
            dtype={"Title EN": str}
        )
        
        # Drop unnecessary columns
        isco_df = isco_df.drop(columns=["Level", "Notes"], errors="ignore")
        
        # Pad ISCO codes
        for col in isco_df.select_dtypes(include=['object']).columns:
            isco_df[col] = isco_df[col].apply(
                lambda x: pad_isco_codes(str(x)) if pd.notna(x) else x
            )
        
        # Keep last duplicate (most complete info)
        isco_df = isco_df.drop_duplicates(subset=["ISCO 08 Code"], keep="last")
        
        # Join with translations
        isco_de_fr = pd.read_csv(data_dir / "isco08_de_fr.csv")
        isco_df = isco_df.merge(isco_de_fr, on="ISCO 08 Code", how="left")
        
        # Reorder columns
        cols = list(isco_df.columns)
        if "Title DE/FR" in cols:
            cols.remove("Title DE/FR")
            title_en_idx = cols.index("Title EN")
            cols.insert(title_en_idx + 1, "Title DE/FR")
        if "ISEI value" in cols:
            cols.remove("ISEI value")
            cols.insert(cols.index("Title DE/FR") + 1 if "Title DE/FR" in cols else title_en_idx + 1, "ISEI value")
        isco_df = isco_df[cols]
        
        # Join with Luxembourg data
        isco_lux = pd.read_csv(
            data_dir / "ISCO-08 Luxembourg.csv",
            dtype={"ISCO 08 Code": str}
        )
        isco_df["ISCO 08 Code"] = isco_df["ISCO 08 Code"].astype(str)
        isco_lux["ISCO 08 Code"] = isco_lux["ISCO 08 Code"].astype(str)

        isco_df = isco_df.merge(isco_lux, on="ISCO 08 Code", how="left")

        # Create chunks and insert
        chunks = isco_to_chunks(isco_df)
        print(f"Inserting {len(chunks)} ISCO chunks...")
        store.insert(chunks)
        print("ISCO store created successfully!")
    else:
        print(f"ISCO store loaded with {store.collection.count()} existing chunks.")
    
    return store


def create_glossary_store(db_path: str, openai_client, data_dir: Path = DEFAULT_DATA_DIR) -> RAGStore:
    """Create or load glossary document store."""
    store = RAGStore(db_path, "glossary_collection", openai_client)
    
    # Only load and process data if the collection is empty
    if store.collection.count() == 0:
        print("Loading glossary data...")
        
        # Load school data (now CSV instead of RDS)
        schools_df = pd.read_csv(data_dir / "l1_schools_2025.csv")
        schools_df = schools_df.rename(columns={"codeSchool": "context", "name": "text"})
        schools_df["text"] = "Abbreviation for " + schools_df["text"] + ", a secondary school"
        
        # Load glossary data
        glossary_df = pd.read_csv(data_dir / "glossary.csv")
        
        # Combine
        combined_df = pd.concat([schools_df, glossary_df], ignore_index=True)
        
        # Create chunks and insert
        chunks = glossary_to_chunks(combined_df)
        print(f"Inserting {len(chunks)} glossary chunks...")
        store.insert(chunks)
        print("Glossary store created successfully!")
    else:
        print(f"Glossary store loaded with {store.collection.count()} existing chunks.")
    
    return store