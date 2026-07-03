"""
Test script for ISCO classification with direct OpenAI API.
Tests single classification using the same knowledge base as main.py.
"""
from main import client, system_prompt, tools, isco_store, glossary_store
from functions import get_isco

# Single query test with the full knowledge base
print("Testing single query with full ISCO and glossary knowledge base...")
print("="*80)

result = get_isco(
    "Prof op der uni",
    client,
    system_prompt,
    tools,
    isco_store,
    glossary_store,
    model="gpt-5-mini"  
)
print(result)
print("\n" + "="*80)




