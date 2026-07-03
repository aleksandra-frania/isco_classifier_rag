"""
Functions for ISCO classification and result processing using OpenAI direct API.
"""
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path


# OpenAI API Pricing (as of 2025, in USD per 1M tokens)
PRICING = {
    "gpt-5-mini": {
        "input": 0.25,
        "output": 0.025,
        "cached_input": 2.00
    }
}


def calculate_cost(tokens: Dict[str, int], model: str = "gpt-5-mini") -> float:
    """
    Calculate the cost of API usage based on token counts.
    
    Args:
        tokens: Dictionary with token counts
        model: Model name
    
    Returns:
        Cost in USD
    """
    if model not in PRICING:
        return 0.0
    
    pricing = PRICING[model]
    
    input_tokens = tokens.get('prompt_tokens', 0)
    output_tokens = tokens.get('completion_tokens', 0)
    cached_tokens = tokens.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    
    # Calculate regular input tokens (total input - cached)
    regular_input_tokens = input_tokens - cached_tokens
    
    # Calculate cost (convert from per 1M to actual cost)
    input_cost = (regular_input_tokens / 1_000_000) * pricing['input']
    cached_cost = (cached_tokens / 1_000_000) * pricing['cached_input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    
    total_cost = input_cost + cached_cost + output_cost
    
    return total_cost


@dataclass
class ISCOResult:
    """Result from ISCO classification."""
    isco: Optional[str]
    pct_certain: int
    pct_serious: int
    isco_title: Optional[str] = None
    isei: Optional[int] = None  # Changed to int
    explanation: Optional[str] = None
    elapsed_seconds: float = 0.0
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    tokens: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    model: str = "gpt-5-mini"
    
    def __str__(self):
        """String representation of the result."""
        lines = [
            f"ISCO Code: {self.isco or 'NA'}",
        ]
        
        if self.isco_title:
            lines.append(f"Title: {self.isco_title}")
        
        if self.isei is not None:
            lines.append(f"ISEI: {self.isei}")
        
        lines.append(f"Certainty: {self.pct_certain}% | Seriousness: {self.pct_serious}%")
        lines.append(f"Elapsed: {self.elapsed_seconds:.2f} seconds")
        
        if self.tokens:
            total_input = self.tokens.get('prompt_tokens', 0)
            total_output = self.tokens.get('completion_tokens', 0)
            total_cached = self.tokens.get('prompt_tokens_details', {}).get('cached_tokens', 0)
            
            token_str = f"Tokens: {total_input} in / {total_output} out"
            if total_cached > 0:
                token_str += f" ({total_cached} cached)"
            token_str += f" | Cost: ${self.cost:.10f}"
            lines.append(token_str)
        
        if self.explanation:
            lines.append(f"\nExplanation:\n{self.explanation}")
        
        if self.tools_used:
            lines.append(f"\nTools used: {len(self.tools_used)} call(s)")
            for tool in self.tools_used:
                tool_line = f"  - {tool['name']}"
                if 'arguments' in tool and 'text' in tool['arguments']:
                    tool_line += f" ('{tool['arguments']['text']}')"
                lines.append(tool_line)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            'isco': self.isco,
            'pct_certain': self.pct_certain,
            'pct_serious': self.pct_serious,
            'isco_title': self.isco_title,
            'isei': self.isei,
            'explanation': self.explanation,
            'elapsed_seconds': self.elapsed_seconds,
            'any_tools_used': len(self.tools_used) > 0,
            'cost': round(self.cost, 10),  # Round to 10 decimal places for storage
            'model': self.model
        }
        
        # Add token information
        if self.tokens:
            result['total_input'] = self.tokens.get('prompt_tokens', 0)
            result['total_output'] = self.tokens.get('completion_tokens', 0)
            result['total_cached_input'] = self.tokens.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        else:
            result['total_input'] = None
            result['total_output'] = None
            result['total_cached_input'] = None
        
        return result


def handle_tool_calls(client, messages, tools, isco_store, glossary_store, 
                     model="gpt-5-mini", max_iterations=5):
    """
    Handle tool calls in a loop until completion.
    Optimized for speed - typically completes in 1-3 iterations.
    """
    tools_used = []
    total_tokens = {}
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Accumulate token usage
        if response.usage:
            for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                total_tokens[key] = total_tokens.get(key, 0) + getattr(response.usage, key, 0)
            
            # Handle prompt_tokens_details if available
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                if 'prompt_tokens_details' not in total_tokens:
                    total_tokens['prompt_tokens_details'] = {}
                details = response.usage.prompt_tokens_details
                for key in ['cached_tokens', 'audio_tokens']:
                    if hasattr(details, key):
                        val = getattr(details, key, 0) or 0
                        total_tokens['prompt_tokens_details'][key] = \
                            total_tokens['prompt_tokens_details'].get(key, 0) + val
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # Check if there are tool calls
        if not assistant_message.tool_calls:
            return messages, tools_used, total_tokens
        
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Record tool usage
            tools_used.append({
                'id': tool_call.id,
                'name': function_name,
                'arguments': function_args
            })
            
            # Execute the appropriate tool (fast retrieval)
            if function_name == "rag_retrieve_isco_excerpts":
                results = isco_store.search(function_args['text'], top_k=7)
            elif function_name == "rag_retrieve_glossary_excerpts":
                results = glossary_store.search(function_args['text'], top_k=3)
            else:
                results = []
            
            # Format results as text (compact format for speed)
            result_text = "\n\n".join([
                f"Context: {r['context']}\n{r['text']}" 
                for r in results
            ])
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text or "No results found."
            })
    
    return messages, tools_used, total_tokens


def process_isco_result(messages: List[Dict], tools_used: List[Dict], 
                       tokens: Dict, elapsed: float, model: str,
                       isco_df: pd.DataFrame = None) -> ISCOResult:
    """Process the final response into an ISCOResult object."""
    # Get the last assistant message
    response_text = ""
    for msg in reversed(messages):
        # If it's a ChatCompletionMessage object
        if hasattr(msg, "role") and hasattr(msg, "content"):
            if msg.role == "assistant" and msg.content:
                response_text = msg.content
                break
        # If it's a dictionary
        elif isinstance(msg, dict):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                response_text = msg["content"]
                break
    
    # Parse the response
    lines = response_text.strip().split('\n')
    first_line = lines[0].strip()
    
    # Extract comma-separated values
    values = [v.strip() for v in first_line.split(',')]
    
    # Calculate cost first so it's available for all return paths
    cost = calculate_cost(tokens, model)
    
    if len(values) < 3:
        return ISCOResult(
            isco=None,
            pct_certain=0,
            pct_serious=0,
            explanation=f"Unexpected format: {response_text[:200]}",
            elapsed_seconds=elapsed,
            tools_used=tools_used,
            tokens=tokens,
            cost=cost,
            model=model
        )
    
    try:
        isco_code = None if values[0] == "NA" else values[0]
        pct_certain = int(values[1])
        pct_serious = int(values[2])
    except (ValueError, IndexError) as e:
        return ISCOResult(
            isco=None,
            pct_certain=0,
            pct_serious=0,
            explanation=f"Parse error: {e}",
            elapsed_seconds=elapsed,
            tools_used=tools_used,
            tokens=tokens,
            cost=cost,
            model=model
        )
    
    # Extract explanation (everything after first line)
    explanation = '\n'.join(line.strip() for line in lines[1:] if line.strip())
    
    # Get ISCO info if code is valid
    isco_title = None
    isei = None
    
    if isco_code:
        # Load ISCO info if not provided
        if isco_df is None:
            data_dir = Path("YOUR_DATA_DIR")
            isco_path = data_dir / "ISCO-08 EN Structure and definitions.csv"
            if isco_path.exists():
                isco_df = pd.read_csv(isco_path)
        
        if isco_df is not None and not isco_df.empty:
            isco_df['ISCO 08 Code'] = isco_df['ISCO 08 Code'].astype(str)
            matching_rows = isco_df[isco_df['ISCO 08 Code'] == str(isco_code)]
            if matching_rows is not None and not matching_rows.empty:
                row = matching_rows.iloc[0]
                isco_title = row.get('Title EN')

                raw_isei = row.get('ISEI value')
                if pd.notna(raw_isei):
                    isei = int(raw_isei)

                
    if not isco_code:
        isco_code = "9999"
        if not explanation:
            explanation = "Unclassifiable job description."
    
    return ISCOResult(
        isco=isco_code,
        pct_certain=pct_certain,
        pct_serious=pct_serious,
        isco_title=isco_title,
        isei=isei,
        explanation=explanation if explanation else None,
        elapsed_seconds=elapsed,
        tools_used=tools_used,
        tokens=tokens,
        cost=cost,
        model=model
    )



def get_isco(query: str, client, system_prompt: str, tools: List[Dict],
             isco_store, glossary_store, model: str = "gpt-5-mini", 
             isco_df: pd.DataFrame = None) -> ISCOResult:
    """
    Classify a job description using the ISCO classification system.
    Fast implementation targeting ~6 seconds per query like R version.
    
    Args:
        query: Job description to classify
        client: OpenAI client
        system_prompt: System prompt for the model
        tools: Tool definitions for RAG
        isco_store: ISCO document store
        glossary_store: Glossary document store
        model: Model to use for classification
        isco_df: Optional ISCO DataFrame for lookups (will load if not provided)
    
    Returns:
        ISCOResult object with classification details
    """
    start_time = time.time()
    
    if query is None or not query.strip():
        elapsed = time.time() - start_time
        return ISCOResult(
            isco="9999",
            pct_certain=100,
            pct_serious=0,
            isco_title="Not classifiable / empty description",
            explanation="Empty job and description fields.",
            elapsed_seconds=elapsed,
            tools_used=[],
            tokens={},
            cost=0.0,
            model=model
        )
    
    # Initialize messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    # Handle tool calls (typically 1-3 iterations)
    messages, tools_used, tokens = handle_tool_calls(
        client, messages, tools, isco_store, glossary_store, model
    )
    
    elapsed = time.time() - start_time
    
    return process_isco_result(messages, tools_used, tokens, elapsed, model, isco_df)


def parallel_chat(queries: List[str], client, system_prompt: str, tools: List[Dict],
                 isco_store, glossary_store, max_workers: int = 20,
                 model: str = "gpt-5-mini", isco_df: pd.DataFrame = None) -> List[ISCOResult]:
    """
    Process multiple queries in parallel for maximum speed.
    
    Args:
        queries: List of job descriptions to classify
        client: OpenAI client
        system_prompt: System prompt for the model
        tools: Tool definitions for RAG
        isco_store: ISCO document store
        glossary_store: Glossary document store
        max_workers: Maximum number of parallel workers
        model: Model to use for classification
        isco_df: Optional ISCO DataFrame for lookups (will load once if not provided)
    
    Returns:
        List of ISCOResult objects
    """
    # Load ISCO info once for all queries if not provided
    if isco_df is None:
        data_dir = Path("/home/users/afrania/LUCET/isco-classifier-claude/data/raw")

        # Load ISCO definitions (REQUIRED for RAG + titles)
        defs_path = data_dir / "ISCO-08 EN Structure and definitions.csv"
        isco_df = pd.read_csv(defs_path)
        isco_df['ISCO 08 Code'] = isco_df['ISCO 08 Code'].astype(str)

        # Load ISEI values (metadata only)
        isei_path = data_dir / "isco08_de_fr.csv"
        if isei_path.exists():
            isei_df = pd.read_csv(isei_path)
            isei_df['ISCO 08 Code'] = isei_df['ISCO 08 Code'].astype(str)

            # Convert ISEI float → int at LOAD TIME
            isei_df['ISEI value'] = (
                isei_df['ISEI value']
                .where(pd.notna(isei_df['ISEI value']))
                .astype('Int64')   # pandas nullable integer
            )

            # Merge ISEI into definitions
            isco_df = isco_df.merge(
                isei_df[['ISCO 08 Code', 'ISEI value']],
                on='ISCO 08 Code',
                how='left'
            )

    
    results = [None] * len(queries)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                get_isco, query, client, system_prompt, tools,
                isco_store, glossary_store, model, isco_df
            ): idx
            for idx, query in enumerate(queries)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing query {idx}: {e}")
                results[idx] = None
    
    return results


def isco_diff_position(code1: Optional[str], code2: Optional[str]) -> Optional[int]:
    """
    Find the first position where two ISCO codes differ.
    
    Args:
        code1: First ISCO code
        code2: Second ISCO code
    
    Returns:
        Position of first difference (1-indexed), or 0 if identical, or None if either is NA
    """
    if code1 is None or code2 is None:
        return None
    
    if code1 == code2:
        return 0
    
    # Find first differing position
    for i, (c1, c2) in enumerate(zip(code1, code2), start=1):
        if c1 != c2:
            return i
    
    # One is a prefix of the other
    return min(len(code1), len(code2)) + 1