def build_file_description_prompt(
    doc_name: str, doc_id: str, markdown_content: str
) -> str:
    return f"""
You are given content extracted from an ORIGINAL SOURCE FILE
(e.g., PDF, Word, PowerPoint, HTML, spreadsheet, CSV, JSON, code file, etc.).
The provided content is formatted as Markdown only because it is an extraction format.

Your task is to write a description of the ORIGINAL SOURCE FILE that is:
1) retrieval-optimized (good for search, matching queries, keywords, synonyms), AND
2) content-clarifying (helps a reader/agent understand what is in the file and how to use it),
especially for downstream code that may rely on exact column/field names.

Write as if describing the real document/data artifact a user would search for and open.

Style:
- concise but information-dense
- prefer concrete keywords and named entities over narrative
- organize with bullets where helpful
- infer conservatively; only infer what is strongly suggested by the content

Include (when supported by the content):
- purpose + typical use-cases
- intended audience (engineers, finance, legal, ops, researchers, customers, etc.)
- document type and format(s) implied by content (report/spec/policy/deck/code/dataset)
- main topics/sections
- important entities: organizations, products, projects, people (if relevant), locations
- tools/systems/standards/libraries, acronyms (expand acronyms when possible)
- synonyms / alternative terms users might search for

Structured data rules (critical):
- If the file contains structured data (tables, CSV, spreadsheets, JSON, logs):
  - Identify EACH distinct table/dataset separately (Table 1, Table 2, etc. or named if possible).
  - List EXACT column/field names (match spelling/case/punctuation EXACTLY).
  - Describe each column/field meaning, units, allowed values, and relationships when inferable.
  - Provide 2–3 representative example rows/entries:
    - keep them short but faithful to the structure and typical values
    - copy verbatim when possible; do NOT invent columns or values not present
  - If column names are ambiguous or partially missing, say so explicitly and avoid guessing.

Hard constraints:
- Do NOT describe the Markdown or the extraction process.
- Do NOT apologize for missing context.
- Do NOT hallucinate structure, columns, or data values.

Output Markdown sections exactly in this structure (same headings, same order):

## Overview
## Document Type
## Topics
## Key Terms
## Structured Data - Exact Column Names (if exist)
## Structured Data - Column Descriptions (if exist)
## Sampled rows/data (if any)
## Likely Queries
## Keywords

Original filename: {doc_name}
Original file id: {doc_id}

Extracted content (Markdown-formatted, representing the original file):
{markdown_content}
""".strip()
