# DataBench Dataset Directory

This directory should contain 10 DataBench CSV datasets. Each dataset needs two files:

1. `{dataset_id}.csv` - The main dataset CSV file
2. `{dataset_id}_questions.csv` - The questions CSV file

## Required Datasets

Place your DataBench datasets in the following subdirectories:

- **001_Forbes/** - Forbes billionaires data
  - `001_Forbes.csv`
  - `001_Forbes_questions.csv`

- **006_London/** - London housing data
  - `006_London.csv`
  - `006_London_questions.csv`

- **007_Fifa/** - FIFA player data
  - `007_Fifa.csv`
  - `007_Fifa_questions.csv`

- **013_Roller/** - Roller coaster data
  - `013_Roller.csv`
  - `013_Roller_questions.csv`

- **014_Airbnb/** - Airbnb listings data
  - `014_Airbnb.csv`
  - `014_Airbnb_questions.csv`

- **020_Real/** - Real estate data
  - `020_Real.csv`
  - `020_Real_questions.csv`

- **030_Professionals/** - Professionals data
  - `030_Professionals.csv`
  - `030_Professionals_questions.csv`

- **040_Speed/** - Speed dating data
  - `040_Speed.csv`
  - `040_Speed_questions.csv`

- **058_US/** - US census data
  - `058_US.csv`
  - `058_US_questions.csv`

- **070_OpenFoodFacts/** - Open Food Facts data
  - `070_OpenFoodFacts.csv`
  - `070_OpenFoodFacts_questions.csv`

## File Format

### Main CSV (`{dataset_id}.csv`)
Standard CSV file with your dataset. Can contain any columns and data types.

### Questions CSV (`{dataset_id}_questions.csv`)
Should contain at least these columns:
- `question` - The question text
- `answer` - The ground truth answer

## Where to Get DataBench Datasets

DataBench datasets can be obtained from:
- The official DataBench repository
- Your organization's data repository
- Contact the DataBench maintainers

## After Adding Files

Once you've added your CSV files, you can run:

```bash
# Test the MCP server
python src/experiments/demo/databench_mcp.py

# Run the example
python src/experiments/demo/databench_mcp_example.py
```

The MCP server will:
1. Load all 10 CSV datasets as a corpus
2. Create vector embeddings for semantic search
3. Expose 2 tools via MCP:
   - `search_databench` - Vector search for summaries
   - `get_document_stream` - Get full document content
