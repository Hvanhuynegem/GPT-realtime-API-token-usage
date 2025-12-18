# GPT-realtime-API-token-usage

This project is used to benchmark and analyze OpenAI models with a focus on token usage, latency, and cost.  
It supports both the Realtime WebSocket API and the REST `/v1/responses` API, and logs detailed per-sample metrics for later analysis and plotting.

## Requirements

- .NET SDK (recommended: .NET 8 or newer)
- Python 3.x (for plotting)
- An OpenAI API key

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
````

On Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

## Running the experiments

To run the project, open a terminal in the root directory and execute:

```bash
dotnet run
```

This will execute the code in `Program.cs`.
The experiment that runs depends on which function call is currently uncommented in `Program.cs` (for example `DatasetExperiment.RunAsync()`).

During execution, the program:

* Loads the dataset
* Sends samples to the selected model
* Measures latency and token usage
* Stores results in the `logs/` directory

Each run creates a new experiment folder containing:

* `config.json`
* `metrics.csv`
* `responses.jsonl`

## Generating plots

After running experiments, you can generate plots using the provided Python script.

From the terminal:

```bash
python plots.py
```

Alternatively, you can open and run `plots.py` from an IDE.

The script automatically processes the most recent experiment results and generates plots comparing models, including:

* Token usage
* Latency
* Cost estimates

The generated plots are saved to disk for further analysis.

## Notes

* The dataset is expected to be located in the `data/` directory.
* Model selection and configuration are handled in the code, not via command-line arguments.
* Both Realtime and REST models can be benchmarked using the same pipeline.

