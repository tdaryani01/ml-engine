# ML Engine

## Overview
ML Engine is a high-performance, modular machine learning pipeline architecture designed for scalability, stream processing, and robust model lifecycle management. Engineered for enterprise SaaS environments, it provides a unified configuration-driven framework to handle data ingestion, preprocessing, architecture definition, and real-time inference.

## Key Features
*   **Modular Architecture**: Decoupled ingestion, transformation, and optimization modules defined via strictly typed configuration schemas.
*   **Dual Ingestion Streams**: 
    *   **CSV Provider**: Batch-oriented data loader for historical data and model training.
    *   **Stream Provider**: Real-time feature ingestion utilizing RabbitMQ/AMQP for live, production-grade streaming.
*   **Config-Driven**: Centralized `PipelineConfig` schema ensures consistency across training runs and environment stages.
*   **Adaptive Normalization**: Built-in online z-score normalization (Welford's algorithm) for handling non-stationary data streams.
*   **Diagnostics & Persistence**: Automated metric tracking and model state serialization.

## Core Modules
*   `config/`: Schema definitions (`schema.py`) and constant management (`constants.py`).
*   `data/`: Data provider interface (`base_provider.py`) and specific implementations (`csv_provider.py`, `stream_provider.py`, `ipc_stream_provider.py`).
*   `engine/`: Core execution logic and model orchestration (internal implementations).

## Getting Started

### Installation
1. Clone the repository: `git clone https://github.com/isolinetech/ml-engine`
2. Install dependencies: `pip install -r requirements.txt`

### Basic Usage
To launch a training pipeline, define your `PipelineConfig` and initialize the preferred provider:

```python
from engine.runner import PipelineRunner
from data.csv_provider import CSVDataProvider

# Initialize provider
provider = CSVDataProvider(data_file_path="data/train.csv", ...)

# Execute pipeline
runner = PipelineRunner(config=my_config, data_provider=provider)
runner.run()
