# Keyword Analysis and Clustering

This repository contains a Python script for analyzing and clustering keywords based on their search volume and trends. The script fetches data from DataForSEO and OpenAI APIs, processes the data, and generates various insights including trend analysis, keyword clustering, and summary reports.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [Output](#output)

## Installation

1. **Clone the repository:**
```
   git clone https://github.com/Elliot-Pitts/keyword_task.git
   cd keyword_task
```

3. **Install dependencies:**
```
   pip install -r requirements.txt
```
## Configuration

1. **Set up your API keys:**

   Create a `.env` file in the project root directory and add your API keys:
```
   DATAFORSEO_API_KEY=your_dataforseo_api_key
   OPENAI_API_KEY=your_openai_api_key
```
Add your keywords.csv file to the folder

## Usage

Run the script:
```
   python keyword_analysis.py
```
## Features

- Fetches search volume and trends data from DataForSEO.
- Analyzes and clusters keywords based on search volume and trends.
- Generates summary reports and visualizations.
- Uses OpenAI's GPT-3.5 to summarize the common themes among keywords in each cluster.

## Output

- `enriched_keywords.csv`: Enriched keyword data with search volume and trends.
- `trends_data.csv`: Trends data for each keyword.
- `keyword_volume.csv`: Monthly search volumes for each keyword.
- `cluster_summary.csv`: Summary report of each cluster.
- `cluster_trends.csv`: Aggregated trends data for each cluster.
- `cluster_volume.csv`: Aggregated monthly searches for each cluster.

