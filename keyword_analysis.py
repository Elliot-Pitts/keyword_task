import warnings
import logging
from urllib3.exceptions import NotOpenSSLWarning
import asyncio
import aiohttp
import os
import requests
import pandas as pd
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API key
chatGPTclient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Suppress the specific warning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[LoggingHandler()])

# Parameters
REDO_CLUSTERING = True  # Set to True to redo clustering, False to use existing clustered_keywords.csv
REDO_ENRICHMENT = False  # Set to True to redo keyword enrichment, False to use existing enriched_keywords.csv
REDO_GRAPHS = True  # Set to True to redo graph generation, False to use existing files
KEYWORDS_TO_AVOID = []
logging.info("Starting now")

def fetch_data(api_url, payload, headers):
    response = requests.post(api_url, headers=headers, data=payload)
    return response.json()

def fetch_data_for_keywords(keywords):
    logging.info("Fetching search volume data for keywords.")
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"
    payload = json.dumps([{"date_from": "2019-06-30", "date_to": "2024-06-30", "keywords": keywords, "sort_by": "relevance"}])
    headers = {
        'Authorization': f'Basic {os.getenv("DATAFORSEO_API_KEY")}',
        'Content-Type': 'application/json'
    }
    return fetch_data(url, payload, headers)

async def fetch_trends_data_for_keyword(session, keyword, api_type):
    logging.info(f"Fetching trends {api_type} data for keyword: {keyword}")
    if api_type == "google":
        url = "https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live"
    else:
        url = "https://api.dataforseo.com/v3/keywords_data/dataforseo_trends/explore/live"
    payload = json.dumps([{"date_from": "2019-06-30", "date_to": "2024-06-30", "type": "web", "keywords": [keyword]}])
    headers = {
        'Authorization': f'Basic {os.getenv("DATAFORSEO_API_KEY")}',
        'Content-Type': 'application/json'
    }
    async with session.post(url, headers=headers, data=payload) as response:
        trends_data = await response.json()
        if trends_data['tasks_error'] > 0:
            logging.error(f"Error fetching trends data for keyword: {keyword}. Retrying...")
            await asyncio.sleep(5)
            async with session.post(url, headers=headers, data=payload) as response:
                trends_data = await response.json()
                if trends_data['tasks_error'] > 0:
                    logging.error(f"Error fetching trends data for keyword: {keyword}. Skipping...")
                    return keyword, {}, api_type
        logging.info(f"Trends data {api_type} fetched for keyword: {keyword}")
        return keyword, trends_data, api_type

def calculate_trend_direction(data, date_field, value_field, threshold=0.01):
    if data.empty:
        return "no data", 0

    data[date_field] = pd.to_datetime(data[date_field])
    data = data.sort_values(by=date_field)

    X = np.array([d.toordinal() for d in data[date_field]]).reshape(-1, 1)
    y = data[value_field].values

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]

    if abs(slope) < threshold:
        direction = "flat"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return direction, slope


def extract_trends(trends_data):
    if trends_data['tasks'][0]['result']:
        items = trends_data['tasks'][0]['result'][0].get('items', [])
        if items and items[0] and 'data' in items[0] and items[0]['data'] is not None:
            trends = items[0]['data']
            return [{"date_from": trend['date_from'], "date_to": trend['date_to'], "value": trend['values'][0]}
                    for trend in trends if trend['values'][0] is not None and trend['values'][0] > 0]
    return []

async def process_data(search_volume_data):
    enriched_data = []
    trends_data_list = []
    monthly_searches_data_list = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for task in search_volume_data['tasks'][0]['result']:
            keyword = task['keyword']
            tasks.append(fetch_trends_data_for_keyword(session, keyword, "dataforseo"))

        # Batch the tasks in groups of 10
        for i in range(0, len(tasks), 10):
            batch = tasks[i:i+10]
            responses = await asyncio.gather(*batch)

            trends_responses = {resp[0]: resp[1] for resp in responses}

            for keyword in trends_responses.keys():
                task = next(t for t in search_volume_data['tasks'][0]['result'] if t['keyword'] == keyword)
                trends_data = trends_responses[keyword]

                monthly_searches_list = [
                    {"date_from": f"{search['year']}-{search['month']:02d}-01",
                     "date_to": f"{search['year']}-{search['month']:02d}-28",
                     "search_volume": search['search_volume']}
                    for search in task.get('monthly_searches', [])
                    if search['search_volume'] is not None and search['search_volume'] > 0
                ]
                monthly_searches_data_list.extend([{"keyword": keyword, **ms} for ms in monthly_searches_list])

                trends_list = extract_trends(trends_data)
                trends_data_list.extend([{"keyword": keyword, **trend} for trend in trends_list])


                search_direction, search_avg_change = calculate_trend_direction(pd.DataFrame(monthly_searches_list), 'date_from', 'search_volume', )
                trend_direction, trend_avg_change = calculate_trend_direction(pd.DataFrame(trends_list), 'date_from', 'value', 0.001)


                keyword_data = {
                    "keyword": keyword,
                    "competition": task.get('competition', 'N/A'),
                    "competition_index": task.get('competition_index', 'N/A'),
                    "search_volume": task.get('search_volume', 0),
                    "search_trend_direction": search_direction,
                    "search_avg_change": search_avg_change,
                    "trend_direction": trend_direction,
                    "trend_avg_change": trend_avg_change,
                    "trends": trends_list,
                    "monthly_searches": monthly_searches_list,
                }
                enriched_data.append(keyword_data)

    return enriched_data, trends_data_list, monthly_searches_data_list


# Check if the enriched keywords file exists
if os.path.exists('enriched_keywords.csv') and not REDO_ENRICHMENT:
    logging.info("Loading enriched keywords from existing file.")
    enriched_df = pd.read_csv('enriched_keywords.csv')
else:
    logging.info("No existing enriched keywords file found. Fetching data from APIs.")
    keywords_df = pd.read_csv('keywords.csv', header=None, names=['keyword'])
    keywords = [kw for kw in keywords_df['keyword'] if kw not in KEYWORDS_TO_AVOID]

    search_volume_data = fetch_data_for_keywords(keywords)
    enriched_data, trends_data_list, monthly_searches_data_list = asyncio.run(process_data(search_volume_data))

    enriched_df = pd.DataFrame(enriched_data)
    # enriched_df.to_csv('enriched_keywords.csv', index=False)
    # logging.info("Enriched data saved to enriched_keywords.csv.")

    pd.DataFrame(trends_data_list).to_csv('trends_data.csv', index=False)
    logging.info("Trends data saved to trends_data.csv.")

    pd.DataFrame(monthly_searches_data_list).to_csv('keyword_volume.csv', index=False)
    logging.info("Monthly searches data saved to keyword_volume.csv.")

# Check if the clustered keywords file exists
if REDO_CLUSTERING:
    logging.info("Loading clustered keywords from existing file.")
    # enriched_df = pd.read_csv('enriched_keywords.csv')
else:
    logging.info("No existing clustered keywords file found or redoing clustering. Performing clustering.")
    if 'cluster' in enriched_df.columns:
        enriched_df = enriched_df.drop(columns=['cluster'])
    # Filter the keywords based on search volume
    # Calculate mean and standard deviation of the search volume
    mean_volume = enriched_df['search_volume'].mean()
    std_volume = enriched_df['search_volume'].std()

    # Define bounds for filtering
    # Adjust the multiplier to exclude high-volume outliers
    upper_bound = mean_volume + 3 * std_volume
    lower_bound = mean_volume - 3 * std_volume

    # Filter the DataFrame based on the calculated bounds
    filtered_df = enriched_df[(enriched_df['search_volume'] > lower_bound) &
                              (enriched_df['search_volume'] < upper_bound)]

    if not filtered_df.empty:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info(f"Using model: {model}")
        embeddings = model.encode(filtered_df['keyword'].tolist())
        logging.info("Text embeddings created for keywords.")

        # Perform hierarchical clustering
        distance_threshold = 1.5  # Adjust this value based on your data
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        filtered_df['cluster'] = clustering.fit_predict(embeddings)

        # Merge the clusters back into the original DataFrame
        enriched_df = filtered_df
    else:
        logging.info("No keywords meet the search volume criteria for clustering.")
        enriched_df['cluster'] = -1  # Assign a default cluster value for all

    enriched_df.to_csv('enriched_keywords.csv', index=False)
    logging.info("Clustered data saved to enriched_keywords.csv.")

# Filter out the keywords to avoid from the DataFrame
enriched_df = enriched_df[~enriched_df['keyword'].isin(KEYWORDS_TO_AVOID)]

# Summary and Insights
logging.info("Generating summary and insights.")
# Custom aggregation function to collect keywords into a list
def collect_keywords(keywords):
    return list(keywords)

# Group by cluster and aggregate the data
summary = enriched_df.groupby('cluster').agg({
    'search_volume': 'sum',
    'competition_index': 'mean',
    'keyword': collect_keywords
}).reset_index()

# Rename the columns for clarity
summary.rename(columns={'keyword': 'keywords_list'}, inplace=True)

# Function to summarize keywords using OpenAI's GPT-3.5
def summarize_keywords(keywords):
    logging.info(f"Summarize the common themes among the following keywords: {', '.join(keywords)}")
    asyncio.sleep(2)
    prompt = f"Summarize these clustered SEO keywords to find common themes so we can understand why they are similar, only respond with a very short (maximum 5 word sentence): {', '.join(keywords)}"
    response = chatGPTclient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    summary_text = response.choices[0].message.content
    return summary_text

logging.info("Summarizing keywords for each cluster.")
summary['cluster_summary'] = summary['keywords_list'].apply(summarize_keywords)


# Aggregating monthly searches by cluster
logging.info("Aggregating monthly searches by cluster.")
enriched_df['monthly_searches'] = enriched_df['monthly_searches'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)
monthly_searches_list = []
for _, row in enriched_df.iterrows():
    for search in row['monthly_searches']:
        monthly_searches_list.append({
            'cluster': row['cluster'],
            'date_from': search['date_from'],
            'date_to': search['date_to'],
            'search_volume': search['search_volume']
        })
monthly_searches_df = pd.DataFrame(monthly_searches_list)
monthly_searches_df['date_from'] = pd.to_datetime(monthly_searches_df['date_from'])
aggregated_monthly_searches = monthly_searches_df.groupby(['cluster', 'date_from']).agg({'search_volume': 'mean'}).reset_index()
aggregated_monthly_searches.to_csv('cluster_volume.csv', index=False)

# Aggregating trends by cluster
logging.info("Aggregating trends by cluster.")
enriched_df['trends'] = enriched_df['trends'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)
trends_list = []
for _, row in enriched_df.iterrows():
    for trend in row['trends']:
        trends_list.append({
            'cluster': row['cluster'],
            'date_from': trend['date_from'],
            'date_to': trend['date_to'],
            'value': trend['value']
        })
trends_df = pd.DataFrame(trends_list)
trends_df['date_from'] = pd.to_datetime(trends_df['date_from'])
aggregated_trends = trends_df.groupby(['cluster', 'date_from']).agg({'value': 'mean'}).reset_index()
aggregated_trends.to_csv('cluster_trends.csv', index=False)


# Calculate trend direction for search volume and trends
summary['search_volume_direction'], summary['search_volume_slope'] = zip(*summary.apply(
    lambda row: calculate_trend_direction(
        monthly_searches_df[monthly_searches_df['cluster'] == row['cluster']],
        'date_from',
        'search_volume'
    ), axis=1
))

summary['trends_direction'], summary['trends_slope'] = zip(*summary.apply(
    lambda row: calculate_trend_direction(
        trends_df[trends_df['cluster'] == row['cluster']],
        'date_from',
        'value',
        0.001
    ), axis=1
))

# if REDO_GRAPHS then generate the graphs
if REDO_GRAPHS:
    # Plotting the total search volume per cluster with trend direction slopes
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='cluster', y='search_volume')
    plt.title('Total Search Volume per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Total Search Volume')
    for index, row in summary.iterrows():
        plt.text(index, row['search_volume'], f'{row["search_volume_slope"]:.2f}', color='black', ha="center")
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='cluster', y='competition_index')
    plt.title('Average Competition Index per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Competition Index')
    plt.show()

    summary.to_csv('cluster_summary.csv', index=False)
    logging.info("Summary report saved to cluster_summary.csv.")

    # Generate word clouds for each cluster
    logging.info("Generating word clouds for each cluster.")
    for cluster_id in enriched_df['cluster'].unique():
        cluster_keywords = enriched_df[enriched_df['cluster'] == cluster_id]['keyword']
        frequencies = cluster_keywords.value_counts().to_dict()
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for Cluster {cluster_id}')
        plt.axis('off')
        plt.show()

    logging.info("Generating chart for top 10 keywords by volume.")
    top_keywords_by_volume_df = enriched_df.sort_values(by='search_volume', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_keywords_by_volume_df, x='search_volume', y='keyword', palette='viridis')
    plt.title('Top 10 Keywords by Search Volume')
    plt.xlabel('Search Volume')
    plt.ylabel('Keyword')
    plt.show()

    logging.info("Generating chart for top 10 keywords by competition.")
    top_keywords_by_competition_df = enriched_df.sort_values(by='competition_index', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_keywords_by_competition_df, x='competition_index', y='keyword', palette='magma')
    plt.title('Top 10 Keywords by Competition Index')
    plt.xlabel('Competition Index')
    plt.ylabel('Keyword')
    plt.show()

    logging.info("Aggregated cluster trends saved to cluster_trends.csv.")
    plt.figure(figsize=(14, 8))
    for cluster_id in aggregated_trends['cluster'].unique():
        cluster_trends = aggregated_trends[aggregated_trends['cluster'] == cluster_id]
        plt.plot(cluster_trends['date_from'], cluster_trends['value'], label=f'Cluster {cluster_id}')
    plt.title('Aggregated Trends by Cluster')
    plt.xlabel('Date')
    plt.ylabel('Trend Value')
    plt.legend()
    plt.show()

    logging.info("Aggregated cluster volume saved to cluster_volume.csv.")
    plt.figure(figsize=(14, 8))
    for cluster_id in aggregated_monthly_searches['cluster'].unique():
        cluster_searches = aggregated_monthly_searches[aggregated_monthly_searches['cluster'] == cluster_id]
        plt.plot(cluster_searches['date_from'], cluster_searches['search_volume'], label=f'Cluster {cluster_id}')
    plt.title('Aggregated Monthly Searches by Cluster')
    plt.xlabel('Date')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.show()

logging.info("Script completed successfully. Enriched data, cluster summaries, word clouds, trends data, keyword volume data, and top keywords charts have been saved.")
