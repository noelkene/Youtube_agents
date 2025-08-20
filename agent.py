import os
import json
import pandas as pd
from typing import Dict, Any

# Vertex AI and GenAI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import aiplatform
import google.generativeai as genai
from google.generativeai import types

# ADK imports
from agents import Agent
from agents.tools import ToolContext

# --- Configuration ---
# Please replace with your actual Google Cloud Project ID
PROJECT_ID = "platinum-banner-303105"

REGION = "us-central1"  # Example region

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# --- Model Definitions ---
# Using Gemini 1.5 Pro for its large context window and advanced reasoning
SYNTHESIS_MODEL_NAME = "gemini-2.5-pro"
# Using Gemini 1.5 Flash for its speed and efficiency in targeted searches
SEARCH_MODEL_NAME = "gemini-2.5-pro"
SYNTHESIS_MODEL_OBJECT = GenerativeModel(SYNTHESIS_MODEL_NAME)

# --- File Paths ---
OUTPUT_DIR = "samples/YouTubeAgent/output"
# The input CSV file provided by the user
DATA_FILE = "samples/YouTubeAgent/music_engagement_data.csv"
# Intermediate file to store detected anomalies
ANOMALY_FILE = os.path.join(OUTPUT_DIR, "anomalies.json")
# Final output file for the human-readable report
FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "final_analysis_report.md")

# --- Analysis Parameters ---
# Using a small window as the provided dataset spans only a few days
MOVING_AVG_WINDOW = 2
# A lower threshold to detect spikes in the short-term data
STD_DEV_THRESHOLD = 2.0

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Google Search Function ---
def google_search(query: str) -> str:
    """
    Performs a Google search using the specified model and returns the result.
    """
    print(f"Executing Google Search with query: '{query}'")
    try:
        # Using the google.generativeai library for the search tool
        model = genai.GenerativeModel('gemini-1.5-flash-001', tools=[types.Tool.from_google_search_retrieval(types.GoogleSearchRetrieval())])
        response = model.generate_content(query, tool_config={'google_search_retrieval': {'max_references_per_query': 5}})
        return response.text.strip()
    except Exception as e:
        print(f"An error occurred during Google Search: {e}")
        return f"Error: Could not perform search for query '{query}'."


# --- Tool Functions for Agents ---

def data_ingestion_and_anomaly_detection_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    (Data Ingestion & Anomaly Detection Agent)
    Reads music engagement data, identifies statistical anomalies (spikes)
    in view counts for each track in each country, and saves them to a file.
    """
    print("\n--- Executing Data Ingestion and Anomaly Detection Tool ---")
    all_anomalies = []
    try:
        # --- ROBUSTNESS FIX ---
        # Read the CSV first, then attempt to convert dates.
        # errors='coerce' will turn any unparseable dates into NaT (Not a Time).
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop any rows where the date could not be parsed.
        df.dropna(subset=['date'], inplace=True)
        # --- END FIX ---

        df.sort_values(by=['track_id', 'country', 'date'], inplace=True)

        # Group data by each unique track in each country
        grouped = df.groupby(['track_id', 'track_name', 'artist_name', 'country'])

        for name, group in grouped:
            track_name, country = name[1], name[3]
            print(f"Analyzing: Track '{track_name}' in {country}")

            if len(group) < MOVING_AVG_WINDOW:
                continue

            # Calculate rolling statistics on PREVIOUS days' data using .shift(1)
            group = group.copy()
            shifted_views = group['views'].shift(1)
            group['moving_avg'] = shifted_views.rolling(window=MOVING_AVG_WINDOW).mean()
            group['moving_std'] = shifted_views.rolling(window=MOVING_AVG_WINDOW).std().fillna(0)
            group['upper_band'] = group['moving_avg'] + (group['moving_std'] * STD_DEV_THRESHOLD)

            # Drop initial rows where rolling metrics can't be calculated
            group.dropna(subset=['moving_avg'], inplace=True)

            # Identify anomalies where views are significantly higher than the average
            anomalies_df = group[(group['views'] > group['upper_band']) & (group['views'] > 1000)]

            for index, row in anomalies_df.iterrows():
                anomaly_data = {
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "track_id": row['track_id'],
                    "track_name": row['track_name'],
                    "artist_name": row['artist_name'],
                    "country": row['country'],
                    "views": int(row['views']),
                    "local_average": round(row['moving_avg'], 2),
                    "platform": row['platform']
                }
                all_anomalies.append(anomaly_data)
                print(f"  > Anomaly Detected! On {anomaly_data['date']}, views spiked to {anomaly_data['views']:,} (Avg: {anomaly_data['local_average']:,})")

    except FileNotFoundError:
        return {"status": "error", "message": f"Data file not found: {DATA_FILE}"}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred during analysis: {e}"}

    if not all_anomalies:
        print("No significant anomalies found.")
        return {"status": "success", "message": "No significant anomalies found."}

    # Save the detected anomalies to the JSON file
    with open(ANOMALY_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_anomalies, f, indent=4)

    print(f"\nSuccessfully saved {len(all_anomalies)} anomalies to {ANOMALY_FILE}")
    return {"status": "success", "anomalies_found": len(all_anomalies), "output_file": ANOMALY_FILE}

def hypothesis_evidence_and_reporting_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    (Hypothesis, Evidence & Reporting Agents)
    Reads anomalies, researches potential causes using Google Search,
    and synthesizes the findings into a final report.
    """
    print("\n--- Executing Hypothesis, Evidence, and Reporting Tool ---")
    try:
        with open(ANOMALY_FILE, 'r', encoding='utf-8') as f:
            anomalies = json.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": f"Anomaly file not found: {ANOMALY_FILE}. Run the detection tool first."}

    researched_anomalies = []
    for anomaly in anomalies:
        # Generate a targeted search query to find context (Hypothesis Generation)
        query = (f"What caused a spike in interest for the song \"{anomaly['track_name']}\" by "
                 f"\"{anomaly['artist_name']}\" in {anomaly['country']} around {anomaly['date']}? "
                 f"Look for social media trends, TikTok challenges, celebrity endorsements, or local events.")

        # Gather evidence for the hypothesis
        print(f"Investigating anomaly for '{anomaly['track_name']}' in {anomaly['country']}...")
        search_summary = google_search(query)

        anomaly_with_research = anomaly.copy()
        anomaly_with_research["research_summary"] = search_summary
        researched_anomalies.append(anomaly_with_research)

    # Synthesize the final report (Reporting Agent)
    prompt = f"""
    You are a senior music industry analyst. Your task is to write an insightful report
    explaining significant engagement spikes for an artist's tracks on YouTube Shorts.

    You have been provided with a list of statistical anomalies and a corresponding
    AI-generated research summary for each one. Your job is to synthesize this information
    into a clear, concise, and actionable report.

    **Data (Anomalies and Research Summaries):**
    ---
    {json.dumps(researched_anomalies, indent=2)}
    ---

    **Report Generation Instructions:**
    1.  **Main Title:** Start with a clear headline, like "Analysis of Engagement Anomalies for Synthwave Surfer".
    2.  **Executive Summary:** Write a brief paragraph summarizing the key findings. What were the main drivers of engagement spikes?
    3.  **Detailed Anomaly Analysis:** For each anomaly, create a separate section with a subheading (e.g., "Spike for 'Neon Rider' in USA on 2025-07-04").
        *   **State the Anomaly:** Clearly describe the event (e.g., "The track 'Neon Rider' experienced a significant viewership spike in the USA, reaching X views against a recent average of Y.").
        *   **Synthesize the Cause:** Review the provided 'research_summary'. Do not just copy it. Interpret the information and state the most plausible cause. For instance, "Our research strongly suggests this spike was driven by the song's use in 4th of July celebration videos trending on social media." or "The spike correlates with a new dance challenge that emerged on TikTok in Germany."
        *   **Confidence Score:** Assign a confidence level (High, Medium, Low) to your conclusion and briefly explain why.
    4.  **Overall Conclusion & Recommendations:** Conclude the report with a summary of patterns and potential recommendations for the artist or marketing team.

    Format the output in Markdown for clarity and readability.
    """

    print("Generating final report with Gemini 1.5 Pro...")
    response = SYNTHESIS_MODEL_OBJECT.generate_content([Part.from_text(prompt)])
    final_report_text = response.text

    with open(FINAL_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_report_text)

    print(f"\nFinal analysis report saved to: {FINAL_REPORT_FILE}")
    return {"status": "success", "output_file": FINAL_REPORT_FILE}

# --- Agent Definitions ---

# This agent corresponds to the "Data Ingestion Agent" and "Anomaly Detection Agent"
data_agent = Agent(
    model=SYNTHESIS_MODEL_NAME,
    name="DataAndAnomalyAgent",
    description="Connects to music engagement data, cleans it, and identifies statistical anomalies.",
    tools=[data_ingestion_and_anomaly_detection_tool],
    instruction="""
    You are the Data and Anomaly Agent. Your job is to process the 'music_engagement_data.csv' file.
    Use the `data_ingestion_and_anomaly_detection_tool` to find dates where a track's views
    in a specific country were statistically significant compared to its recent trend.
    The tool has direct access to the required data file.
    """,
)

# This agent combines the "Hypothesis Generation", "Evidence Gathering", and "Reporting" roles
reporting_agent = Agent(
    model=SYNTHESIS_MODEL_NAME,
    name="InvestigationAndReportingAgent",
    description="Researches anomalies, generates hypotheses, gathers evidence, and creates a final report.",
    tools=[hypothesis_evidence_and_reporting_tool],
    instruction="""
    You are the Investigation and Reporting Agent. Your purpose is to explain *why* engagement spikes occurred.
    Use the `hypothesis_evidence_and_reporting_tool` to take the identified anomalies,
    research their potential causes using web searches, and synthesize all findings into a
    comprehensive, human-readable analysis report.
    """,
)

# This is the "Orchestrator Agent"
root_agent = Agent(
    model=SYNTHESIS_MODEL_NAME,
    name="root_agent",
    description="Manages the entire music analysis workflow from start to finish.",
    flow="sequential",
    children=[data_agent, reporting_agent],
    instruction="""
    You are the Orchestrator Agent. Your goal is to generate a comprehensive analysis of music engagement data.
    1. First, trigger the `DataAndAnomalyAgent` to process the data and find anomalies.
    2. Next, trigger the `InvestigationAndReportingAgent` to research those anomalies and generate the final report.
    """,
)


# --- Main Execution Logic ---
if __name__ == "__main__":
    print("Starting the Music Engagement Analysis Multi-Agent System...")

    # Step 1: Ingest data and detect anomalies.
    # This simulates the Orchestrator telling the first agent to run.
    anomaly_result = data_ingestion_and_anomaly_detection_tool(ToolContext())
    print(f"\nData Agent Result: {json.dumps(anomaly_result, indent=2)}")

    # Step 2: If anomalies were found, proceed to investigate and report.
    # This simulates the Orchestrator telling the second agent to run.
    if anomaly_result.get("status") == "success" and anomaly_result.get("anomalies_found", 0) > 0:
        report_result = hypothesis_evidence_and_reporting_tool(ToolContext())
        print(f"\nReporting Agent Result: {json.dumps(report_result, indent=2)}")
    elif anomaly_result.get("status") == "error":
        print("\nWorkflow stopped due to an error in the data analysis phase.")
    else:
        print("\nWorkflow complete. No anomalies required further investigation.")

    print("\nMulti-agent workflow finished.")