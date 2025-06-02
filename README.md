# AI-Powered Sales Forecasting Web Application

## Description

This project is an AI-powered web application designed to forecast future sales trends and provide actionable business insights. Users can input historical sales data either manually or by uploading a CSV file. The application then uses Facebook's Prophet model for time-series forecasting and integrates with the DeepSeek AI API to generate business insights from the forecast and offer an interactive chat experience for further exploration. The frontend is built with HTML, CSS, and JavaScript (using Chart.js for visualizations), and the backend is powered by Python with the FastAPI framework.

## Key Features

* **Flexible Data Input:** Supports sales data input via CSV file upload or manual text entry.
* **Prophet Forecasting:** Utilizes Facebook Prophet to generate sales forecasts (currently for a 91-day horizon).
* **Customizable Chart Display:**
    * Visualizes actual sales (historical) and predicted future sales on a single interactive chart.
    * Option to display actual sales as:
        * Daily (smoothed with a 7-day Simple Moving Average)
        * Weekly Totals
        * Monthly Totals
    * Predicted future sales are aggregated to match the selected granularity of actuals.
    * Clear color differentiation and connected lines for easy interpretation.
* **AI-Generated Business Insights:** Integrates with DeepSeek AI to provide initial textual insights based on the forecast data.
* **Interactive AI Chat:** Allows users to ask follow-up questions about the forecast and insights directly on the results page.
* **Web-Based Interface:** Accessible via a web browser, built with FastAPI and HTML/CSS/JavaScript.

## Technology Stack

* **Backend:** Python 3.x, FastAPI, Uvicorn
* **Forecasting & Data Handling:** Prophet, Pandas
* **AI Integration:** DeepSeek API (via `requests`)
* **Frontend:** HTML5, CSS3, JavaScript
* **Charting:** Chart.js (with `chartjs-adapter-date-fns`)
* **Environment Management:** `python-dotenv`
* **Templating:** Jinja2

## Screenshots

*(Placeholder: Add a screenshot of the Index Page here. e.g., `![Index Page](./screenshots/index_page.png)`)*
**Index Page:** Allows data input via CSV or text, and selection of display aggregation.

*(Placeholder: Add a screenshot of the Results Page here - perhaps showing a monthly aggregated view. e.g., `![Results Page](./screenshots/results_page_monthly.png)`)*
**Results Page:** Displays the forecast chart, AI-generated insights, and the interactive AI chat.

## Setup and Installation

Follow these steps to set up and run the application locally:

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* A DeepSeek API Key (obtainable from [https://platform.deepseek.com/](https://platform.deepseek.com/))

### Installation Steps

1.  **Clone the Repository (Conceptual):**
    If this project were on GitHub, you would clone it. For now, ensure all project files are in a local directory.
    ```bash
    # git clone <repository-url>
    # cd <project-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    * macOS/Linux: `source venv/bin/activate`
    * Windows: `venv\Scripts\activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a file named `.env` in the root project directory and add your DeepSeek API key:
    ```env
    DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here
    ```

5.  **Prepare Initial Data (Optional Reference):**
    The `sales_data.csv` file can be present in the root directory for the application to load as an initial reference (though user-provided data is used for each specific forecast). Ensure it has `Date` and `Sales_Revenue` columns.

## Running the Application

1.  Ensure your virtual environment is activated and you are in the project's root directory.
2.  Start the FastAPI application using Uvicorn:
    ```bash
    uvicorn sales_forecasting_api:app --reload --port 8000
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:8000`.

## Usage Guide

### 1. Data Input (Index Page)

* Navigate to the application's home page.
* **Choose Data Input Method:**
    * **Option 1: CSV Upload:** Click "Select CSV File" and choose your `.csv` file. The CSV should have a header row and columns for dates (e.g., "Date") and sales figures (e.g., "Sales\_Revenue").
    * **Option 2: Manual Text Input:**
        * Enter dates (format `YYYY-MM-DD`, comma-separated) in the "Sales Dates" text area.
        * Enter corresponding sales figures (numeric, comma-separated) in the "Sales Revenue" text area.
        * Ensure the number of dates matches the number of sales entries.
* **Select Chart Display Option:**
    * Use the "Display Actual Sales As" dropdown to choose how historical data appears on the chart:
        * `Daily (Smoothed)`: Daily data with a 7-day moving average.
        * `Weekly Totals`: Aggregated weekly sums.
        * `Monthly Totals`: Aggregated monthly sums.
* Click "Get Forecast & Insights".

### 2. Viewing Results (Results Page)

* **Forecast Chart:** Displays your "Actual Sales" (processed according to your display choice) and "Predicted Future Sales" (aggregated to match actuals' granularity). Hover over lines for details.
* **AI-Generated Business Insights:** Read the initial analysis and suggestions provided by the AI based on your forecast.
* **Continue Chat with AI:** Type follow-up questions about the forecast or insights into the chat input box and click "Send" to get further clarification or analysis from the AI.

## Project File Structure
