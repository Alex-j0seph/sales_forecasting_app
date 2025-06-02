import os
import pandas as pd
import requests
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prophet import Prophet
from dotenv import load_dotenv
import logging
import io
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# --- Configuration & Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Sales Forecasting API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# --- (Reference Initial Data Loading - not critical for user forecast) ---
try:
    df_initial_ref = pd.read_csv("sales_data.csv")
    df_initial_ref.columns = df_initial_ref.columns.str.strip()
    if 'Date' not in df_initial_ref.columns or 'Sales_Revenue' not in df_initial_ref.columns:
        logger.warning("Initial sales_data.csv missing 'Date' or 'Sales_Revenue' columns.")
    else:
        df_initial_ref['ds'] = pd.to_datetime(df_initial_ref['Date'], errors='coerce')
        df_initial_ref['y'] = pd.to_numeric(df_initial_ref['Sales_Revenue'], errors='coerce')
        df_initial_ref.dropna(subset=['ds', 'y'], inplace=True)
        if len(df_initial_ref) < 2:
            logger.warning("Initial sales_data.csv has insufficient data (reference model).")
        else:
            logger.info("Initial sales_data.csv loaded for reference.")
except FileNotFoundError:
    logger.info("Note: sales_data.csv not found. No reference data available.")
except ValueError as ve:
    logger.error(f"ValueError in initial data loading (sales_data.csv): {ve}")
except Exception as e:
    logger.error(f"Error during initial data loading (sales_data.csv): {e}")

# --- Pydantic Models for Chat Request ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/result", response_class=HTMLResponse)
async def generate_forecast_and_insights(
    request: Request,
    user_dates: Optional[str] = Form(None),
    user_sales: Optional[str] = Form(None),
    csv_file: Optional[UploadFile] = File(None),
    display_aggregation: str = Form("daily_smoothed")
):
    ai_insight_text = "Could not retrieve AI insights or an error occurred."
    error_message = None
    raw_user_df = None
    actual_sales_for_chart = []
    future_predictions_for_chart = []

    MAX_ACTUAL_POINTS_TO_DISPLAY_DAILY = 90
    MAX_ACTUAL_POINTS_TO_DISPLAY_WEEKLY = 52  # Approx 1 year
    MAX_ACTUAL_POINTS_TO_DISPLAY_MONTHLY = 24 # Approx 2 years
    SMA_WINDOW = 7

    logger.info(f"Display aggregation selected: {display_aggregation}")
    logger.info(f"Received CSV file: {csv_file.filename if csv_file else 'No file'}")
    logger.info(f"Received raw user_dates: '{user_dates}'")
    logger.info(f"Received raw user_sales: '{user_sales}'")

    try:
        # Determine input method and process data to create raw_user_df
        if csv_file and csv_file.filename:
            logger.info(f"Processing uploaded CSV file: {csv_file.filename}")
            if not csv_file.filename.endswith('.csv'):
                error_message = "Invalid file type. Please upload a .csv file."
            else:
                try:
                    contents = await csv_file.read()
                    df_from_csv = pd.read_csv(io.StringIO(contents.decode('utf-8')))
                    df_from_csv.columns = df_from_csv.columns.str.strip()
                    date_col, revenue_col = None, None
                    possible_date_cols = ['Date', 'date', 'DS', 'ds']
                    possible_revenue_cols = ['Sales_Revenue', 'Sales Revenue', 'Revenue', 'Sales', 'Y', 'y', 'Value']
                    for col in possible_date_cols:
                        if col in df_from_csv.columns: date_col = col; break
                    for col in possible_revenue_cols:
                        if col in df_from_csv.columns: revenue_col = col; break
                    if not date_col or not revenue_col:
                        error_message = "CSV must contain a recognizable date column (e.g., 'Date') and a sales revenue column (e.g., 'Sales_Revenue')."
                    else:
                        raw_user_df = pd.DataFrame()
                        raw_user_df['ds'] = pd.to_datetime(df_from_csv[date_col], errors='coerce')
                        raw_user_df['y'] = pd.to_numeric(df_from_csv[revenue_col], errors='coerce')
                        raw_user_df.dropna(subset=['ds', 'y'], inplace=True)
                        if len(raw_user_df) == 0: error_message = "CSV file contained no valid data rows after processing."
                except Exception as e_csv:
                    error_message = f"Error processing CSV file: {e_csv}"
                    logger.error(f"CSV error: {e_csv}", exc_info=True)

        elif user_dates and user_sales:
            logger.info("Processing data from text areas.")
            dates_list_str = [d.strip() for d in user_dates.split(',') if d.strip()]
            sales_list_str = [s.strip() for s in user_sales.split(',') if s.strip()]
            if not dates_list_str or not sales_list_str:
                error_message = "Manual input: Dates or Sales Revenue data is empty."
            elif len(dates_list_str) != len(sales_list_str):
                error_message = "Manual input: Mismatch between number of dates and sales entries."
            else:
                raw_user_df = pd.DataFrame({'ds': pd.to_datetime(dates_list_str, errors='coerce'), 'y': pd.to_numeric(sales_list_str, errors='coerce')})
                raw_user_df.dropna(subset=['ds', 'y'], inplace=True)
        else:
            error_message = "No sales data provided. Please upload a CSV or fill in the text areas."

        # Proceed if raw_user_df is valid
        if raw_user_df is not None and len(raw_user_df) >= 2 and not error_message:
            raw_user_df = raw_user_df.sort_values(by='ds').reset_index(drop=True)
            logger.info(f"Full raw daily user_df for Prophet model fitting has {len(raw_user_df)} rows.")

            current_request_model = Prophet()
            current_request_model.fit(raw_user_df[['ds', 'y']])

            future_df_for_prophet = current_request_model.make_future_dataframe(periods=91, include_history=True)
            daily_forecast_output_all = current_request_model.predict(future_df_for_prophet)

            # Aggregate ACTUAL data for chart
            df_actual_for_chart_processed = raw_user_df.copy()
            current_max_actual_points = MAX_ACTUAL_POINTS_TO_DISPLAY_DAILY

            if display_aggregation == "daily_smoothed":
                if len(df_actual_for_chart_processed['y']) >= SMA_WINDOW:
                    df_actual_for_chart_processed['y_display'] = df_actual_for_chart_processed['y'].rolling(window=SMA_WINDOW, min_periods=1).mean()
                else:
                    df_actual_for_chart_processed['y_display'] = df_actual_for_chart_processed['y']
                if len(df_actual_for_chart_processed) > current_max_actual_points:
                    df_actual_for_chart_processed = df_actual_for_chart_processed.tail(current_max_actual_points)
                df_actual_for_chart_processed['ds_str'] = df_actual_for_chart_processed['ds'].dt.strftime('%Y-%m-%d')
                actual_sales_for_chart = df_actual_for_chart_processed[['ds_str', 'y_display']].rename(columns={'ds_str':'ds', 'y_display':'y'}).to_dict(orient='records')

            elif display_aggregation == "weekly_sum":
                current_max_actual_points = MAX_ACTUAL_POINTS_TO_DISPLAY_WEEKLY
                df_actual_weekly = df_actual_for_chart_processed.set_index('ds').resample('W-MON', label='left', closed='left')['y'].sum().reset_index()
                df_actual_weekly.rename(columns={'y': 'y_display'}, inplace=True)
                if len(df_actual_weekly) > current_max_actual_points:
                    df_actual_weekly = df_actual_weekly.tail(current_max_actual_points)
                df_actual_weekly['ds_str'] = df_actual_weekly['ds'].dt.strftime('%Y-%m-%d')
                actual_sales_for_chart = df_actual_weekly[['ds_str', 'y_display']].rename(columns={'ds_str':'ds', 'y_display':'y'}).to_dict(orient='records')

            elif display_aggregation == "monthly_sum":
                current_max_actual_points = MAX_ACTUAL_POINTS_TO_DISPLAY_MONTHLY
                df_actual_monthly = df_actual_for_chart_processed.set_index('ds').resample('MS', label='left', closed='left')['y'].sum().reset_index()
                df_actual_monthly.rename(columns={'y': 'y_display'}, inplace=True)
                if len(df_actual_monthly) > current_max_actual_points:
                    df_actual_monthly = df_actual_monthly.tail(current_max_actual_points)
                df_actual_monthly['ds_str'] = df_actual_monthly['ds'].dt.strftime('%Y-%m-%d')
                actual_sales_for_chart = df_actual_monthly[['ds_str', 'y_display']].rename(columns={'ds_str':'ds', 'y_display':'y'}).to_dict(orient='records')

            # Aggregate FUTURE predictions for chart
            last_raw_actual_date = raw_user_df['ds'].max()
            # Predictions start from the last actual date to connect lines
            daily_predictions_to_aggregate = daily_forecast_output_all[daily_forecast_output_all['ds'] >= last_raw_actual_date].copy()
            
            logger.debug(f"Raw daily_predictions_to_aggregate (first 10 for future period):\n{daily_predictions_to_aggregate[['ds', 'yhat']].head(10).to_string()}")

            if display_aggregation == "daily_smoothed":
                daily_predictions_to_aggregate['ds_str'] = daily_predictions_to_aggregate['ds'].dt.strftime('%Y-%m-%d')
                future_predictions_for_chart = daily_predictions_to_aggregate[['ds_str', 'yhat']].rename(columns={'ds_str':'ds'}).to_dict(orient='records')
            
            elif display_aggregation == "weekly_sum":
                df_future_weekly = daily_predictions_to_aggregate.set_index('ds').resample('W-MON', label='left', closed='left')['yhat'].sum().reset_index()
                df_future_weekly['ds_str'] = df_future_weekly['ds'].dt.strftime('%Y-%m-%d')
                future_predictions_for_chart = df_future_weekly[['ds_str', 'yhat']].rename(columns={'ds_str':'ds'}).to_dict(orient='records')

            elif display_aggregation == "monthly_sum":
                df_future_monthly = daily_predictions_to_aggregate.set_index('ds').resample('MS', label='left', closed='left')['yhat'].sum().reset_index()
                df_future_monthly['ds_str'] = df_future_monthly['ds'].dt.strftime('%Y-%m-%d')
                future_predictions_for_chart = df_future_monthly[['ds_str', 'yhat']].rename(columns={'ds_str':'ds'}).to_dict(orient='records')

            logger.info(f"Actuals for chart ({display_aggregation}): {len(actual_sales_for_chart)} points. Future preds ({display_aggregation}): {len(future_predictions_for_chart)} points.")
            logger.debug(f"Aggregated future_predictions_for_chart (first 5 points): {future_predictions_for_chart[:5]}")


            # AI Insight Data Preparation
            ai_prompt_forecast_data = daily_forecast_output_all[['ds', 'yhat']].tail(30).copy()
            ai_prompt_forecast_data['ds'] = ai_prompt_forecast_data['ds'].dt.strftime('%Y-%m-%d')
            ai_prompt_forecast_data_dict = ai_prompt_forecast_data.to_dict(orient='records')

            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                ai_insight_text = "DeepSeek API key not configured."
            else:
                insight_prompt_data_for_ai = ai_prompt_forecast_data_dict[-10:]
                prompt = (
                    f"Analyze the following sales forecast data. "
                    f"Provide 3 actionable business insights. Focus on trends, opportunities, or potential risks. "
                    f"Forecast data (date, predicted sales): {insight_prompt_data_for_ai}"
                )
                try:
                    response = requests.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={"Authorization": f"Bearer {deepseek_api_key}", "Content-Type": "application/json"},
                        json={"model": "deepseek-chat", "messages": [{"role": "system", "content": "You are a helpful business analyst."}, {"role": "user", "content": prompt}], "max_tokens": 350, "temperature": 0.7 }
                    )
                    response.raise_for_status()
                    api_response_json = response.json()
                    if api_response_json.get("choices") and api_response_json["choices"][0].get("message"):
                        ai_insight_text = api_response_json["choices"][0]["message"]["content"].strip()
                    else: ai_insight_text = "Unexpected response format from DeepSeek."
                except requests.exceptions.RequestException as req_e: ai_insight_text = f"Error with DeepSeek API: {req_e}"
                except Exception as e_deepseek: ai_insight_text = f"Error processing DeepSeek response: {e_deepseek}"

        elif not error_message:
            if raw_user_df is None: error_message = "No valid sales data could be processed."
            else: error_message = "Insufficient valid data (at least 2 data points required for forecasting)."
            logger.warning(f"Data validation failed: {error_message}")

    except ValueError as ve:
        error_message = f"Invalid data format: {ve}."
        logger.error(f"ValueError: {ve}", exc_info=True)
    except Exception as e_main:
        logger.error(f"Unexpected error in /result: {e_main}", exc_info=True)
        error_message = "An unexpected internal error occurred. Please check logs."

    template_forecast_context = {
        "actual": actual_sales_for_chart,
        "future": future_predictions_for_chart,
        "aggregation": display_aggregation
    }
    if error_message:
        return templates.TemplateResponse("result.html", {
            "request": request, "actual_sales_for_chart": [], "future_predictions_for_chart": [],
            "insight": error_message, "error_message": error_message,
            "forecast": None, "display_aggregation": display_aggregation
        })
    return templates.TemplateResponse("result.html", {
        "request": request,
        "actual_sales_for_chart": actual_sales_for_chart,
        "future_predictions_for_chart": future_predictions_for_chart,
        "insight": ai_insight_text,
        "error_message": None,
        "forecast": template_forecast_context,
        "display_aggregation": display_aggregation
    })

@app.post("/chat_with_ai")
async def chat_with_ai_endpoint(chat_request: ChatRequest):
    logger.info(f"Received /chat_with_ai request with {len(chat_request.messages)} messages.")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        logger.error("DeepSeek API key not found for chat endpoint.")
        return HTMLResponse(content='{"ai_response": "Error: AI service is not configured (missing API key)."}', media_type="application/json", status_code=400)
    try:
        api_payload = {
            "model": "deepseek-chat",
            "messages": [message.model_dump() for message in chat_request.messages],
            "max_tokens": 300,
            "temperature": 0.7
        }
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {deepseek_api_key}", "Content-Type": "application/json"},
            json=api_payload
        )
        response.raise_for_status()
        api_response_json = response.json()
        if api_response_json.get("choices") and api_response_json["choices"][0].get("message"):
            ai_message_content = api_response_json["choices"][0]["message"]["content"].strip()
            return {"ai_response": ai_message_content}
        else:
            return HTMLResponse(content='{"ai_response": "Error: Received an unexpected response from the AI service."}', media_type="application/json", status_code=500)
    except requests.exceptions.HTTPError as http_err:
        error_text = http_err.response.text if http_err.response else "No response body"
        logger.error(f"HTTP error: {http_err} - Response: {error_text}")
        return HTMLResponse(content=f'{{"ai_response": "Error: Could not communicate with AI service (HTTP {http_err.response.status_code})."}}', media_type="application/json", status_code=http_err.response.status_code)
    except requests.exceptions.RequestException as req_e:
        logger.error(f"Request error: {req_e}")
        return HTMLResponse(content='{"ai_response": "Error: A problem occurred while trying to reach the AI service."}', media_type="application/json", status_code=503)
    except Exception as e:
        logger.error(f"Unexpected error in /chat_with_ai: {e}", exc_info=True)
        return HTMLResponse(content='{"ai_response": "Error: An unexpected internal error occurred."}', media_type="application/json", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)