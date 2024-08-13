from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask import Flask, request, jsonify
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from pymongo import MongoClient
from pytrends.request import TrendReq
import logging
import pandas as pd
import plotly.express as px
import plotly.io as pio


app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['pandemic_paranoma']
volunteers_collection = db['volunteers']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mission.html')
def mission():
    return render_template('mission.html')

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')


@app.route('/volunteer.html')
def volunteer():
    return render_template('volunteer.html')

@app.route('/resource.html')
def resource():
    return render_template('resource.html')

@app.route('/register-volunteer', methods=['POST'])
def register_volunteer():
    data = request.json
    name = data.get('name')
    age = data.get('age')
    locality = data.get('locality')
    phone = data.get('phone')

    if name and age and locality and phone:
        # Insert data into MongoDB
        volunteers_collection.insert_one({
            'name': name,
            'age': age,
            'locality': locality,
            'phone': phone
        })
        return {'status': 'success', 'message': 'Volunteer registered successfully'}, 200
    else:
        return {'status': 'error', 'message': 'Missing data'}, 400

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Function to fetch Google Trends data by region
def fetch_google_trends_data(keyword, geo='US', timeframe='now 7-d'):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], geo=geo, timeframe=timeframe)
    data = pytrends.interest_by_region()
    data.reset_index(inplace=True)
    return data

# Function to fetch Google Trends data over time
def fetch_trends_over_time(keywords, geo='US', timeframe='today 5-y'):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(keywords, geo=geo, timeframe=timeframe)
    data = pytrends.interest_over_time()
    data.reset_index(inplace=True)
    return data

# Function to fetch comparison data
def fetch_comparison_data(keyword, geo='US'):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], geo=geo, timeframe='today 5-y')
    data = pytrends.interest_over_time()

    # Log the data for debugging
    logging.debug("Data fetched for comparison:")
    logging.debug(data)

    # Reset index to ensure 'date' is a column
    if not data.empty:
        data.reset_index(inplace=True)

    # Check if the data is not empty and contains the 'date' column
    if data.empty or 'date' not in data.columns:
        logging.debug("No valid data returned or 'date' column is missing.")
        return pd.DataFrame()

    # Extract year and month for comparison
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.strftime('%B')
    data_monthly = data.groupby(['year', 'month']).mean().reset_index()

    return data_monthly

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(
        inputs,
        max_length=1000,
        num_return_sequences=1,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def get_response(text):
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment = sentiment_result['label']
    if sentiment == 'NEGATIVE':
        prompt = (
            f"Someone is expressing feelings of deep distress or hopelessness. "
            f"Offer a personalized, caring, and supportive response to this statement: '{text}'"
            "\nResponse:"
        )
    else:
        prompt = (
            f"Someone is expressing positive feelings. "
            f"Offer a personalized, caring, and encouraging response to this statement: '{text}'"
            "\nResponse:"
        )
    response = generate_response(prompt)
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_text = data.get('text')
    if user_text:
        response = get_response(user_text)
        return jsonify({'response': response})
    return jsonify({'response': 'Sorry, something went wrong.'}), 400

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard_view():
    graph_html = None
    line_chart_html = None
    slider_chart_html = None
    comparison_html = None

    if request.method == 'POST':
        keyword = request.form['keyword']
        line_keywords = [keyword]

        # Fetch data from Google Trends for bar chart
        data = fetch_google_trends_data(keyword)

        # Plotly bar chart
        fig = px.bar(data, x='geoName', y=keyword, title=f'{keyword} Popularity by State')
        graph_html = pio.to_html(fig, full_html=False)

        # Fetch data for line chart
        line_data = fetch_trends_over_time(line_keywords)

        # Plotly line chart
        fig_line = px.line(line_data, x='date', y=line_keywords, title=f'Trend Over Time for {", ".join(line_keywords)}')
        line_chart_html = pio.to_html(fig_line, full_html=False)

        # Replace geographical heatmap with time slider line chart
        fig_slider = px.line(
            line_data,
            x='date',
            y=keyword,
            title=f'Trend Over Time for {keyword} with Time Slider',
            labels={'date': 'Date', keyword: 'Search Interest'},
        )
        fig_slider.update_xaxes(rangeslider_visible=True)
        fig_slider.update_layout(transition_duration=500)
        slider_chart_html = pio.to_html(fig_slider, full_html=False)

        # Fetch data for comparison chart
        comparison_data = fetch_comparison_data(keyword)

        if not comparison_data.empty:
            # Plotly comparison bar chart (Yearly)
            fig_comparison = px.bar(
                comparison_data, 
                x='month', 
                y=keyword, 
                color='year', 
                barmode='group', 
                title=f'Monthly Comparison of {keyword} Over the Years'
            )
            comparison_html = pio.to_html(fig_comparison, full_html=False)

    return render_template(
        'dashboard.html',
        graph_html=graph_html,
        line_chart_html=line_chart_html,
        slider_chart_html=slider_chart_html,
        comparison_html=comparison_html
    )


if __name__ == '__main__':
    app.run(debug=True, port = 5001)
