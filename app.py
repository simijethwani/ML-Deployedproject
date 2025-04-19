from flask import Flask, render_template, request
import pickle
import numpy as np
#from pyngrok import ngrok

app = Flask(__name__, template_folder='templates')

# Load models
model_addiction = pickle.load(open('addiction.pkl', 'rb'))
model_satisfaction = pickle.load(open('satisfaction.pkl', 'rb'))
model_productivity = pickle.load(open('productivity.pkl', 'rb'))

# Feature sets
features_map = {
    'addiction': ['Self Control', 'Watch Time', 'Frequency', 'Number of Videos Watched', 'Age', 'Total Time Spent'],
    'satisfaction': ['Self Control', 'Watch Time', 'Frequency', 'Income', 'Scroll Rate', 'Number of Sessions', 'Age'],
    'productivity': ['Satisfaction', 'Addiction Level', 'Watch Time', 'Frequency']
}
feature_descriptions = {
    'Self Control': 'Self Control – How well do you manage your screen time? (0-10)',
    'Watch Time': 'Watch Time – Average time spent watching per session (in minutes)',
    'Frequency': 'Frequency – How often do you use the platform daily? (0-3)',
    'Number of Videos Watched': 'Number of Videos Watched – Total per day',
    'Age': 'Age – Your current age',
    'Total Time Spent': 'Total Time Spent – Total hours spent per day on the platform',
    'Income': 'Income – Monthly income in USD',
    'Scroll Rate': 'Scroll Rate – How fast do you scroll (e.g., videos/min)',
    'Number of Sessions': 'Number of Sessions – Times you open the app daily',
    'Satisfaction': 'Satisfaction – Your satisfaction level with the content (1-8)',
    'Addiction Level': 'Addiction Level – Perceived level of addiction to social media (0-7)'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['POST'])
def form():
    target = request.form['target']
    features = features_map.get(target, [])
    return render_template('form.html', target=target, features=features, feature_descriptions=feature_descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    target = request.form['target']
    features = features_map.get(target, [])

    try:
        input_data = [float(request.form[feature]) for feature in features]
        input_array = np.array([input_data])

        if target == 'addiction':
            prediction = model_addiction.predict(input_array)[0]
        elif target == 'satisfaction':
            prediction = model_satisfaction.predict(input_array)[0]
        elif target == 'productivity':
            prediction = model_productivity.predict(input_array)[0]
        else:
            prediction = 'Invalid target'

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"
'''
if __name__ == '__main__':
    port = 5000
    public_url = ngrok.connect(port)
    print(f"🌍 Public URL: {public_url}")  # Access your app using this
    app.run(port=port)
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
