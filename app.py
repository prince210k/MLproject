from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # Validate form data
            reading_score = float(request.form.get('reading_score'))
            writing_score = float(request.form.get('writing_score'))
            
            if not (0 <= reading_score <= 100) or not (0 <= writing_score <= 100):
                raise ValueError("Scores must be between 0 and 100")
                
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            pred_df = data.get_data_as_dataframe()
            print("Input Data:\n", pred_df)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('index.html', results=round(results[0], 2))
            
        except ValueError as ve:
            error_message = f"Invalid input: {str(ve)}"
            return render_template('index.html', error=error_message)
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)