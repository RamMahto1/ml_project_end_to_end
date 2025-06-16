from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Read form data exactly as strings or numbers
            gender = request.form['gender']
            race_ethnicity = request.form['race_ethnicity']
            parental_level_of_education = request.form['parental_level_of_education']
            lunch = request.form['lunch']
            test_preparation_course = request.form['test_preparation_course']
            reading_score = int(request.form['reading_score'])
            writing_score = int(request.form['writing_score'])

            # Create data object
            input_data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score,
            )
            df = input_data.get_data_as_data_frame()

            # Predict
            pipeline = PredictPipeline()
            prediction = pipeline.predict(df)[0]

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)


if __name__ == '__main__':
    app.run(debug=True)
