from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model (adjust the file path as needed)
model = joblib.load(r"C:\Users\shovi\OneDrive\Desktop\dps\model_pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    year = data['JAHR']-2000
    month = data['MONAT']
    
    # Assuming your model expects input as a 2D array (e.g., [[year, month]])
    list1 = [0,1,year, month]
    # convert the list into dataframe row
    data = pd.DataFrame(list1).T
    
    # add columns
    data.columns = ['MONATSZAHL	', 'AUSPRAEGUNG',
                    'JAHR', 'MONAT']

    
    # Get the prediction from the model
    prediction = model.predict(data)

    prediction_list = prediction.tolist()
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction_list[0]})

if __name__ == '__main__':
    app.run(debug=True)
