from flask import Flask, render_template, request, redirect, url_for
import csv

app = Flask(__name__)

# Define the path to the CSV file
csv_file = 'data.csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        city = request.form['city']
        phone = request.form['phone']

        # Append the data to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, age, city, phone])

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
