from flask import Flask,render_template,request
import requests

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city_name = request.form.get('city')

        r = requests.get('https://api.openweathermap.org/data/2.5/weather?q='+city_name+'&appid=cbf1b9f6f3127d65b15aa57c0cd3d28a')

        json_object = r.json()
 
        temperature = int(json_object['main']['temp']-273.15) #this temparetuure in kelvin
        humidity = int(json_object['main']['humidity'])
        pressure = int(json_object['main']['pressure'])
        wind = int(json_object['wind']['speed'])

        condition = json_object['weather'][0]['main']
        desc = json_object['weather'][0]['description']
        
        return render_template('index.html',temperature=temperature,pressure=pressure,humidity=humidity,city_name=city_name,condition=condition,wind=wind,desc=desc)
    else:
        return render_template('index.html') 


if __name__ == '__main__':
    app.run(debug=True)