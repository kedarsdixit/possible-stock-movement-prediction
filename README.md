# Stock Market Movement Prediction

---

- The Web App aims at profitable user investment by assisting the users in analyzing current as well as past stockmarket trends using LSTM model and Technical Indicators.
- It predicts next five days closing share prices using LSTM.
- Visualizing past 3(½) years data.
- Performs Technical Analysis on company data and visualizing it through Candlestick Graph.<br><br>

### Technical Indicators

---

- Moving average (MA)
- Relative strength index (RSI)
- Moving average convergence divergence (MACD)
- Stochastic Oscillator (SO)
- Bollinger Bands (BB)
  <br><br>

### Technology Stack

---

- The revelant company data for past 3(½) years is fetched using Yahoo Finance API
- Flask (Back end) is a micro web framework written in Python which is integrated with React (Front end) a javaScript library is used for building user interfaces or UI components
  <br><br>

### Setup

---

1. Install all needed dependencies from requirements.txt

```
pip install -r requirements.txt
```

2. In the ui folder, run the following commands

```
npm install -g serve
npm run build
serve -s build -l 3000
```

3. In order to generate the LSTM models, execute below command
   This demonstrates the command for generating the Model for Wipro Company

```
python .\Wipromodel_generator.py

```

4. To run the Flask server

```
cd service
set FLASK_APP=app.py
flask run
```

One can now go to localhost:3000 to see that the UI is up and running along with Flask.
<br><br>

### Contributors

---

This project has been developed as a part of Final year B.Tech project (2019-2020) sponsored by _Persistent Systems_, by the students of Cummins College of Engineering, Pune.
The contributors to the project and this repository are :

- Shruti Nehe (`shrutibnehe`)<br>
- Swarnika Maurya (`SwarnikaMaurya`)<br>
- Madhura Kurhadkar (`madhuraKurhadkar`)<br>
- Aishwarya Wawdhane (`waishwarya`)<br>
