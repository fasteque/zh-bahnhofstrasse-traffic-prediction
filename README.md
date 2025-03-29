# 🚶‍♀️ Zurich Bahnhofstrasse Foot Traffic Prediction

This project analyzes and predicts pedestrian traffic on Zurich's Bahnhofstrasse using open data from opendata.swiss.  
The goal is to explore traffic trends and apply machine learning to forecast footfall — useful for urban planners, retailers, and data enthusiasts alike.

## 📦 Project Structure

```
foot-traffic-predictor/
├── notebooks/
│   └── zurich_bahnhofstrasse.ipynb   # Main analysis and model notebook
├── .gitignore
└── README.md
```

> ⚠️ Note: The dataset is not included in the repository due to size. See below for download instructions.

## 📊 Data Source

**Dataset**: [Passantenfrequenzen an der Bahnhofstrasse (Stundenwerte)](https://opendata.swiss/en/dataset/passantenfrequenzen-an-der-bahnhofstrasse-stundenwerte)

- Laser-based pedestrian counters along Bahnhofstrasse  
- Hourly resolution  
- Split by:  
  - Direction (to Bürkliplatz / to Hauptbahnhof)  
  - Zones and sidewalk sides  
  - Adult / child  
- Weather and temperature included  

## 🔧 What This Project Does

- Data cleaning and filtering (excluding non-Bahnhofstrasse zones, zone_99, nulls)  
- Feature engineering:
  - Hour, weekday, weekend, month, year
  - Lag features (previous hour, same hour previous day)
  - Weather and location one-hot encoding  
- Trained Random Forest Regressor  
- Log-transformed target for better accuracy  
- Visualized predictions vs. actual traffic  

## 📈 Results

- **Model**: Random Forest Regressor  
- **Mean Absolute Error (MAE)**: ~175  
- Baseline MAE was ~408 → huge improvement using engineered features  
- Visual plots show strong correlation with true values at most traffic levels  

## 💡 Use Cases

- Help retailers adjust staffing or promotions based on time/day  
- Support urban planning with traffic insights  
- Extend into GenAI summaries or predictive dashboards  

## ▶️ Getting Started

1. Clone the repo

```bash
git clone https://github.com/fasteque/zh-bahnhofstrasse-traffic-prediction.git
cd zh-bahnhofstrasse-traffic-prediction
```

2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install requirements (coming later if needed)

```bash
pip install -r requirements.txt
```

4. Download the dataset manually from [here](https://opendata.swiss/en/dataset/passantenfrequenzen-an-der-bahnhofstrasse-stundenwerte)  
   and place it inside a `/data/` folder

## 📜 License

MIT License — feel free to use, fork, and build on this project.

## ✨ Author

Made by [Daniele Altomare](https://github.com/fasteque) with coffee, code, and public data from Zurich ☕🇨🇭