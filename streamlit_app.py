import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.set_page_config(page_title="Zurich Foot Traffic", layout="centered")

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This dashboard displays foot traffic data collected on Zurich's Bahnhofstrasse.

**Instructions:**
1. Upload a CSV file.
2. View interactive charts in each tab.
3. (Coming soon) See predictive insights.

Source: [opendata.swiss](https://opendata.swiss)
""")

# Load trained model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load("model/xgb_model.pkl")
    features = joblib.load("model/features.pkl")
    return model, features

model, model_features = load_model()

def clean_data(df):
    df = df[df["collection_type"].notna()]
    df = df[df["location_name"].str.contains("Bahnhofstrasse")]
    df = df.drop(columns=[col for col in df.columns if "zone_99" in col])
    df = df.dropna()
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"] >= 5
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    return df

@st.cache_data
def load_data():
    df = pd.read_csv("data/foot_traffic.csv", parse_dates=["timestamp"])
    df = clean_data(df)
    return df

st.markdown("<h2 style='text-align: center;'>🚶‍♀️ Zurich Bahnhofstrasse</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Pedestrian Traffic Dashboard</h2>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
Welcome to the **Zurich Bahnhofstrasse Foot Traffic Dashboard**!  
Upload your own CSV file to explore pedestrian patterns by hour, weekday, location, and more.  
<br><br>
📥 Get the official dataset here:  
<a href='https://opendata.swiss/en/dataset/passantenfrequenzen-an-der-bahnhofstrasse-stundenwerte' target='_blank'>
Zurich Bahnhofstrasse hourly pedestrian traffic (opendata.swiss)
</a>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    df = clean_data(df)
    st.success(f"✅ Loaded file: {uploaded_file.name}")
else:
    df = None

st.markdown("---")
st.header("🔮 Predict Pedestrian Count")

st.markdown("Use the sliders and selectors below to simulate conditions and get a prediction.")

prediction_mode = st.radio("Choose prediction mode:", ["Use uploaded CSV data", "Manual input (no CSV)"])

if prediction_mode == "Use uploaded CSV data" and df is None:
    st.warning("📂 Please upload a CSV file to use this prediction mode.")

if df is not None and prediction_mode == "Use uploaded CSV data":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⏰ Hourly Traffic",
        "📅 Weekday Traffic",
        "📆 Monthly Trends",
        "📍 Locations",
        "🗓️ Weekend vs Weekday",
        "🕒 Hourly by Location"
    ])

    with tab1:
        st.subheader("⏰ Average Traffic by Hour")
        hourly = df.groupby("hour")["pedestrians_count"].mean()
        fig1, ax1 = plt.subplots()
        ax1.plot(hourly.index, hourly.values)
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Avg Pedestrian Count")
        st.pyplot(fig1)

    with tab2:
        st.subheader("📅 Average Traffic by Weekday")
        weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        weekday = df.groupby("weekday")["pedestrians_count"].mean().rename(index=weekday_map)
        fig2, ax2 = plt.subplots()
        ax2.bar(weekday.index, weekday.values)
        ax2.set_ylabel("Avg Pedestrian Count")
        st.pyplot(fig2)

    with tab3:
        st.subheader("📆 Average Monthly Traffic")
        monthly = df.groupby("month")["pedestrians_count"].mean()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig3, ax3 = plt.subplots()
        ax3.plot(month_names, monthly.values, marker="o")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Avg Pedestrian Count")
        st.pyplot(fig3)

    with tab4:
        st.subheader("📍 Average Traffic by Sensor Location")
        location = df.groupby("location_name")["pedestrians_count"].mean().sort_values()
        fig4, ax4 = plt.subplots()
        ax4.barh(location.index, location.values)
        ax4.set_xlabel("Avg Pedestrian Count")
        st.pyplot(fig4)

    with tab5:
        st.subheader("🗓️ Weekend vs Weekday Traffic")
        weekend_avg = df.groupby("is_weekend")["pedestrians_count"].mean()
        weekend_labels = ["Weekday", "Weekend"]
        fig5, ax5 = plt.subplots()
        ax5.bar(weekend_labels, weekend_avg)
        ax5.set_ylabel("Avg Pedestrian Count")
        st.pyplot(fig5)

    with tab6:
        st.subheader("🕒 Hourly Trend by Sensor Location")
        hourly_loc = df.groupby(["hour", "location_name"])["pedestrians_count"].mean().unstack()
        fig6, ax6 = plt.subplots()
        hourly_loc.plot(ax=ax6)
        ax6.set_xlabel("Hour of Day")
        ax6.set_ylabel("Avg Pedestrian Count")
        ax6.legend(title="Location")
        st.pyplot(fig6)
else:
    st.info("👈 Please upload a CSV file to view the dashboard.")

if prediction_mode == "Manual input (no CSV)" or (prediction_mode == "Use uploaded CSV data" and df is not None):
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            hour = st.slider("Hour of day", 0, 23, 12)
            weekday = st.selectbox("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            temperature = st.slider("Temperature (°C)", -10.0, 40.0, 15.0)
        
        with col2:
            month = st.selectbox("Month", list(range(1, 13)))
            is_weekend = st.radio("Weekend?", ["No", "Yes"])
            location = st.selectbox("Location", sorted(df["location_name"].unique())) if df is not None else st.selectbox("Location", [])
            weather = st.selectbox("Weather", sorted(df["weather_condition"].unique())) if df is not None else st.selectbox("Weather", [])

        submitted = st.form_submit_button("Predict")

        if submitted:
            if prediction_mode == "Use uploaded CSV data":
                prev_hour = df["pedestrians_count"].iloc[-1]
                prev_hour_2 = df["pedestrians_count"].iloc[-2]
                prev_day_hour = df["pedestrians_count"].iloc[-24]
                prev_year_hour = df["pedestrians_count"].iloc[-8760]
                rolling_3h = df["pedestrians_count"].rolling(3).mean().iloc[-1]
                rolling_6h = df["pedestrians_count"].rolling(6).mean().iloc[-1]
                rolling_24h = df["pedestrians_count"].rolling(24).mean().iloc[-1]
            else:
                prev_hour = st.slider("Previous hour count", 0, 10000, 1000)
                prev_hour_2 = st.slider("2 hours ago", 0, 10000, 1000)
                prev_day_hour = st.slider("Same hour yesterday", 0, 10000, 1000)
                prev_year_hour = st.slider("Same hour last year", 0, 10000, 1000)
                rolling_3h = st.slider("3h rolling avg", 0, 10000, 1000)
                rolling_6h = st.slider("6h rolling avg", 0, 10000, 1000)
                rolling_24h = st.slider("24h rolling avg", 0, 10000, 1000)

            input_data = {
                "hour": hour,
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(weekday),
                "is_weekend": 1 if is_weekend == "Yes" else 0,
                "month": month,
                "temperature": temperature,
                "prev_hour_count": prev_hour,
                "prev_hour_count_2": prev_hour_2,
                "prev_day_same_hour": prev_day_hour,
                "prev_year_same_hour": prev_year_hour,
                "rolling_3h": rolling_3h,
                "rolling_6h": rolling_6h,
                "rolling_24h": rolling_24h
            }

            # Add one-hot encoded values
            for col in model_features:
                if col.startswith("weather_condition_"):
                    input_data[col] = 1 if col == f"weather_condition_{weather}" else 0
                elif col.startswith("location_name_"):
                    input_data[col] = 1 if col == f"location_name_{location}" else 0
                elif col not in input_data:
                    input_data[col] = 0  # default

            input_df = pd.DataFrame([input_data])
            log_pred = model.predict(input_df)[0]
            prediction = int(np.expm1(log_pred))  # revert log1p

            st.success(f"📈 Predicted pedestrian count: **{prediction}**")

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>📊 Built by <a href='https://github.com/fasteque' target='_blank'>@fasteque</a> | "
    "<a href='https://github.com/fasteque/zh-bahnhofstrasse-traffic-prediction' target='_blank'>View on GitHub</a></p>",
    unsafe_allow_html=True
)
