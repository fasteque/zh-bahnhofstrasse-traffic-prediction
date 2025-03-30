import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zurich Foot Traffic", layout="centered")
st.markdown("<h2 style='text-align: center;'>🚶‍♀️ Zurich Bahnhofstrasse</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Pedestrian Traffic Dashboard</h2>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/foot_traffic.csv", parse_dates=["timestamp"])

    # 🚫 Drop rows where collection_type is null
    df = df[df["collection_type"].notna()]

    # 🚫 Drop Lintheschergasse — focus only on Bahnhofstrasse areas
    df = df[df["location_name"].str.contains("Bahnhofstrasse")]

    # 🚫 Drop all 'zone_99_*' columns
    df = df.drop(columns=[col for col in df.columns if "zone_99" in col])

    # 🧼 Drop other columns with nulls (after zone_99 + collection_type)
    df = df.dropna()

    # 🧠 Feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"] >= 5
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    return df

df = load_data()

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
