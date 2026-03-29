import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


def generate_all_charts(df):

    # 1. SoC Distribution
    plt.figure()
    sns.violinplot(y=df["battery_soc_pct"])
    plt.title("SoC Distribution")
    plt.savefig("outputs/charts/soc_distribution.png")

    # 2. Temp vs SoC
    plt.figure()
    sns.scatterplot(
        x="battery_soc_pct",
        y="battery_temp_c",
        hue="battery_state",
        data=df
    )
    plt.savefig("outputs/charts/temp_vs_soc.png")

    # 3. SoH Time Series (sample devices)
    plt.figure()
    sample = df["device_id"].dropna().unique()[:3]
    for d in sample:
        subset = df[df["device_id"] == d]
        plt.plot(subset["ts"], subset["battery_soh_pct"], label=d)
    plt.legend()
    plt.savefig("outputs/charts/soh_timeseries.png")

    # 4. GPS Heatmap
    m = folium.Map(location=[20, 78], zoom_start=5)
    heat_data = df[["gps_lat", "gps_lon"]].dropna().values.tolist()
    HeatMap(heat_data).add_to(m)
    m.save("outputs/charts/gps_heatmap.html")

    # 5. Correlation Heatmap
    plt.figure()
    sns.heatmap(df.corr(numeric_only=True))
    plt.savefig("outputs/charts/correlation_heatmap.png")

    # 6. Timeline (basic version)
    plt.figure()
    df["device_status"].value_counts().plot(kind="bar")
    plt.savefig("outputs/charts/state_timeline.png")