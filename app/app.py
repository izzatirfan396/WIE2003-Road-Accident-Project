from pathlib import Path

import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Cleaned_Road_Safety_Data_Fixed.xlsx"

PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#d62728"
SUCCESS_COLOR = "#2ca02c"
WARNING_COLOR = "#ff7f0e"


st.set_page_config(
    page_title="Malaysian Road Accident Dashboard",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] {
            color: #475569;
        }
        .section-note {
            color: #475569;
            font-size: 0.96rem;
            line-height: 1.55;
        }
        .small-caption {
            color: #64748b;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df.sort_values("Year").reset_index(drop=True)


def style_axis(ax, title: str, xlabel: str = "Year", ylabel: str = ""):
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def show_metric_row(data: pd.DataFrame):
    latest_row = data.sort_values("Year").iloc[-1]
    first_row = data.sort_values("Year").iloc[0]
    death_change = latest_row["Road Deaths"] - first_row["Road Deaths"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Years Selected", f"{int(data['Year'].min())}-{int(data['Year'].max())}")
    col2.metric("Latest Road Deaths", f"{latest_row['Road Deaths']:,.0f}")
    col3.metric("Change in Deaths", f"{death_change:,.0f}")
    col4.metric("Latest Fatality Rate", f"{latest_row['Fatality_Rate_per_100k']:.2f}")


def regression_metrics(y_actual, y_pred):
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred) if len(y_actual) >= 2 else None
    correlation = pd.Series(y_actual).reset_index(drop=True).corr(pd.Series(y_pred))
    return mae, r2, correlation


def display_model_metrics(mae: float, r2, correlation):
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("R-squared Score", "N/A" if r2 is None else f"{r2:.3f}")
    col3.metric("Correlation", "N/A" if pd.isna(correlation) else f"{correlation:.3f}")


def fit_linear_model(data: pd.DataFrame, features: list, target: str):
    X = data[features]
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mae, r2, correlation = regression_metrics(y, predictions)
    return model, predictions, mae, r2, correlation


def forecast_with_year_model(data: pd.DataFrame, target: str, years: list):
    model = LinearRegression()
    model.fit(data[["Year"]], data[target])
    future_years = pd.DataFrame({"Year": years})
    forecast_values = model.predict(future_years)
    return forecast_values.round(0).astype(int)


def plot_line(data: pd.DataFrame, column: str, title: str, ylabel: str, color: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        data["Year"],
        data[column],
        marker="o",
        linewidth=2.5,
        color=color,
    )
    style_axis(ax, title, ylabel=ylabel)
    return fig


df = load_data(DATA_PATH)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "EDA", "Modelling"])

selected_years = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max())),
)

filtered_df = df[
    (df["Year"] >= selected_years[0]) & (df["Year"] <= selected_years[1])
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(filtered_df)} records selected")


if page == "Home":
    st.title("Malaysian Road Accident Analysis Dashboard")
    st.caption("WIE2003 Introduction to Data Science")

    show_metric_row(filtered_df)

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Project Overview")
        st.markdown(
            """
            <div class="section-note">
            This dashboard explores Malaysian road safety trends using official yearly
            indicators. It combines exploratory data analysis, simple linear regression,
            and multiple linear regression for road deaths.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.pyplot(
            plot_line(
                filtered_df,
                "Road Deaths",
                "Road Deaths Trend",
                "Road Deaths",
                PRIMARY_COLOR,
            )
        )

    with right:
        st.subheader("Dashboard Sections")
        st.markdown(
            """
            - **Dataset**: preview, summary statistics, and CSV download.
            - **EDA**: trend charts, road user comparison, and correlation heatmap.
            - **Modelling**: regression metrics, model comparison plots, and forecasts.
            """
        )
        

elif page == "Dataset":
    st.title("Dataset Preview")
    st.caption("Cleaned Malaysian road safety dataset")

    show_metric_row(filtered_df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Filtered Data")
    with col2:
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered CSV",
            csv,
            file_name="filtered_road_safety_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(filtered_df, use_container_width=True)

    with st.expander("Summary Statistics"):
        stats_df = filtered_df.drop(columns=["Year"]).describe().T
        st.dataframe(stats_df, use_container_width=True)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.caption("Charts update based on the selected year range")

    show_metric_row(filtered_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Road Death Trend")
        st.pyplot(
            plot_line(
                filtered_df,
                "Road Deaths",
                "Road Deaths Over Time",
                "Road Deaths",
                PRIMARY_COLOR,
            )
        )

    with col2:
        st.subheader("Road Crashes Trend")
        st.pyplot(
            plot_line(
                filtered_df,
                "Road Crashes",
                "Road Crashes Over Time",
                "Road Crashes",
                WARNING_COLOR,
            )
        )

    st.subheader("Road User Death/Injury Categories")
    road_user_cols = [
        "Bicycle",
        "Bus",
        "Jeep",
        "Motorcar",
        "Motorcycle",
        "Others",
        "Pedestrian",
        "Trailer/Lorry",
        "Vans",
    ]
    road_user_totals = filtered_df[road_user_cols].sum().sort_values(ascending=True)

    fig_users, ax_users = plt.subplots(figsize=(10, 5.8))
    ax_users.barh(road_user_totals.index, road_user_totals.values, color=SUCCESS_COLOR)
    style_axis(
        ax_users,
        "Total by Road User Category",
        xlabel="Total Count",
        ylabel="Road User Category",
    )
    st.pyplot(fig_users)

    st.subheader("Correlation Heatmap")
    corr_cols = [
        "Registered Vehicles",
        "Population",
        "Road Crashes",
        "Road Deaths",
        "Serious Injury",
        "Slight Injury",
        "Fatality_Rate_per_100k",
    ]
    corr = filtered_df[corr_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(9.5, 6.5))
    heatmap = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(corr_cols)))
    ax_corr.set_yticks(range(len(corr_cols)))
    ax_corr.set_xticklabels(corr_cols, rotation=35, ha="right")
    ax_corr.set_yticklabels(corr_cols)
    ax_corr.set_title("Correlation Between Key Variables", fontsize=13, pad=12)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax_corr.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.55 else "#111827",
                fontsize=9,
            )

    fig_corr.colorbar(heatmap, ax=ax_corr, fraction=0.046, pad=0.04)
    st.pyplot(fig_corr)
    st.caption(
        "High correlations are expected because many road safety indicators trend together over time."
    )

elif page == "Modelling":
    st.title("Regression Modelling")
    st.caption("Models are fitted on the year range selected in the sidebar")

    if len(filtered_df) < 2:
        st.warning("Please select at least two years to fit and evaluate regression models.")
        st.stop()

    tab_simple, tab_multi = st.tabs(
        ["Simple Linear Regression", "Multiple Linear Regression"]
    )

    with tab_simple:
        st.subheader("Simple Linear Regression")
        st.markdown("**Feature used:** Year")

        _, simple_pred, simple_mae, simple_r2, simple_corr = fit_linear_model(
            filtered_df, ["Year"], "Road Deaths"
        )
        display_model_metrics(simple_mae, simple_r2, simple_corr)

        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=filtered_df["Year"],
            y=filtered_df["Road Deaths"],
            mode="markers",
            name="Actual Data",
            marker=dict(color=PRIMARY_COLOR, size=8),
            hovertemplate="Year: %{x}<br>Actual Deaths: %{y:,.0f}<extra></extra>"
        ))

        fig1.add_trace(go.Scatter(
            x=filtered_df["Year"],
            y=simple_pred,
            mode="lines",
            name="Regression Line",
            line=dict(color=ACCENT_COLOR, width=2.5),
            hovertemplate="Year: %{x}<br>Predicted Deaths: %{y:,.0f}<extra></extra>"
        ))

        fig1.update_layout(
            title="Simple Linear Regression: Road Deaths by Year",
            xaxis_title="Year",
            yaxis_title="Road Deaths",
            hovermode="x unified"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecasting Using Simple Regression")
        st.markdown(
            """
            Forecasting uses **Year** as the input feature because future values for
            vehicles, population, crashes, and road user categories are not available
            without making extra assumptions.
            """
        )

        forecast_year_input = st.number_input(
           "Enter a year to forecast (must be after 2016):",
        min_value=2017,
        max_value=2100,
        value=2025,
        step=1
        )
        forecast_years = [forecast_year_input]
        forecast_df = pd.DataFrame({"Year": forecast_years})
        forecast_df["Forecast Road Deaths"] = forecast_with_year_model(
            df, "Road Deaths", forecast_years
        )
        forecast_df["Forecast Total Injuries"] = forecast_with_year_model(
            df, "Total_Injuries_UserFile", forecast_years
        )
        forecast_df["Forecast Total Injuries"] = forecast_df["Forecast Total Injuries"].clip(lower=0)

        col_table, col_chart = st.columns([1, 1.35])
        with col_table:
            st.dataframe(forecast_df, use_container_width=True)
            st.caption(
                "Forecast values are estimates based on historical trends and should not be treated as exact future outcomes."
            )

        with col_chart:
            fig_forecast, ax_forecast = plt.subplots(figsize=(8, 4.8))
            ax_forecast.plot(
                df["Year"],
                df["Road Deaths"],
                marker="o",
                linewidth=2.5,
                color=PRIMARY_COLOR,
                label="Historical Road Deaths",
            )
            ax_forecast.plot(
                forecast_df["Year"],
                forecast_df["Forecast Road Deaths"],
                marker="s",
                linewidth=2.5,
                linestyle="--",
                color=ACCENT_COLOR,
                label="Forecast Road Deaths",
            )
            style_axis(
                ax_forecast,
                "Historical and Forecast Road Deaths",
                ylabel="Road Deaths",
            )
            ax_forecast.legend()
            st.pyplot(fig_forecast)

    with tab_multi:
        st.subheader("Multiple Linear Regression")
        st.markdown(
            "**Features used:** Year, Registered Vehicles, Population, Road Crashes"
        )
        st.caption(
            "Road user and vehicle-type columns are kept for EDA interpretation. They are not used for future forecasting because their future values are unknown and some may overlap directly with the target outcome."
        )

        multi_features = ["Year", "Registered Vehicles", "Population", "Road Crashes"]
        _, multi_pred, multi_mae, multi_r2, multi_corr = fit_linear_model(
            filtered_df, multi_features, "Road Deaths"
        )
        display_model_metrics(multi_mae, multi_r2, multi_corr)

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=filtered_df["Year"],
            y=filtered_df["Road Deaths"],
            mode="lines+markers",
            name="Actual Deaths",
            marker=dict(color=PRIMARY_COLOR, size=7),
            line=dict(color=PRIMARY_COLOR, width=3),
            hovertemplate="Year: %{x}<br>Actual Deaths: %{y:,.0f}<extra></extra>"
        ))

        fig2.add_trace(go.Scatter(
            x=filtered_df["Year"],
            y=multi_pred,
            mode="lines+markers",
            name="Predicted Deaths",
            marker=dict(color=ACCENT_COLOR, size=5, symbol="square"),
            line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
            hovertemplate="Year: %{x}<br>Predicted Deaths: %{y:,.0f}<extra></extra>"
        ))

        fig2.update_layout(
            title="Actual vs Predicted Road Deaths (Multiple Regression)",
            xaxis_title="Year",
            yaxis_title="Road Deaths",
            hovermode="x unified"
        )

        st.plotly_chart(fig2, use_container_width=True)

        residuals = filtered_df["Road Deaths"] - multi_pred
        fig_res, ax_res = plt.subplots(figsize=(10, 3.7))
        ax_res.axhline(0, color="#111827", linewidth=1)
        ax_res.bar(filtered_df["Year"], residuals, color=WARNING_COLOR, alpha=0.75)
        style_axis(
            ax_res,
            "Residuals: Actual Deaths Minus Predicted Deaths",
            ylabel="Residual",
        )
        st.pyplot(fig_res)

        st.info(
            "Multiple regression is used here for performance comparison and explanation. Predictions beyond the dataset are not shown because future values for vehicles, population, and crashes would require extra assumptions."
        )
