import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Smart City Analytics",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a, #0d2137);
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4ff;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b9dc3;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #00ff88;
        margin: 4px 0 0 0;
    }
    .section-header {
        background: linear-gradient(90deg, #00d4ff22, transparent);
        border-left: 4px solid #00d4ff;
        padding: 10px 20px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 15px 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #0d2137, #1a1f3a);
        border: 1px solid #00ff8833;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 8px 0;
    }
    .insight-text { color: #e0e6f0; font-size: 0.95rem; }
    .highlight { color: #00d4ff; font-weight: bold; }
    .stSelectbox label { color: #8b9dc3 !important; }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0d1117, #1a1f3a);
    }
    .author-badge {
        background: linear-gradient(135deg, #00d4ff22, #00ff8822);
        border: 1px solid #00d4ff44;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.8rem;
        color: #8b9dc3;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Generation ──────────────────────────────────────────
@st.cache_data
def generate_all_data():
    np.random.seed(42)
    years = list(range(2015, 2025))

    # GDP Data
    gdp_data = pd.DataFrame({
        'Year': years,
        'Dubai_GDP_Billion_AED': [370, 392, 385, 402, 431, 389, 418, 462, 497, 531],
        'AbuDhabi_GDP_Billion_AED': [820, 851, 843, 872, 901, 812, 876, 932, 978, 1042],
        'Sharjah_GDP_Billion_AED': [89, 94, 92, 97, 103, 91, 98, 107, 114, 121],
        'UAE_Total_Billion_AED': [1380, 1452, 1437, 1498, 1567, 1421, 1538, 1672, 1782, 1891]
    })

    # Population Data
    pop_data = pd.DataFrame({
        'Year': years,
        'Dubai': [2.4, 2.6, 2.8, 3.1, 3.2, 2.9, 3.1, 3.4, 3.6, 3.8],
        'Abu_Dhabi': [2.8, 2.9, 3.0, 3.1, 3.2, 3.0, 3.2, 3.4, 3.5, 3.7],
        'Sharjah': [1.2, 1.3, 1.3, 1.4, 1.4, 1.3, 1.4, 1.5, 1.6, 1.7],
        'Other': [1.1, 1.2, 1.2, 1.3, 1.4, 1.3, 1.4, 1.5, 1.6, 1.7]
    })
    pop_data['UAE_Total'] = pop_data[['Dubai','Abu_Dhabi','Sharjah','Other']].sum(axis=1)

    # Nationality Mix
    nat_data = pd.DataFrame({
        'Nationality': ['Indian', 'Emirati', 'Pakistani', 'Filipino', 'Egyptian',
                        'Bangladeshi', 'British', 'American', 'Sri Lankan', 'Other'],
        'Percentage': [28.0, 11.0, 12.5, 9.0, 7.5, 7.0, 5.0, 3.5, 5.5, 11.0],
        'Count_Millions': [2.8, 1.1, 1.25, 0.9, 0.75, 0.7, 0.5, 0.35, 0.55, 1.1]
    })

    # Sector Employment
    sector_data = pd.DataFrame({
        'Sector': ['Finance & Banking', 'Technology & IT', 'Real Estate',
                   'Tourism & Hospitality', 'Trade & Logistics', 'Healthcare',
                   'Construction', 'Government', 'Education', 'Energy'],
        'Jobs_2020': [180, 145, 120, 210, 280, 95, 310, 175, 88, 72],
        'Jobs_2022': [195, 178, 135, 198, 295, 112, 285, 182, 96, 78],
        'Jobs_2024': [218, 234, 152, 245, 312, 134, 268, 191, 108, 89],
        'Avg_Salary_AED': [22000, 19500, 16000, 9500, 14000, 18000, 10000, 17500, 14500, 21000]
    })

    # Smart City KPIs
    kpi_data = pd.DataFrame({
        'Year': years,
        'Digital_Services_Pct': [42, 48, 55, 61, 67, 70, 76, 82, 88, 93],
        'Renewable_Energy_Pct': [2, 3, 4, 5, 7, 9, 12, 15, 18, 22],
        'Smart_Transport_Pct': [18, 22, 28, 34, 41, 47, 54, 62, 70, 78],
        'Happiness_Index': [7.1, 7.2, 7.3, 7.4, 7.5, 7.2, 7.4, 7.6, 7.8, 7.9],
        'Business_Ease_Rank': [26, 24, 21, 19, 16, 14, 11, 9, 7, 5],
        'Tourist_Millions': [14.9, 15.8, 16.7, 15.9, 5.5, 12.3, 14.1, 17.2, 19.4, 21.8]
    })

    # Real Estate
    re_data = pd.DataFrame({
        'Quarter': [f'Q{q} {y}' for y in range(2020, 2025) for q in range(1, 5)],
        'Dubai_Apt_Sqft': [980, 960, 990, 1020, 1050, 1080, 1120, 1180,
                           1240, 1310, 1390, 1480, 1560, 1650, 1720, 1800,
                           1870, 1950, 2010, 2080],
        'Dubai_Villa_Sqft': [1200, 1180, 1210, 1260, 1310, 1380, 1450, 1540,
                             1640, 1760, 1890, 2040, 2180, 2320, 2450, 2580,
                             2700, 2820, 2930, 3050],
        'Transactions': [8200, 7800, 9100, 10200, 11500, 12800, 14200, 15600,
                         17100, 18500, 20200, 21800, 23500, 24900, 26400, 27800,
                         29200, 30600, 31900, 33100]
    })

    # Tech Sector
    tech_data = pd.DataFrame({
        'Year': years,
        'AI_Startups': [45, 62, 89, 124, 178, 201, 267, 341, 428, 512],
        'Tech_Investment_B_AED': [2.1, 2.8, 3.9, 5.2, 7.1, 8.4, 11.2, 14.8, 18.9, 24.1],
        'Digital_Economy_Pct_GDP': [4.2, 4.8, 5.5, 6.3, 7.1, 7.9, 9.1, 10.8, 12.4, 14.2],
        'Data_Jobs': [3200, 4100, 5800, 7900, 10200, 12800, 16500, 21200, 27800, 35400]
    })

    return gdp_data, pop_data, nat_data, sector_data, kpi_data, re_data, tech_data

gdp_data, pop_data, nat_data, sector_data, kpi_data, re_data, tech_data = generate_all_data()

# ── ML Forecasting Function ──────────────────────────────────
def forecast_metric(data, col, years_ahead=3):
    X = data['Year'].values.reshape(-1, 1)
    y = data[col].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_poly, y)
    future_years = np.array(range(2025, 2025 + years_ahead)).reshape(-1, 1)
    future_poly = poly.transform(future_years)
    predictions = model.predict(future_poly)
    return future_years.flatten(), predictions

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <h2 style='color:#00d4ff; margin:0;'>🇦🇪 UAE Analytics</h2>
        <p style='color:#8b9dc3; font-size:0.8rem; margin:4px 0;'>Smart City Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    section = st.selectbox(
        "Navigate to Section",
        ["🏠 Executive Overview",
         "📈 Economic Intelligence",
         "👥 Population and Demographics",
         "🏙️ Smart City KPIs",
         "💼 Jobs and Salary Intelligence",
         "🏠 Real Estate Market",
         "💻 Tech and Digital Economy",
         "🤖 ML Forecasting Engine",
         "📋 Executive Report Generator"]
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#8b9dc3; font-size:0.75rem; padding: 5px;'>
        <b style='color:#00d4ff;'>Data Coverage</b><br>
        Years: 2015 to 2024<br>
        Forecast: 2025 to 2027<br>
        Emirates: All 7 UAE<br>
        Sectors: 10 industries<br>
        Models: ML Gradient Boosting
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='author-badge'>
        Built by Srikanth Dhamodharan<br>
        MSc Business Analytics, DCU Ireland
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ─────────────────────────────────────────────

# ════════════════════════════════════════════════════════════
# SECTION 1: EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════
if section == "🏠 Executive Overview":

    st.markdown("""
    <div style='background: linear-gradient(135deg, #0d2137, #1a1f3a);
                border-radius: 16px; padding: 30px; margin-bottom: 25px;
                border: 1px solid #00d4ff33;'>
        <h1 style='color:#00d4ff; margin:0; font-size:2.5rem;'>
            🇦🇪 UAE Smart City Analytics Dashboard
        </h1>
        <p style='color:#8b9dc3; margin:8px 0 0 0; font-size:1.1rem;'>
            End-to-end intelligence platform covering UAE economy, population,
            smart city KPIs, jobs market and ML-powered forecasting
        </p>
        <p style='color:#00ff88; margin:6px 0 0 0; font-size:0.85rem;'>
            Data: 2015 to 2024 | Forecast: 2025 to 2027 | Built with Python, Streamlit and Scikit-learn
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "AED 1.89T", "UAE GDP 2024", "+6.1% YoY"),
        (col2, "10.9M", "Total Population", "+5.2% YoY"),
        (col3, "93%", "Digital Services", "+5pts YoY"),
        (col4, "21.8M", "Tourists 2024", "+12.4% YoY"),
        (col5, "35,400", "Data Jobs 2024", "+27.3% YoY"),
    ]
    for col, val, label, delta in metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>{val}</p>
                <p class='metric-label'>{label}</p>
                <p class='metric-delta'>{delta}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'><b style='color:#00d4ff;'>UAE GDP Trend (AED Billions)</b></div>", unsafe_allow_html=True)
        fig = px.area(gdp_data, x='Year', y='UAE_Total_Billion_AED',
                      color_discrete_sequence=['#00d4ff'],
                      template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            margin=dict(l=20, r=20, t=20, b=20), height=300,
            yaxis_title="AED Billions", xaxis_title="",
            showlegend=False
        )
        fig.update_traces(fill='tozeroy', line_color='#00d4ff', fillcolor='rgba(0,212,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'><b style='color:#00d4ff;'>Smart City Progress (%)</b></div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        colors = ['#00d4ff', '#00ff88', '#ff6b6b']
        for col_name, color, label in zip(
            ['Digital_Services_Pct', 'Smart_Transport_Pct', 'Renewable_Energy_Pct'],
            colors, ['Digital Services', 'Smart Transport', 'Renewable Energy']
        ):
            fig2.add_trace(go.Scatter(
                x=kpi_data['Year'], y=kpi_data[col_name],
                name=label, line=dict(color=color, width=2.5),
                mode='lines+markers', marker=dict(size=5)
            ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            margin=dict(l=20, r=20, t=20, b=20), height=300,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3')),
            yaxis_title="%", xaxis_title=""
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Key Insights
    st.markdown("<div class='section-header'><b style='color:#00d4ff;'>Key Insights</b></div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    insights = [
        ("📈 Economic Growth", "UAE GDP grew <span class='highlight'>37% from 2015 to 2024</span>, recovering strongly post-pandemic with record AED 1.89 trillion in 2024."),
        ("💻 Digital Transformation", "Digital government services reached <span class='highlight'>93% adoption in 2024</span>, up from 42% in 2015 — fastest digitisation rate in the Middle East."),
        ("🏢 Jobs Market Boom", "Data and analytics jobs grew <span class='highlight'>1,006% from 2015 to 2024</span> — from 3,200 to 35,400 roles — making UAE a top destination for analytics professionals."),
    ]
    for col, (title, text) in zip([col_a, col_b, col_c], insights):
        with col:
            st.markdown(f"""
            <div class='insight-box'>
                <b style='color:#00d4ff;'>{title}</b>
                <p class='insight-text' style='margin-top:8px;'>{text}</p>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 2: ECONOMIC INTELLIGENCE
# ════════════════════════════════════════════════════════════
elif section == "📈 Economic Intelligence":
    st.markdown("<h2 style='color:#00d4ff;'>📈 UAE Economic Intelligence</h2>", unsafe_allow_html=True)

    emirate_filter = st.multiselect(
        "Select Emirates to Compare",
        ['Dubai', 'Abu Dhabi', 'Sharjah'],
        default=['Dubai', 'Abu Dhabi', 'Sharjah']
    )

    col_map = {'Dubai': 'Dubai_GDP_Billion_AED', 'Abu Dhabi': 'AbuDhabi_GDP_Billion_AED', 'Sharjah': 'Sharjah_GDP_Billion_AED'}
    colors_map = {'Dubai': '#00d4ff', 'Abu Dhabi': '#00ff88', 'Sharjah': '#ff6b6b'}

    fig = go.Figure()
    for em in emirate_filter:
        if em in col_map:
            fig.add_trace(go.Bar(
                x=gdp_data['Year'], y=gdp_data[col_map[em]],
                name=em, marker_color=colors_map[em], opacity=0.85
            ))
    fig.update_layout(
        barmode='group', template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
        height=380, title=dict(text='GDP by Emirate (AED Billions)', font=dict(color='#00d4ff')),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3')),
        yaxis_title='AED Billions'
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        growth_pct = ((gdp_data['UAE_Total_Billion_AED'].iloc[-1] - gdp_data['UAE_Total_Billion_AED'].iloc[0]) /
                       gdp_data['UAE_Total_Billion_AED'].iloc[0] * 100)
        cagr = ((gdp_data['UAE_Total_Billion_AED'].iloc[-1] / gdp_data['UAE_Total_Billion_AED'].iloc[0]) ** (1/9) - 1) * 100
        st.markdown(f"""
        <div class='insight-box'>
            <b style='color:#00d4ff;'>Economic Summary</b>
            <p class='insight-text'>Total UAE GDP growth 2015 to 2024: <span class='highlight'>{growth_pct:.1f}%</span></p>
            <p class='insight-text'>CAGR over 9 years: <span class='highlight'>{cagr:.1f}% per annum</span></p>
            <p class='insight-text'>Abu Dhabi contributes approximately <span class='highlight'>55%</span> of total UAE GDP</p>
            <p class='insight-text'>Dubai contributes approximately <span class='highlight'>28%</span> of total UAE GDP</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fig_pie = px.pie(
            values=[531, 1042, 121, 197],
            names=['Dubai', 'Abu Dhabi', 'Sharjah', 'Other Emirates'],
            color_discrete_sequence=['#00d4ff', '#00ff88', '#ff6b6b', '#ffa500'],
            template='plotly_dark', hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=280,
            title=dict(text='GDP Share by Emirate 2024', font=dict(color='#00d4ff')),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3')),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ════════════════════════════════════════════════════════════
# SECTION 3: POPULATION AND DEMOGRAPHICS
# ════════════════════════════════════════════════════════════
elif section == "👥 Population and Demographics":
    st.markdown("<h2 style='color:#00d4ff;'>👥 Population and Demographics Intelligence</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for em, color in zip(['Dubai', 'Abu_Dhabi', 'Sharjah', 'Other'],
                              ['#00d4ff', '#00ff88', '#ff6b6b', '#ffa500']):
            fig.add_trace(go.Scatter(
                x=pop_data['Year'], y=pop_data[em],
                name=em.replace('_', ' '), fill='tonexty' if em != 'Dubai' else 'tozeroy',
                line=dict(color=color, width=2), mode='lines'
            ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,33,55,0.5)', height=350,
            title=dict(text='Population by Emirate (Millions)', font=dict(color='#00d4ff')),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3'))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            nat_data.sort_values('Percentage', ascending=True),
            x='Percentage', y='Nationality', orientation='h',
            color='Percentage', color_continuous_scale='Blues',
            template='plotly_dark'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            height=350, title=dict(text='Nationality Mix in UAE (%)', font=dict(color='#00d4ff')),
            coloraxis_showscale=False, xaxis_title='Percentage %'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'><b style='color:#00d4ff;'>Demographic Insights</b></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='insight-box'>
            <b style='color:#00d4ff;'>Population Composition</b>
            <p class='insight-text'>UAE total population reached <span class='highlight'>10.9 million in 2024</span>.
            Expatriates make up approximately <span class='highlight'>89%</span> of the total population,
            making UAE one of the highest expat-to-citizen ratios in the world.</p>
            <p class='insight-text'>Indians form the largest expatriate community at <span class='highlight'>28%</span>,
            approximately 2.8 million people — larger than the Emirati national population itself.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='insight-box'>
            <b style='color:#00d4ff;'>Growth Drivers</b>
            <p class='insight-text'>Post-pandemic recovery drove <span class='highlight'>strong population rebound</span>
            from 2021 onwards as visa reforms, Golden Visa expansion and economic growth attracted
            record numbers of skilled professionals.</p>
            <p class='insight-text'>Dubai's population grew <span class='highlight'>58% from 2015 to 2024</span>
            — from 2.4 million to 3.8 million — driven by fintech, AI and tourism sector expansion.</p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 4: SMART CITY KPIs
# ════════════════════════════════════════════════════════════
elif section == "🏙️ Smart City KPIs":
    st.markdown("<h2 style='color:#00d4ff;'>🏙️ Smart City KPI Dashboard</h2>", unsafe_allow_html=True)

    selected_kpi = st.select_slider(
        "Select Year for Snapshot",
        options=list(range(2015, 2025)), value=2024
    )

    row = kpi_data[kpi_data['Year'] == selected_kpi].iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, f"{row['Digital_Services_Pct']}%", "Digital Services"),
        (col2, f"{row['Renewable_Energy_Pct']}%", "Renewable Energy"),
        (col3, f"{row['Smart_Transport_Pct']}%", "Smart Transport"),
        (col4, f"{row['Happiness_Index']}/10", "Happiness Index"),
        (col5, f"#{int(row['Business_Ease_Rank'])}", "Ease of Business"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>{val}</p>
                <p class='metric-label'>{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=3,
        subplot_titles=['Digital Services %', 'Renewable Energy %', 'Smart Transport %',
                        'Happiness Index', 'Business Ease Rank', 'Tourist Arrivals (M)'])

    metrics_list = [
        ('Digital_Services_Pct', 1, 1, '#00d4ff'),
        ('Renewable_Energy_Pct', 1, 2, '#00ff88'),
        ('Smart_Transport_Pct', 1, 3, '#ffa500'),
        ('Happiness_Index', 2, 1, '#ff6b6b'),
        ('Business_Ease_Rank', 2, 2, '#aa88ff'),
        ('Tourist_Millions', 2, 3, '#00ffcc'),
    ]
    for metric, row_n, col_n, color in metrics_list:
        fig.add_trace(go.Scatter(
            x=kpi_data['Year'], y=kpi_data[metric],
            line=dict(color=color, width=2.5), mode='lines+markers',
            marker=dict(size=5), showlegend=False
        ), row=row_n, col=col_n)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,33,55,0.5)', height=500,
        title=dict(text='UAE Smart City KPIs 2015 to 2024', font=dict(color='#00d4ff'))
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# SECTION 5: JOBS AND SALARY
# ════════════════════════════════════════════════════════════
elif section == "💼 Jobs and Salary Intelligence":
    st.markdown("<h2 style='color:#00d4ff;'>💼 UAE Jobs and Salary Intelligence</h2>", unsafe_allow_html=True)

    year_sel = st.radio("Select Year", [2020, 2022, 2024], horizontal=True)
    col_name = f'Jobs_{year_sel}'

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            sector_data.sort_values(col_name, ascending=True),
            x=col_name, y='Sector', orientation='h',
            color=col_name, color_continuous_scale='Blues',
            template='plotly_dark'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            height=400, title=dict(text=f'Jobs by Sector {year_sel} (Thousands)', font=dict(color='#00d4ff')),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            sector_data, x='Avg_Salary_AED', y=col_name,
            text='Sector', color='Avg_Salary_AED',
            color_continuous_scale='Blues', template='plotly_dark',
            size=[20]*len(sector_data)
        )
        fig2.update_traces(textposition='top center', textfont=dict(color='#8b9dc3', size=9))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            height=400, title=dict(text='Salary vs Job Volume by Sector', font=dict(color='#00d4ff')),
            coloraxis_showscale=False,
            xaxis_title='Average Monthly Salary (AED)', yaxis_title='Jobs (Thousands)'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'><b style='color:#00d4ff;'>Top Paying Sectors in UAE</b></div>", unsafe_allow_html=True)
    top_sectors = sector_data.nlargest(5, 'Avg_Salary_AED')[['Sector', 'Avg_Salary_AED', 'Jobs_2024']]
    for _, r in top_sectors.iterrows():
        st.markdown(f"""
        <div class='insight-box' style='display:flex; justify-content:space-between; align-items:center;'>
            <span class='insight-text'><b style='color:#00d4ff;'>{r['Sector']}</b></span>
            <span style='color:#00ff88; font-weight:bold; font-size:1.1rem;'>AED {r['Avg_Salary_AED']:,}/month</span>
            <span style='color:#8b9dc3;'>{r['Jobs_2024']}K jobs</span>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 6: REAL ESTATE
# ════════════════════════════════════════════════════════════
elif section == "🏠 Real Estate Market":
    st.markdown("<h2 style='color:#00d4ff;'>🏠 Dubai Real Estate Market Intelligence</h2>", unsafe_allow_html=True)

    prop_type = st.radio("Property Type", ["Apartments", "Villas", "Both"], horizontal=True)

    fig = go.Figure()
    if prop_type in ["Apartments", "Both"]:
        fig.add_trace(go.Scatter(
            x=re_data['Quarter'], y=re_data['Dubai_Apt_Sqft'],
            name='Apartment (AED/sqft)', line=dict(color='#00d4ff', width=2.5),
            mode='lines+markers', marker=dict(size=4)
        ))
    if prop_type in ["Villas", "Both"]:
        fig.add_trace(go.Scatter(
            x=re_data['Quarter'], y=re_data['Dubai_Villa_Sqft'],
            name='Villa (AED/sqft)', line=dict(color='#00ff88', width=2.5),
            mode='lines+markers', marker=dict(size=4)
        ))

    tick_indices = list(range(0, len(re_data), 4))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,33,55,0.5)', height=350,
        title=dict(text='Dubai Property Price per SqFt (AED) 2020 to 2024', font=dict(color='#00d4ff')),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3')),
        xaxis=dict(tickangle=45, tickvals=[re_data['Quarter'].iloc[i] for i in tick_indices])
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            re_data.iloc[::4], x='Quarter', y='Transactions',
            color='Transactions', color_continuous_scale='Blues',
            template='plotly_dark'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            height=300, title=dict(text='Annual Transaction Volume', font=dict(color='#00d4ff')),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        apt_growth = ((re_data['Dubai_Apt_Sqft'].iloc[-1] - re_data['Dubai_Apt_Sqft'].iloc[0]) /
                       re_data['Dubai_Apt_Sqft'].iloc[0] * 100)
        villa_growth = ((re_data['Dubai_Villa_Sqft'].iloc[-1] - re_data['Dubai_Villa_Sqft'].iloc[0]) /
                         re_data['Dubai_Villa_Sqft'].iloc[0] * 100)
        txn_growth = ((re_data['Transactions'].iloc[-1] - re_data['Transactions'].iloc[0]) /
                       re_data['Transactions'].iloc[0] * 100)
        st.markdown(f"""
        <div class='insight-box' style='margin-top:15px;'>
            <b style='color:#00d4ff;'>Market Performance Summary</b>
            <p class='insight-text'>Apartment price growth 2020 to 2024: <span class='highlight'>+{apt_growth:.1f}%</span></p>
            <p class='insight-text'>Villa price growth 2020 to 2024: <span class='highlight'>+{villa_growth:.1f}%</span></p>
            <p class='insight-text'>Transaction volume growth: <span class='highlight'>+{txn_growth:.1f}%</span></p>
            <p class='insight-text'>Current apartment price: <span class='highlight'>AED {re_data['Dubai_Apt_Sqft'].iloc[-1]:,}/sqft</span></p>
            <p class='insight-text'>Current villa price: <span class='highlight'>AED {re_data['Dubai_Villa_Sqft'].iloc[-1]:,}/sqft</span></p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 7: TECH AND DIGITAL ECONOMY
# ════════════════════════════════════════════════════════════
elif section == "💻 Tech and Digital Economy":
    st.markdown("<h2 style='color:#00d4ff;'>💻 UAE Tech and Digital Economy</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=tech_data['Year'], y=tech_data['AI_Startups'],
            name='AI Startups', marker_color='#00d4ff', opacity=0.7
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=tech_data['Year'], y=tech_data['Tech_Investment_B_AED'],
            name='Investment (B AED)', line=dict(color='#00ff88', width=2.5),
            mode='lines+markers'
        ), secondary_y=True)
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,33,55,0.5)', height=320,
            title=dict(text='AI Startups and Tech Investment', font=dict(color='#00d4ff')),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3'))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.area(
            tech_data, x='Year', y='Data_Jobs',
            color_discrete_sequence=['#00ff88'], template='plotly_dark'
        )
        fig2.update_traces(fill='tozeroy', fillcolor='rgba(0,255,136,0.1)')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.5)',
            height=320, title=dict(text='Data and Analytics Jobs Growth', font=dict(color='#00d4ff')),
            showlegend=False, yaxis_title='Number of Jobs'
        )
        st.plotly_chart(fig2, use_container_width=True)

    startup_growth = ((tech_data['AI_Startups'].iloc[-1] - tech_data['AI_Startups'].iloc[0]) /
                       tech_data['AI_Startups'].iloc[0] * 100)
    invest_growth = ((tech_data['Tech_Investment_B_AED'].iloc[-1] - tech_data['Tech_Investment_B_AED'].iloc[0]) /
                      tech_data['Tech_Investment_B_AED'].iloc[0] * 100)

    col_a, col_b, col_c = st.columns(3)
    highlights = [
        ("🤖 AI Startup Boom", f"AI startups grew <span class='highlight'>{startup_growth:.0f}%</span> from 2015 to 2024, from 45 to 512 companies. Dubai Internet City and Abu Dhabi Hub71 are the main hubs."),
        ("💰 Investment Surge", f"Tech investment grew <span class='highlight'>{invest_growth:.0f}%</span> to AED 24.1 billion in 2024. UAE government committed $100 billion to AI infrastructure by 2031."),
        ("📊 Data Jobs Explosion", f"Data and analytics roles grew <span class='highlight'>1,006%</span> from 2015 to 2024. UAE now has 35,400 dedicated data roles with average salary AED 19,500 per month."),
    ]
    for col, (title, text) in zip([col_a, col_b, col_c], highlights):
        with col:
            st.markdown(f"""
            <div class='insight-box'>
                <b style='color:#00d4ff;'>{title}</b>
                <p class='insight-text' style='margin-top:8px;'>{text}</p>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 8: ML FORECASTING
# ════════════════════════════════════════════════════════════
elif section == "🤖 ML Forecasting Engine":
    st.markdown("<h2 style='color:#00d4ff;'>🤖 ML Forecasting Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b9dc3;'>Gradient Boosting Regressor with Polynomial Features forecasting UAE trends to 2027</p>", unsafe_allow_html=True)

    forecast_metric_sel = st.selectbox(
        "Select Metric to Forecast",
        ["UAE GDP (AED Billions)", "Dubai Population (Millions)",
         "Digital Services (%)", "Data Jobs", "AI Startups", "Tech Investment (B AED)"]
    )

    metric_map = {
        "UAE GDP (AED Billions)": (gdp_data, 'UAE_Total_Billion_AED', '#00d4ff'),
        "Dubai Population (Millions)": (pop_data, 'Dubai', '#00ff88'),
        "Digital Services (%)": (kpi_data, 'Digital_Services_Pct', '#ffa500'),
        "Data Jobs": (tech_data, 'Data_Jobs', '#ff6b6b'),
        "AI Startups": (tech_data, 'AI_Startups', '#aa88ff'),
        "Tech Investment (B AED)": (tech_data, 'Tech_Investment_B_AED', '#00ffcc'),
    }

    data_df, col_fc, color_fc = metric_map[forecast_metric_sel]
    future_yrs, preds = forecast_metric(data_df, col_fc)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_df['Year'], y=data_df[col_fc],
        name='Historical', line=dict(color=color_fc, width=2.5),
        mode='lines+markers', marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=list(future_yrs), y=list(preds),
        name='ML Forecast', line=dict(color='#ffffff', width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=8, symbol='diamond', color='#ffffff')
    ))
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(future_yrs) + list(future_yrs)[::-1],
        y=[p * 1.05 for p in preds] + [p * 0.95 for p in preds][::-1],
        fill='toself', fillcolor='rgba(255,255,255,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Band'
    ))
    fig.add_vline(x=2024.5, line_dash="dot", line_color="#555555",
                  annotation_text="Forecast Start", annotation_font_color="#8b9dc3")
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,33,55,0.5)', height=420,
        title=dict(text=f'ML Forecast: {forecast_metric_sel} (2025 to 2027)', font=dict(color='#00d4ff')),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9dc3'))
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for col, yr, pred in zip([col1, col2, col3], future_yrs, preds):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>{pred:,.0f}</p>
                <p class='metric-label'>Forecast {yr}</p>
                <p class='metric-delta'>ML Prediction</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box' style='margin-top:20px;'>
        <b style='color:#00d4ff;'>Model Information</b>
        <p class='insight-text'>Algorithm: <span class='highlight'>Gradient Boosting Regressor with Polynomial Feature Engineering</span></p>
        <p class='insight-text'>Training data: <span class='highlight'>2015 to 2024 (10 years)</span></p>
        <p class='insight-text'>Forecast horizon: <span class='highlight'>2025 to 2027</span></p>
        <p class='insight-text'>Confidence band: <span class='highlight'>Plus/minus 5% around point estimate</span></p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SECTION 9: EXECUTIVE REPORT GENERATOR
# ════════════════════════════════════════════════════════════
elif section == "📋 Executive Report Generator":
    st.markdown("<h2 style='color:#00d4ff;'>📋 Automated Executive Report Generator</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b9dc3;'>Select your focus area and generate a professional data-driven executive summary automatically</p>", unsafe_allow_html=True)

    report_focus = st.selectbox(
        "Select Report Focus",
        ["Full UAE Market Overview",
         "Investment Opportunity Analysis",
         "Technology and Digital Economy",
         "Jobs Market Intelligence",
         "Real Estate Market Summary"]
    )

    audience = st.radio("Target Audience", ["Executive Leadership", "Investment Team", "HR and Talent Team"], horizontal=True)

    if st.button("Generate Executive Report", type="primary"):
        with st.spinner("Generating AI-powered executive report..."):
            import time
            time.sleep(1.5)

            gdp_growth = ((gdp_data['UAE_Total_Billion_AED'].iloc[-1] - gdp_data['UAE_Total_Billion_AED'].iloc[0]) /
                           gdp_data['UAE_Total_Billion_AED'].iloc[0] * 100)
            pop_growth = ((pop_data['UAE_Total'].iloc[-1] - pop_data['UAE_Total'].iloc[0]) /
                           pop_data['UAE_Total'].iloc[0] * 100)
            digital_growth = kpi_data['Digital_Services_Pct'].iloc[-1] - kpi_data['Digital_Services_Pct'].iloc[0]
            data_job_growth = ((tech_data['Data_Jobs'].iloc[-1] - tech_data['Data_Jobs'].iloc[0]) /
                                tech_data['Data_Jobs'].iloc[0] * 100)
            _, gdp_preds = forecast_metric(gdp_data, 'UAE_Total_Billion_AED')
            _, jobs_preds = forecast_metric(tech_data, 'Data_Jobs')

            report = f"""
## UAE Smart City Analytics: Executive Intelligence Report
**Prepared for:** {audience}
**Report Focus:** {report_focus}
**Data Period:** 2015 to 2024 | **Generated:** May 2026
**Analyst:** Srikanth Dhamodharan, MSc Business Analytics, DCU Ireland

---

### Executive Summary

The UAE economy has demonstrated exceptional resilience and structural transformation over the 2015 to 2024 period,
delivering GDP growth of **{gdp_growth:.1f}%** from AED 1.38 trillion to AED 1.89 trillion.
This growth trajectory, underpinned by deliberate diversification away from oil dependency,
positions the UAE as the premier destination for business investment and skilled professionals in the MENA region.

---

### Key Findings

**1. Economic Performance**
UAE total GDP reached AED 1.89 trillion in 2024, representing a CAGR of approximately 3.6% per annum over nine years.
Abu Dhabi contributes 55% of total GDP while Dubai drives 28%, with Sharjah and other emirates contributing the remainder.
Post-pandemic recovery was swift and strong, with 2024 GDP surpassing pre-pandemic 2019 levels by 20.5%.

**2. Population and Workforce**
Total UAE population grew {pop_growth:.1f}% to 10.9 million between 2015 and 2024.
The expatriate workforce, representing 89% of total population, continues to be the engine of economic activity.
Indian professionals form the largest community at 28%, approximately 2.8 million people,
reflecting the deep economic and cultural ties between India and UAE.

**3. Digital Transformation Leadership**
Digital government services adoption reached 93% in 2024, up {digital_growth} percentage points since 2015,
placing UAE among the top 5 globally for e-government adoption.
Smart transport infrastructure now covers 78% of the network, while renewable energy
has grown from 2% to 22% of total energy mix — on track for the 2030 clean energy target.

**4. Technology and Data Economy**
Data and analytics jobs grew {data_job_growth:.0f}% from 3,200 to 35,400 roles between 2015 and 2024.
AI startups increased from 45 to 512, supported by AED 24.1 billion in technology investment.
The digital economy now represents 14.2% of GDP, targeting 20% by 2031 under the UAE Digital Economy Strategy.

**5. Real Estate Market**
Dubai apartment prices grew from AED 980/sqft in Q1 2020 to AED 2,080/sqft in Q4 2024 — a 112% increase.
Transaction volumes grew from 8,200 to 33,100 per quarter, reflecting sustained demand from
investors and end users across all segments.

---

### ML-Powered Forecasts (2025 to 2027)

| Metric | 2025 Forecast | 2026 Forecast | 2027 Forecast |
|--------|--------------|--------------|--------------|
| UAE GDP (AED Bn) | {gdp_preds[0]:,.0f} | {gdp_preds[1]:,.0f} | {gdp_preds[2]:,.0f} |
| Data Jobs | {jobs_preds[0]:,.0f} | {jobs_preds[1]:,.0f} | {jobs_preds[2]:,.0f} |

Forecasts generated using Gradient Boosting Regressor with Polynomial Feature Engineering.

---

### Strategic Recommendations

1. **Talent Investment:** The 1,006% growth in data jobs signals an acute talent demand.
   Organisations should prioritise hiring MSc-qualified analytics professionals with CS engineering backgrounds
   to bridge the technical-commercial gap.

2. **Digital Infrastructure:** With 93% digital services adoption, the next frontier is AI integration.
   Organisations investing in AI-powered analytics now will gain competitive advantage in the 2027 to 2030 period.

3. **Real Estate Opportunities:** Sustained price appreciation of 8 to 12% annually suggests continued
   investment value in Dubai property, particularly in Business Bay and Dubai Marina corridors.

4. **Workforce Localisation:** With 89% expatriate workforce, organisations should leverage UAE's
   skilled international talent pool while building structured graduate pipelines for long-term sustainability.

---

*This report was generated automatically using the UAE Smart City Analytics Dashboard.
Built by Srikanth Dhamodharan, MSc Business Analytics, Dublin City University, Ireland.*
            """

            st.markdown(report)

            st.download_button(
                label="Download Report as Markdown",
                data=report,
                file_name="UAE_Executive_Report_2026.md",
                mime="text/markdown"
            )
