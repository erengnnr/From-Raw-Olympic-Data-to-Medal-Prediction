import streamlit as st
import pandas as pd
import plotly.express as px

# streamlit run app.py
#  SAYFA AYARLARI VE KARANLIK TEMA CSS
# cd C:\Users\tcell\Desktop\dsProject
st.set_page_config(page_title="Olympic Data Explorer", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:        #0a0a0c;
    --surface:   #111116;
    --border:    #1e1e28;
    --accent:    #e8c84a;   
    --text:      #e8e8ec;
    --muted:     #666680;
}
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: transparent !important; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(3rem, 6vw, 4.5rem); line-height: 0.95; margin: 0; }
.hero-title span { color: var(--accent); }
.hero-sub { font-size: 0.85rem; color: var(--muted); letter-spacing: 0.18em; text-transform: uppercase; margin-top: 0.6rem; margin-bottom: 2rem; }
.kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; text-align: center; }
.kpi-value { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: var(--accent); line-height: 1; }
.kpi-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  VERİ YÜKLEME (Doğrudan CSV'den)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    # Doğrudan senin masaüstündeki veri seti
    df = pd.read_csv(r"C:\Users\tcell\Desktop\athlete_events.csv")
    df['Medal_Won'] = df['Medal'].notna()
    return df

with st.spinner("Olimpiyat Verileri Yükleniyor..."):
    df = load_data()

# cd C:\Users\tcell\Desktop\dsProject
#  BAŞLIK VE ANA KPI (ÖZET) KARTLARI
# streamlit run app.py
st.markdown("""
<p class="hero-title">OLYMPIC<br><span>DATA</span> EXPLORER</p>
<p class="hero-sub">📊 &nbsp; Tarihsel Olimpiyat Verileri Keşif Paneli</p>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Toplam Katılım</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["ID"].nunique():,}</div><div class="kpi-label">Farklı Sporcu</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["Medal_Won"].sum():,}</div><div class="kpi-label">Dağıtılan Madalya</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["Sport"].nunique()}</div><div class="kpi-label">Farklı Spor Dalı</div></div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
#  FİLTRELEME ALANI
# ─────────────────────────────────────────────
st.subheader("Filtreler")
filt_col1, filt_col2, filt_col3 = st.columns(3)

with filt_col1:
    secilen_sezon = st.multiselect("Sezon Seçin", options=df['Season'].unique(), default=df['Season'].unique())
with filt_col2:
    secilen_cinsiyet = st.multiselect("Cinsiyet Seçin", options=df['Sex'].unique(), default=df['Sex'].unique())
with filt_col3:
    sporlar = sorted(df['Sport'].unique())
    secilen_sporlar = st.multiselect("Spor Dalı Filtrele (İsteğe Bağlı)", options=sporlar)

# Veriyi filtrelere göre daralt
df_filtered = df[(df['Season'].isin(secilen_sezon)) & (df['Sex'].isin(secilen_cinsiyet))]
if secilen_sporlar:
    df_filtered = df_filtered[df_filtered['Sport'].isin(secilen_sporlar)]

st.divider()

# ─────────────────────────────────────────────
#  GÖRSELLEŞTİRMELER (GRAFİKLER)
# ─────────────────────────────────────────────
st.subheader("Veri Analiz Grafikleri")
chart_col1, chart_col2 = st.columns(2)

# 1. En Çok Madalya Kazanan İlk 10 Ülke (Sadece Madalyalılar)
with chart_col1:
    madalyali_df = df_filtered[df_filtered['Medal_Won'] == True]
    top_ulkeler = madalyali_df['NOC'].value_counts().head(10).reset_index()
    top_ulkeler.columns = ['Ülke', 'Madalya Sayısı']
    
    fig1 = px.bar(top_ulkeler, x='Ülke', y='Madalya Sayısı', title="En Çok Madalya Kazanan Top 10 Ülke",
                  color='Madalya Sayısı', color_continuous_scale='Oryel', template="plotly_dark")
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

# 2. Yaş Dağılımı Histogramı
with chart_col2:
    fig2 = px.histogram(df_filtered, x='Age', nbins=30, title="Sporcuların Yaş Dağılımı",
                        color_discrete_sequence=['#e8c84a'], template="plotly_dark")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

# 3. Yıllara Göre Kadın/Erkek Katılım Oranı
with chart_col3:
    yillik_katilim = df_filtered.groupby(['Year', 'Sex']).size().reset_index(name='Katılımcı Sayısı')
    fig3 = px.line(yillik_katilim, x='Year', y='Katılımcı Sayısı', color='Sex', title="Yıllara Göre Cinsiyet Dağılımı",
                   color_discrete_map={'M': '#4a8fe8', 'F': '#e84a5f'}, template="plotly_dark")
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

# 4. Boy ve Kilo Dağılımı (Scatter Plot)
with chart_col4:
    # Çok veri olduğu için performansı artırmak adına 1000 rastgele örneklem alıyoruz
    df_sample = df_filtered.dropna(subset=['Height', 'Weight']).sample(n=min(1000, len(df_filtered)), random_state=42)
    
    fig4 = px.scatter(df_sample, x='Weight', y='Height', color='Sex', hover_data=['Sport'],
                      title="Boy ve Kilo İlişkisi (1000 Rastgele Örneklem)",
                      color_discrete_map={'M': '#4a8fe8', 'F': '#e84a5f'}, template="plotly_dark")
    fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
#  HAM VERİ TABLOSU
# ─────────────────────────────────────────────
st.divider()
st.subheader("Veri Seti Önizleme")
st.dataframe(df_filtered.head(100), use_container_width=True)