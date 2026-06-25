import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #cce0f5; margin: 0.5rem 0 0; font-size: 1rem; }

    /* Kartu section form */
    .section-card {
        background: #f8fafd;
        border: 1px solid #dce8f5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem 0.5rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e3a5f;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Kartu hasil */
    .result-churn {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border: 2px solid #ff4b4b;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #21c35422, #21c35411);
        border: 2px solid #21c354;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-sub { font-size: 0.95rem; color: #555; }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e0e9f5;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 2rem; font-weight: 800; }
    .metric-lbl { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }

    /* Tombol prediksi */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem;
        transition: opacity 0.2s;
    }
    div.stButton > button[kind="primary"]:hover { opacity: 0.88; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #f0f5fb; }

    /* Sembunyikan footer streamlit */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(BASE_DIR, "best_model_rf.joblib"))
    scaler   = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    features = joblib.load(os.path.join(BASE_DIR, "selected_features.joblib"))
    return model, scaler, features

model, scaler, selected_features = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## Churn Prediction")
    st.markdown("---")

    st.markdown("### 🤖 Tentang Model")
    st.info(
        "**Algoritma:** Random Forest\n\n"
        "**Skenario:** Hyperparameter Tuning\n\n"
        "**Performa:**\n"
        "| Metrik | Nilai |\n"
        "|--------|-------|\n"
        "| Accuracy | 0.8573 |\n"
        "| Precision | 0.5235 |\n"
        "| Recall | 0.7761 |\n"
        "| F1-Score | 0.6252 |"
    )

    st.markdown("### 📖 Cara Penggunaan")
    st.markdown(
        "1. Isi semua field pada form\n"
        "2. Klik tombol **Prediksi Churn**\n"
        "3. Baca hasil dan rekomendasi\n\n"
        "🟢 **Churn = 0** → Pelanggan bertahan\n\n"
        "🔴 **Churn = 1** → Berpotensi churn"
    )

    st.markdown("### 📌 Fitur Kunci")
    st.markdown(
        "- **Tenure Days** — lama berlangganan\n"
        "- **Days Since Last Purchase** — keaktifan\n"
        "- **Satisfaction Score** — kepuasan\n"
        "- **NPS Score** — loyalitas\n"
        "- **Support Tickets** — pengalaman layanan"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Prediction</h1>
    <p>Prediksi potensi churn pelanggan menggunakan model Random Forest
    yang dioptimalkan dengan Hyperparameter Tuning &amp; SMOTE balancing.</p>
</div>
""", unsafe_allow_html=True)

# ── Form Input ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card"><div class="section-title">👤 Data Demografis & Akun</div>', unsafe_allow_html=True)
    age             = st.number_input("Usia", min_value=18, max_value=100, value=35)
    is_premium_user = st.selectbox("Premium User", [0, 1], format_func=lambda x: "Ya ✓" if x == 1 else "Tidak ✗")
    tenure_days     = st.number_input("Lama Berlangganan (hari)", min_value=0, max_value=2000, value=365)
    days_since_last = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0, max_value=1000, value=30)
    last_3month     = st.number_input("Frekuensi Beli 3 Bulan Terakhir", min_value=0, max_value=50, value=3)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card"><div class="section-title">💰 Aktivitas & Transaksi</div>', unsafe_allow_html=True)
    total_visits      = st.number_input("Total Kunjungan", min_value=0, max_value=500, value=20)
    avg_session_time  = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0, max_value=120.0, value=10.0, step=0.5)
    pages_per_session = st.number_input("Halaman per Sesi", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    total_spent       = st.number_input("Total Pengeluaran", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)
    avg_order_value   = st.number_input("Rata-rata Nilai Transaksi", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
    lifetime_value    = st.number_input("Lifetime Value", min_value=0.0, max_value=100000.0, value=1000.0, step=100.0)
    discount_used     = st.selectbox("Gunakan Diskon?", [0, 1], format_func=lambda x: "Ya ✓" if x == 1 else "Tidak ✗")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card"><div class="section-title">📧 Email, Kepuasan & Layanan</div>', unsafe_allow_html=True)
    email_open_rate    = st.slider("Email Open Rate", 0.0, 1.0, 0.3, step=0.01)
    email_click_rate   = st.slider("Email Click Rate", 0.0, 1.0, 0.1, step=0.01)
    satisfaction_score = st.slider("Skor Kepuasan", 0.0, 10.0, 7.0, step=0.1)
    nps_score          = st.slider("NPS Score", -100, 100, 30)
    support_tickets    = st.number_input("Jumlah Tiket Support", min_value=0, max_value=20, value=1)
    refund_requested   = st.selectbox("Pernah Request Refund?", [0, 1], format_func=lambda x: "Ya ✓" if x == 1 else "Tidak ✗")
    delivery_delay     = st.number_input("Keterlambatan Pengiriman (hari)", min_value=0, max_value=30, value=0)
    marketing_spend    = st.number_input("Marketing Spend per User", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tombol Prediksi ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Prediksi Churn Sekarang", type="primary", use_container_width=True)

if predict_btn:
    input_dict = {
        "age": age, "is_premium_user": is_premium_user,
        "total_visits": total_visits, "avg_session_time": avg_session_time,
        "pages_per_session": pages_per_session, "email_open_rate": email_open_rate,
        "email_click_rate": email_click_rate, "total_spent": total_spent,
        "avg_order_value": avg_order_value, "discount_used": discount_used,
        "support_tickets": support_tickets, "refund_requested": refund_requested,
        "delivery_delay_days": delivery_delay, "satisfaction_score": satisfaction_score,
        "nps_score": nps_score, "marketing_spend_per_user": marketing_spend,
        "lifetime_value": lifetime_value, "last_3_month_purchase_freq": last_3month,
        "tenure_days": tenure_days, "days_since_last_purchase": days_since_last,
    }

    input_filtered = pd.DataFrame(0.0, index=[0], columns=selected_features)
    for col, val in input_dict.items():
        if col in selected_features:
            input_filtered[col] = val

    input_scaled = scaler.transform(input_filtered)
    pred         = model.predict(input_scaled)[0]
    pred_proba   = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.markdown("## 📈 Hasil Prediksi")

    # ── Baris atas: verdict + 2 metric ───────────────────────────────────────
    r1, r2, r3 = st.columns([3, 1, 1])

    with r1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-label" style="color:#ff4b4b;">⚠️ PELANGGAN BERPOTENSI CHURN</div>
                <div class="result-sub">Model memprediksi pelanggan ini akan <strong>berhenti</strong> menggunakan layanan.
                Segera lakukan tindakan retensi.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-label" style="color:#21c354;">✅ PELANGGAN AKAN BERTAHAN</div>
                <div class="result-sub">Model memprediksi pelanggan ini akan <strong>tetap</strong> menggunakan layanan.</div>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#21c354;">{pred_proba[0]*100:.1f}%</div>
            <div class="metric-lbl">Probabilitas Bertahan</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        color = "#ff4b4b" if pred_proba[1] > 0.5 else "#f0a500"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color};">{pred_proba[1]*100:.1f}%</div>
            <div class="metric-lbl">Probabilitas Churn</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Baris bawah: chart + rekomendasi ─────────────────────────────────────
    c1, c2 = st.columns([1, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        fig.patch.set_facecolor("#f8fafd")
        ax.set_facecolor("#f8fafd")
        labels = ["Bertahan (0)", "Churn (1)"]
        colors = ["#21c354", "#ff4b4b"]
        bars   = ax.barh(labels, pred_proba, color=colors, edgecolor="white",
                         height=0.45, zorder=3)
        ax.set_xlim(0, 1.25)
        ax.axvline(0.5, color="#999", linestyle="--", linewidth=1, label="Threshold 0.5", zorder=2)
        for bar, val in zip(bars, pred_proba):
            ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=12, fontweight="bold")
        ax.set_xlabel("Probabilitas", fontsize=10)
        ax.set_title("Distribusi Probabilitas", fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("### 💡 Rekomendasi Tindakan")
        if pred == 1:
            st.warning(
                "**Segera lakukan tindakan retensi:**\n\n"
                "🎁 Berikan penawaran diskon atau promo eksklusif\n\n"
                "📞 Hubungi pelanggan secara personal untuk feedback\n\n"
                "🔄 Tawarkan upgrade langganan dengan harga spesial\n\n"
                "📧 Kirim email re-engagement dengan konten relevan\n\n"
                "🎯 Prioritaskan dalam program loyalitas pelanggan"
            )
        else:
            st.success(
                "**Pelanggan ini loyal — pertahankan dengan:**\n\n"
                "⭐ Program reward untuk pelanggan setia\n\n"
                "📊 Pantau terus aktivitas dan kepuasan\n\n"
                "🛍️ Dorong cross-selling produk yang relevan\n\n"
                "📬 Kirim newsletter dengan konten bernilai"
            )