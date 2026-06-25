import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# ── Load model, scaler, dan fitur ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("best_model_rf.joblib")
    scaler   = joblib.load("scaler.joblib")
    features = joblib.load("selected_features.joblib")
    return model, scaler, features

model, scaler, selected_features = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Aplikasi prediksi **churn pelanggan** menggunakan model "
    "**Random Forest** yang dioptimalkan dengan Hyperparameter Tuning "
    "pada dataset *Sales and Marketing Customer*."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.info(
        "**Model:** Random Forest (Ensemble Bagging)\n\n"
        "**Skenario:** Hyperparameter Tuning\n\n"
        "**Performa pada data uji:**\n"
        "- Accuracy  : 0.8573\n"
        "- Precision : 0.5235\n"
        "- Recall    : 0.7761\n"
        "- F1-Score  : 0.6252"
    )
    st.header("📖 Panduan Penggunaan")
    st.markdown(
        "1. Isi seluruh form input di bawah\n"
        "2. Klik tombol **Prediksi Churn**\n"
        "3. Lihat hasil prediksi dan probabilitas\n\n"
        "> **Churn = 1** → Pelanggan berpotensi berhenti\n\n"
        "> **Churn = 0** → Pelanggan diprediksi bertahan"
    )
    st.header("📌 Penjelasan Fitur Penting")
    st.markdown(
        "- **Tenure Days**: Lama pelanggan berlangganan\n"
        "- **Days Since Last Purchase**: Hari sejak transaksi terakhir\n"
        "- **Satisfaction Score**: Skor kepuasan 0–10\n"
        "- **NPS Score**: Net Promoter Score\n"
        "- **Total Spent**: Total pengeluaran pelanggan\n"
        "- **Support Tickets**: Jumlah tiket bantuan yang dibuka"
    )

# ── Form Input ────────────────────────────────────────────────────────────────
st.subheader("🔢 Input Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📋 Data Demografis & Akun**")
    age             = st.number_input("Usia", min_value=18, max_value=100, value=35)
    is_premium_user = st.selectbox("Premium User", [0, 1],
                                    format_func=lambda x: "Ya ✓" if x == 1 else "Tidak ✗")
    tenure_days     = st.number_input("Lama Berlangganan (hari)", min_value=0, max_value=2000, value=365)
    days_since_last = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0, max_value=1000, value=30)
    last_3month     = st.number_input("Frekuensi Beli 3 Bulan Terakhir", min_value=0, max_value=50, value=3)

with col2:
    st.markdown("**💰 Aktivitas & Transaksi**")
    total_visits      = st.number_input("Total Kunjungan", min_value=0, max_value=500, value=20)
    avg_session_time  = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0, max_value=120.0, value=10.0, step=0.5)
    pages_per_session = st.number_input("Halaman per Sesi", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    total_spent       = st.number_input("Total Pengeluaran", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)
    avg_order_value   = st.number_input("Rata-rata Nilai Transaksi", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
    lifetime_value    = st.number_input("Lifetime Value", min_value=0.0, max_value=100000.0, value=1000.0, step=100.0)
    discount_used     = st.selectbox("Gunakan Diskon?", [0, 1],
                                      format_func=lambda x: "Ya ✓" if x == 1 else "Tidak ✗")

with col3:
    st.markdown("**📧 Email, Kepuasan & Layanan**")
    email_open_rate    = st.slider("Email Open Rate", 0.0, 1.0, 0.3, step=0.01)
    email_click_rate   = st.slider("Email Click Rate", 0.0, 1.0, 0.1, step=0.01)
    satisfaction_score = st.slider("Skor Kepuasan", 0.0, 10.0, 7.0, step=0.1)
    nps_score          = st.slider("NPS Score", -100, 100, 30)
    support_tickets    = st.number_input("Jumlah Tiket Support", min_value=0, max_value=20, value=1)
    refund_requested   = st.selectbox("Pernah Request Refund?", [0, 1],
                                       format_func=lambda x: "Ya ✓" if x == 1 else "Tidak ✗")
    delivery_delay     = st.number_input("Keterlambatan Pengiriman (hari)", min_value=0, max_value=30, value=0)
    marketing_spend    = st.number_input("Marketing Spend per User", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)

# ── Prediksi ──────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Prediksi Churn", type="primary", use_container_width=True)

if predict_btn:
    # Susun input
    input_dict = {
        "age": age,
        "is_premium_user": is_premium_user,
        "total_visits": total_visits,
        "avg_session_time": avg_session_time,
        "pages_per_session": pages_per_session,
        "email_open_rate": email_open_rate,
        "email_click_rate": email_click_rate,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "discount_used": discount_used,
        "support_tickets": support_tickets,
        "refund_requested": refund_requested,
        "delivery_delay_days": delivery_delay,
        "satisfaction_score": satisfaction_score,
        "nps_score": nps_score,
        "marketing_spend_per_user": marketing_spend,
        "lifetime_value": lifetime_value,
        "last_3_month_purchase_freq": last_3month,
        "tenure_days": tenure_days,
        "days_since_last_purchase": days_since_last,
    }

    # Filter ke fitur yang digunakan model
    input_filtered = pd.DataFrame(0.0, index=[0], columns=selected_features)
    for col, val in input_dict.items():
        if col in selected_features:
            input_filtered[col] = val

    # Scaling & prediksi
    input_scaled = scaler.transform(input_filtered)
    pred         = model.predict(input_scaled)[0]
    pred_proba   = model.predict_proba(input_scaled)[0]

    # ── Tampilan Hasil ────────────────────────────────────────────────────────
    st.subheader("📈 Hasil Prediksi")
    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

    with res_col1:
        if pred == 1:
            st.error("## ⚠️ PELANGGAN BERPOTENSI CHURN")
            st.markdown("Model memprediksi pelanggan ini **akan berhenti** menggunakan layanan. "
                        "Segera lakukan tindakan retensi.")
        else:
            st.success("## ✅ PELANGGAN AKAN BERTAHAN")
            st.markdown("Model memprediksi pelanggan ini **akan tetap** menggunakan layanan.")

    with res_col2:
        st.metric(label="Probabilitas Bertahan", value=f"{pred_proba[0]*100:.1f}%")

    with res_col3:
        st.metric(label="Probabilitas Churn", value=f"{pred_proba[1]*100:.1f}%",
                  delta=f"{'Risiko Tinggi' if pred_proba[1] > 0.5 else 'Risiko Rendah'}",
                  delta_color="inverse")

    # Visualisasi probabilitas
    fig, ax = plt.subplots(figsize=(7, 2.5))
    labels = ["Bertahan (0)", "Churn (1)"]
    colors = ["#55A868", "#DD8452"]
    bars   = ax.barh(labels, pred_proba, color=colors, edgecolor="white", height=0.45)
    for bar, val in zip(bars, pred_proba):
        ax.text(min(val + 0.02, 1.05), bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Probabilitas", fontsize=10)
    ax.set_title("Distribusi Probabilitas Prediksi", fontsize=12, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 0.5")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Rekomendasi aksi
    st.subheader("💡 Rekomendasi Tindakan")
    if pred == 1:
        st.warning(
            "**Tindakan yang disarankan untuk mencegah churn:**\n"
            "- 🎁 Berikan penawaran diskon atau promo eksklusif\n"
            "- 📞 Hubungi pelanggan secara personal untuk feedback\n"
            "- 🔄 Tawarkan upgrade langganan dengan harga spesial\n"
            "- 📧 Kirim email re-engagement dengan konten yang relevan\n"
            "- 🎯 Prioritaskan dalam program loyalitas pelanggan"
        )
    else:
        st.info(
            "**Pelanggan ini loyal — pertahankan dengan:**\n"
            "- ⭐ Program reward untuk pelanggan setia\n"
            "- 📊 Pantau terus aktivitas dan kepuasan\n"
            "- 🛍️ Dorong cross-selling produk relevan"
        )
