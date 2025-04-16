# app.py  ────────────────────────────────────────────────────────────────
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle, re

from yahooquery import Ticker
import yahoo_fin.stock_info as si

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 0️⃣ 页面配置：宽屏模式
st.set_page_config(
    page_title="Peter Lynch Chat‑Bot & Screener",
    page_icon="💬",
    layout="wide"
)

# 0️⃣‑bis 自定义 CSS：右对齐用户消息 & 表格滚动
st.markdown(
    """
    <style>
      div[data-testid="stChatMessage"][data-owner="user"] > div {margin-left:auto;}
      div[data-testid="stChatMessage"][data-owner="user"] .stChatMessageAvatar {
        order:2; margin-left:0.5rem; margin-right:0;
      }
      div[data-testid="stChatMessage"][data-owner="user"] .stChatMessageContent {
        background-color:#0d6efd14;
      }
      .dataframe {max-height:300px; overflow:auto;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- ① Transformer 自定义组件 ----------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self._compute_positional_encoding(position, d_model)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"position": self.position, "d_model": self.d_model})
        return cfg

    def _compute_positional_encoding(self, position, d_model):
        positions = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        dims      = tf.range(d_model,   dtype=tf.float32)[tf.newaxis, :]
        angle_rads = positions / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        sines   = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_enc = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return tf.cast(pos_enc, tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, inputs):
        q,k,v,mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
        batch = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch)
        k = self.split_heads(self.wk(k), batch)
        v = self.split_heads(self.wv(v), batch)
        dk = tf.cast(self.depth, tf.float32)
        scaled = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        if mask is not None:
            scaled += mask * -1e9
        weights = tf.nn.softmax(scaled, axis=-1)
        out = tf.matmul(weights, v)
        out = tf.transpose(out, perm=[0,2,1,3])
        concat = tf.reshape(out, (batch, -1, self.num_heads * self.depth))
        return self.dense(concat)

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.maximum(look, create_padding_mask(x))

# ---------------- ② Screener 数据加载 & PCA + KMeans ----------------
@st.cache_data(ttl=86400)
def load_fundamentals(symbols):
    ticker = Ticker(symbols)
    fin = ticker.financial_data
    df = pd.DataFrame.from_dict(fin, orient="index")
    df.index.name = "symbol"; df.reset_index(inplace=True)
    df["equityToDebt"] = 1 / df.get("debtToEquity", pd.Series(1, index=df.index))
    return df.fillna(0)

def run_kmeans(df, n_clusters):
    # 保留数值列 & 去常量
    num_df = df.select_dtypes(include=[np.number]).loc[:, lambda d: d.nunique()>1].fillna(0)
    X0 = MinMaxScaler().fit_transform(num_df.values)
    # PCA 降到 2 维
    pca = PCA(n_components=2)
    X = pca.fit_transform(X0)
    # KMeans 聚类
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X0)
    df2 = df.copy(); df2["cluster"] = labels
    # Long/Short 推荐
    avgs = df2.groupby("cluster")["equityToDebt"].mean()
    long_c, short_c = int(avgs.idxmax()), int(avgs.idxmin())
    long_list  = df2[df2.cluster==long_c]["symbol"].tolist()
    short_list = df2[df2.cluster==short_c]["symbol"].tolist()
    # 簇特征摘要
    cluster_summary = (
        df2.groupby("cluster")
           .agg({"equityToDebt":"mean", "recommendationMean":"mean"})
           .rename(columns={
               "equityToDebt":"Avg Equity/Debt",
               "recommendationMean":"Avg Analyst Rating"
           })
    )
    # PCA loadings
    loadings = pd.DataFrame(pca.components_.T,
                            index=num_df.columns,
                            columns=["PC1","PC2"])
    return X, labels, df2, long_list, short_list, cluster_summary, loadings

# ---------------- ③ 加载模型 & tokenizer ----------------
@st.cache_resource
def load_assets():
    custom = {
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
        "create_padding_mask": create_padding_mask,
        "create_look_ahead_mask": create_look_ahead_mask,
    }
    model = tf.keras.models.load_model("model.h5", custom_objects=custom, compile=False)
    tok = pickle.load(open("tokenizer.pkl", "rb"))
    return model, tok, [tok.vocab_size], [tok.vocab_size+1]

model, tokenizer, START_TOKEN, END_TOKEN = load_assets()
MAX_LENGTH = 50

# ---------------- ④ ChatBot 推理（带上下文） ----------------
def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    return s.strip()

def chatbot_reply(user_input):
    ctx = st.session_state.get("last_analysis", "")
    prompt = f"{ctx}\nUser: {user_input}"
    enc = tf.expand_dims(START_TOKEN + tokenizer.encode(preprocess_sentence(prompt)) + END_TOKEN, 0)
    dec = tf.expand_dims(START_TOKEN, 0)
    for _ in range(MAX_LENGTH):
        preds = model([enc, dec], training=False)[:, -1:, :]
        pid = tf.cast(tf.argmax(preds, axis=-1), tf.int32)
        if tf.equal(pid, END_TOKEN[0]):
            break
        dec = tf.concat([dec, pid], axis=-1)
    return tokenizer.decode([i for i in tf.squeeze(dec, 0) if i < tokenizer.vocab_size])

# ---------------- ⑤ UI：Sidebar + 两列主界面 ----------------
# Sidebar
st.sidebar.header("🛠 Screener Settings")
default = si.tickers_dow()
symbols = st.sidebar.multiselect("Select tickers", default, default=default)
n_clusters = st.sidebar.slider("Clusters", 2, 10, 4)
if st.sidebar.button("Run Screener"):
    df = load_fundamentals(symbols)
    X, labels, df_cl, long_list, short_list, summary, loadings = run_kmeans(df, n_clusters)
    st.session_state.update({
        "screener_df": df_cl,
        "screener_X": X,
        "screener_labels": labels,
        "screener_k": n_clusters,
        "long_list": long_list,
        "short_list": short_list,
        "cluster_summary": summary,
        "pca_loadings": loadings,
        "last_analysis": (
            f"Cluster {summary['Avg Equity/Debt'].idxmax()} selected as Long (highest Avg Equity/Debt).\n"
            f"Long: {long_list}\nShort: {short_list}"
        )
    })

# 主区两列
col1, col2 = st.columns([2,1])

with col1:
    st.header("📊 Screener Dashboard")
    if "screener_df" in st.session_state:
        # 散点图
        X   = st.session_state["screener_X"]
        lbl = st.session_state["screener_labels"]
        k   = st.session_state["screener_k"]
        fig, ax = plt.subplots(figsize=(6,4))
        for c in range(k):
            idx = lbl==c
            ax.scatter(X[idx,0], X[idx,1], label=f"Cluster {c}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
        st.pyplot(fig, use_container_width=True)

        # PCA Loadings
        st.subheader("🔍 PCA Loadings")
        st.dataframe(st.session_state["pca_loadings"].style.background_gradient(axis=0),
                     use_container_width=True)

        # Cluster Summary
        st.subheader("📋 Cluster Summary")
        st.dataframe(st.session_state["cluster_summary"].style.background_gradient(axis=1),
                     use_container_width=True)

        # Key Financial Ratios
        st.subheader("🔢 Key Financial Ratios")
        st.dataframe(st.session_state["screener_df"], use_container_width=True)

        # Recommendations
        long_c = st.session_state["cluster_summary"]["Avg Equity/Debt"].idxmax()
        short_c= st.session_state["cluster_summary"]["Avg Equity/Debt"].idxmin()
        st.subheader(f"✔️ Long (Cluster {long_c})")
        st.write(st.session_state["long_list"])
        st.subheader(f"✖️ Short (Cluster {short_c})")
        st.write(st.session_state["short_list"])
    else:
        st.info("Use the sidebar to select tickers and run screener.")

with col2:
    st.header("💬 Chat with Peter Lynch Bot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"])
    if user_q := st.chat_input("Type your question…"):
        st.session_state.messages.append({"role":"user","avatar":"👤","content":user_q})
        st.chat_message("user", avatar="👤").write(user_q)
        with st.spinner("Thinking…"):
            ans = chatbot_reply(user_q)
        st.session_state.messages.append({"role":"assistant","avatar":"🤖","content":ans})
        st.chat_message("assistant", avatar="🤖").write(ans)
