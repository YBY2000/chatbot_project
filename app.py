# app.py  ────────────────────────────────────────────────────────────────
import streamlit as st
import tensorflow as tf
import pickle, re

st.set_page_config(page_title="Peter Lynch Chat‑Bot", page_icon="💬")

# ---------------- ① 必要的自定义层 / 函数 ----------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = self._positional_encoding(position, d_model)

    @staticmethod
    def _positional_encoding(position, d_model):
        angle_rads = PositionalEncoding._get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model,   dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    @staticmethod
    def _get_angles(pos, i, d_model):
        return pos / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

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
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
        batch = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch)
        k = self.split_heads(self.wk(k), batch)
        v = self.split_heads(self.wv(v), batch)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        if mask is not None:
            scaled += mask * -1e9
        weights = tf.nn.softmax(scaled, axis=-1)
        output = tf.matmul(weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(output, (batch, -1, self.num_heads * self.depth))
        return self.dense(concat)

# —— 两个掩码函数（Lambda 层反序列化时需要） ——
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.maximum(look_ahead, create_padding_mask(x))
# -----------------------------------------------------------------------

# ---------------- ② 加载模型 / tokenizer（带缓存） ----------------
@st.cache_resource
def load_assets():
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
        "create_padding_mask": create_padding_mask,
        "create_look_ahead_mask": create_look_ahead_mask,
    }
    model = tf.keras.models.load_model("model.h5",
                                       custom_objects=custom_objects,
                                       compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    start, end = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    return model, tokenizer, start, end

model, tokenizer, START_TOKEN, END_TOKEN = load_assets()
MAX_LENGTH = 50  # 与训练时保持一致

# ---------------- ③ 预处理 + 生成回复 ----------------
def preprocess_sentence(sentence: str) -> str:
    s = sentence.lower().strip()
    s = re.sub(r"([?.!,])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    return s.strip()

def predict(sentence: str) -> str:
    sentence = preprocess_sentence(sentence)
    encoder_input = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, 0)
    decoder_input = tf.expand_dims(START_TOKEN, 0)

    for _ in range(MAX_LENGTH):
        preds = model([encoder_input, decoder_input], training=False)
        preds = preds[:, -1:, :]
        pred_id = tf.cast(tf.argmax(preds, axis=-1), tf.int32)
        if tf.equal(pred_id, END_TOKEN[0]):
            break
        decoder_input = tf.concat([decoder_input, pred_id], axis=-1)

    decoded = tokenizer.decode([i for i in tf.squeeze(decoder_input, 0) if i < tokenizer.vocab_size])
    return decoded

# ---------------- ④ Streamlit UI ----------------
# st.set_page_config(page_title="Peter Lynch Chat‑Bot", page_icon="💬")
st.title("💬 Peter Lynch Investment Q&A")
st.markdown("Ask anything about Peter Lynch’s investment philosophy.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 输入框
if prompt := st.chat_input("Type your question here…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking…"):
        answer = predict(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
# --------------------------------------------------------------------
