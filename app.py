import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import emoji
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from scipy.stats import zscore

# ------------------- Chat Parser -------------------
def parse_chat(chat_text):
    pattern = r"\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}:\d{2}\u202f(?:AM|PM|am|pm))\] (.*?): (.*)"
    data = []
    for line in chat_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            date, time, user, message = match.groups()
            data.append([date, time, user, message])
    df = pd.DataFrame(data, columns=['Date', 'Time', 'User', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    return df

# ------------------- Word Cloud -------------------
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)
    plt.close()

# ------------------- Daily Trend -------------------
def plot_message_trend(df):
    st.subheader("📅 Daily Message Timeline")
    daily_counts = df.groupby(df['Date'].dt.date).count()['Message']
    plt.figure(figsize=(12,5))
    daily_counts.plot()
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid()
    st.pyplot(plt)
    plt.close()

# ------------------- Busiest Times -------------------
def busiest_day_hour(df):
    def extract_hour(time_str):
        match = re.search(r'(\d{1,2}):', time_str)
        return int(match.group(1)) if match else None

    df['Hour'] = df['Time'].apply(extract_hour)
    df = df.dropna(subset=['Hour'])
    df['Day'] = df['Date'].dt.day_name()
    st.subheader("⏰ Busiest Hours and Days")

    fig, axes = plt.subplots(1, 2, figsize=(18,5))
    sns.countplot(ax=axes[0], x='Hour', data=df, palette='viridis')
    axes[0].set_title("Messages by Hour")

    sns.countplot(ax=axes[1], x='Day', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], palette='magma')
    axes[1].set_title("Messages by Day")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# ------------------- User Contribution -------------------
def user_contribution(df):
    st.subheader("👥 User Contribution")
    user_counts = df['User'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=user_counts.values, y=user_counts.index, palette='coolwarm')
    plt.title("Top 10 Most Active Users")
    st.pyplot(plt)
    plt.close()

# ------------------- Sentiment Analysis -------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def sentiment_analysis(df):
    st.subheader("📈 Sentiment Analysis (Transformer-based)")
    sentiment_pipeline = load_sentiment_model()
    messages = df['Message'].dropna().astype(str).tolist()
    sample_size = min(1000, len(messages))
    results = sentiment_pipeline(messages[:sample_size])

    sentiments = [r['label'] for r in results]
    df.loc[:sample_size-1, 'Sentiment'] = sentiments

    sentiment_counts = df['Sentiment'].value_counts()
    st.write("**Overall Sentiment Distribution:**")
    st.bar_chart(sentiment_counts)

    fig, ax = plt.subplots(figsize=(6, 4))
    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90, ax=ax)
    ax.set_ylabel('')
    ax.set_title("Sentiment Proportion")
    st.pyplot(fig)

    st.write("**Sentiment Over Time:**")
    df['DateOnly'] = df['Date'].dt.date
    sentiment_timeline = df.groupby(['DateOnly', 'Sentiment']).size().unstack().fillna(0)
    st.line_chart(sentiment_timeline)

# ------------------- Emoji Analysis -------------------
def emoji_analysis(df):
    st.subheader("😊 Emoji Analysis")
    emojis_list = []
    for message in df['Message']:
        emojis_list.extend([c for c in message if c in emoji.EMOJI_DATA])

    if emojis_list:
        emoji_freq = Counter(emojis_list).most_common(10)
        emoji_df = pd.DataFrame(emoji_freq, columns=['Emoji', 'Count'])

        st.write("**Top 10 Most Used Emojis:**")
        st.dataframe(emoji_df)

        plt.figure(figsize=(8, 4))
        sns.barplot(data=emoji_df, x='Emoji', y='Count', palette="plasma")
        plt.title("Top Emojis")
        plt.xlabel("Emoji")
        plt.ylabel("Count")
        st.pyplot(plt)
        plt.close()
    else:
        st.info("No emojis found in chat.")

# ------------------- Message Len + Outlier -------------------

def message_length_analysis(df):
    st.subheader("✉️ Message Length Analysis & Outlier Detection")

    # Compute message lengths
    df['MessageLength'] = df['Message'].astype(str).apply(len)

    # Basic stats and histogram
    st.write("**Basic Stats on Message Lengths:**")
    st.write(df['MessageLength'].describe())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['MessageLength'], bins=50, kde=True, color="teal")
    ax.set_title("Distribution of Message Lengths")
    ax.set_xlabel("Message Length")
    st.pyplot(fig)

    # Outlier detection using Z-score
    df['ZScore'] = zscore(df['MessageLength'])
    outliers = df[(df['ZScore'] > 3) | (df['ZScore'] < -3)]

    st.write(f"**Outliers Detected (Z-score > 3 or < -3): {len(outliers)}**")
    if not outliers.empty:
        st.dataframe(outliers[['Date', 'Time', 'User', 'Message', 'MessageLength']].sort_values(by='MessageLength', ascending=False))
    else:
        st.info("No significant outliers found based on message length.")

# ------------------- Network Graph Analysis -------------------

def interaction_graph(df):
    st.subheader("🔗 Network Graph of User Interactions")

    interactions = []
    for _, row in df.iterrows():
        sender = row['User']
        msg = row['Message']
        for receiver in df['User'].unique():
            if receiver != sender and receiver.lower() in msg.lower():
                interactions.append((sender, receiver))

    if not interactions:
        st.info("No direct interactions or mentions found to form a graph.")
        return

    # Count edge frequency
    edge_counts = {}
    for edge in interactions:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Filter weak edges
    filtered_edges = [e for e, count in edge_counts.items() if count >= 2]
    if not filtered_edges:
        st.warning("Graph too sparse after filtering low interactions.")
        return

    # Create graph
    G = nx.DiGraph()
    G.add_edges_from(filtered_edges)

    # Node centrality for color scaling
    centrality = nx.degree_centrality(G)
    node_colors = [centrality[n] for n in G.nodes]

    # Edge width based on frequency
    edge_weights = [edge_counts[e] for e in G.edges]

    # Layout and draw
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray', arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")

    plt.title("Cleaned Interaction Network", fontsize=14)
    plt.axis("off")
    st.pyplot(plt)
    plt.close()

    # Centrality table
    st.write("**Top Influencers (by Degree Centrality):**")
    top_users = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    st.table(pd.DataFrame(top_users, columns=['User', 'Centrality Score']))


# ------------------- Streamlit App -------------------
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📊 WhatsApp Chat Analyzer")
st.write("Upload your exported WhatsApp chat (.txt) to see detailed analysis:")

uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type="txt")

if uploaded_file is not None:
    chat_text = uploaded_file.read().decode("utf-8")
    chat_df = parse_chat(chat_text)

    if chat_df.empty:
        st.error("Could not parse any messages. Check if the file format is correct.")
        st.stop()
    else:
        st.success(f"Parsed {len(chat_df)} messages successfully.")

        # --- Sidebar: Analysis Choice ---
        st.sidebar.title("⚙️ Analysis Settings")
        analysis_type = st.sidebar.radio(
            "Choose Analysis Type:",
            ["Whole Group Chat", "Individual Participant(s)"]
        )

        if analysis_type == "Individual Participant(s)":
            unique_users = sorted(chat_df['User'].unique())
            selected_users = st.sidebar.multiselect("Select participant(s):", unique_users)

            if not selected_users:
                st.warning("Please select at least one participant to continue.")
                st.stop()

            chat_df = chat_df[chat_df['User'].isin(selected_users)]
            st.sidebar.markdown(f"**Selected:** {', '.join(selected_users)}")
        else:
            st.sidebar.markdown("**Selected:** All Participants")

        st.write(f"**Total Messages Analyzed:** {len(chat_df)}")
        st.write(f"**Participants in Selection:** {chat_df['User'].nunique()}")

        # --- Sidebar Navigation ---
        st.sidebar.markdown("---")
        st.sidebar.title("🧭 Navigation")
        section = st.sidebar.radio("Select a feature to view:", [
            "Overview", "Daily Message Trend", "Busiest Times",
            "User Contribution", "Sentiment Analysis", "Emoji Analysis","Message Length & Outliers",
            "Network Graph"
        ])

        all_text = " ".join(chat_df['Message'])

        # --- Main Display Based on Selected Section ---
        if section == "Overview":
            st.header("📌 Overview & Word Cloud")
            generate_wordcloud(all_text, "Overall Chat Word Cloud")

        elif section == "Daily Message Trend":
            plot_message_trend(chat_df)

        elif section == "Busiest Times":
            busiest_day_hour(chat_df)

        elif section == "User Contribution":
            user_contribution(chat_df)

        elif section == "Sentiment Analysis":
            sentiment_analysis(chat_df)

        elif section == "Emoji Analysis":
            emoji_analysis(chat_df)

        elif section == "Network Graph":
            interaction_graph(chat_df)

        elif section == "Message Length & Outliers":
            message_length_analysis(chat_df)


        # Optional: Download processed chat data
        csv = chat_df.to_csv(index=False).encode('utf-8')
        st.sidebar.markdown("---")
        st.sidebar.download_button("📥 Download CSV", csv, "chat_data.csv", "text/csv")
else:
    st.info("Please upload a WhatsApp .txt file to begin analysis.")
