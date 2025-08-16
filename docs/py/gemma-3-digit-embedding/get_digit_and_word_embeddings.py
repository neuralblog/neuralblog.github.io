import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

# Extract embeddings for digits 0-9
digits = "0123456789"
digit_tokens = tokenizer.encode(digits, add_special_tokens=False)

# Define word representations for digits
word_numbers = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
word_tokens = [
    tokenizer.encode(word, add_special_tokens=False)[0] for word in word_numbers
]

# Define colors for each digit-word pair
colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
embeds = model.model.embed_tokens.weight
digit_embeddings = embeds[digit_tokens].detach().numpy()
word_embeddings = embeds[word_tokens].detach().numpy()

# Combine digit and word embeddings for PCA
all_embeddings = np.vstack([digit_embeddings, word_embeddings])

# Apply PCA to reduce to 3D
pca = PCA(n_components=3)
all_embeddings_3d = pca.fit_transform(all_embeddings)

# Split back into digits and words
digit_embeddings_3d = all_embeddings_3d[:10]
word_embeddings_3d = all_embeddings_3d[10:]

# Create 3D interactive scatter plot with two traces
fig = go.Figure()

# Add digits trace
fig.add_trace(
    go.Scatter3d(
        x=digit_embeddings_3d[:, 0],
        y=digit_embeddings_3d[:, 1],
        z=digit_embeddings_3d[:, 2],
        mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=2, color="black")),
        text=list(digits),
        textposition="top center",
        textfont=dict(size=14, color="black"),
        hovertemplate="<b>Digit %{text}</b><br>"
        + "PC1: %{x:.3f}<br>"
        + "PC2: %{y:.3f}<br>"
        + "PC3: %{z:.3f}<br>"
        + "<extra></extra>",
        name="Digits",
        showlegend=False,
    )
)

# Add words trace
fig.add_trace(
    go.Scatter3d(
        x=word_embeddings_3d[:, 0],
        y=word_embeddings_3d[:, 1],
        z=word_embeddings_3d[:, 2],
        mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=2, color="black")),
        text=word_numbers,
        textposition="top center",
        textfont=dict(size=14, color="black"),
        hovertemplate="<b>Word %{text}</b><br>"
        + "PC1: %{x:.3f}<br>"
        + "PC2: %{y:.3f}<br>"
        + "PC3: %{z:.3f}<br>"
        + "<extra></extra>",
        name="Words",
        showlegend=False,
    )
)

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    width=None,
    autosize=True,
)

# Save as interactive HTML file with responsive configuration
html_string = fig.to_html(
    config={"responsive": True, "displayModeBar": True},
    include_plotlyjs="cdn",
    div_id="plotly-div",
)

# Add custom CSS for full responsiveness
custom_css = """
<style>
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
}
#plotly-div {
    height: 100vh;
    width: 100%;
}
</style>
"""

# Insert custom CSS before closing head tag
html_with_css = html_string.replace("</head>", custom_css + "</head>")

with open("digit_and_word_embed_3d.html", "w") as f:
    f.write(html_with_css)

print(f"Interactive 3D visualization saved as 'digit_and_word_embed_3d.html'")
print(
    f"Total explained variance (3 components): {sum(pca.explained_variance_ratio_):.2%}"
)
print(
    f"Individual variances: PC1={pca.explained_variance_ratio_[0]:.2%}, "
    f"PC2={pca.explained_variance_ratio_[1]:.2%}, PC3={pca.explained_variance_ratio_[2]:.2%}"
)
