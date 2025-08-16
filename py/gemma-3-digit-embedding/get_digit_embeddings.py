import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

# Extract embeddings for digits 0-9
digits = "0123456789"
digit_tokens = tokenizer.encode(digits, add_special_tokens=False)
embeds = model.model.embed_tokens.weight
digit_embeddings = embeds[digit_tokens].detach().numpy()

# Apply PCA to reduce to 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(digit_embeddings)

# Create 3D interactive scatter plot
fig = go.Figure(data=go.Scatter3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    mode='markers+text',
    marker=dict(
        size=12,
        color=list(range(10)),
        colorscale='rainbow',
        showscale=False,
        line=dict(width=2, color='black')
    ),
    text=list(digits),
    textposition="middle center",
    textfont=dict(size=14, color='white'),
    hovertemplate='<b>Digit %{text}</b><br>' +
                  'PC1: %{x:.3f}<br>' +
                  'PC2: %{y:.3f}<br>' +
                  'PC3: %{z:.3f}<br>' +
                  '<extra></extra>',
    name='Digits'
))

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        bgcolor='rgba(0,0,0,0)',
        xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='lightgray'),
        yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='lightgray'),
        zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='lightgray')
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    width=None,  # Remove fixed width for responsiveness
    autosize=True  # Enable autosize for responsive behavior
)

# Save as interactive HTML file with responsive configuration
html_string = fig.to_html(config={'responsive': True, 'displayModeBar': True},
                         include_plotlyjs='cdn',
                         div_id="plotly-div")

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
html_with_css = html_string.replace('</head>', custom_css + '</head>')

with open("digit_embed_3d.html", "w") as f:
    f.write(html_with_css)

print(f"Interactive 3D visualization saved as 'digit_embed_3d.html'")
print(f"Total explained variance (3 components): {sum(pca.explained_variance_ratio_):.2%}")
print(f"Individual variances: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}, PC3={pca.explained_variance_ratio_[2]:.2%}")