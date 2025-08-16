# Neural Blogs

A Jekyll blog focused on neural networks, AI, and machine learning interpretability research.

🌐 **Live Site**: [https://neuralblog.github.io](https://neuralblog.github.io)

## Overview

This blog explores various aspects of neural networks and AI, with a particular focus on interpretability research. The site features interactive visualizations and technical analysis of modern AI models.

## Features

- **Interactive Visualizations**: 3D Plotly charts embedded directly in blog posts with fullscreen capability
- **Responsive Design**: Clean, mobile-friendly layout using the Minima theme
- **GitHub Pages Ready**: Configured for automatic deployment via GitHub Pages
- **Source Code Integration**: Direct links to Python scripts used for generating visualizations

## Current Content

- **Gemma 3 270M Digit Embeddings**: Analysis of digit and word embeddings in Google's Gemma 3 270M model, featuring interactive 3D PCA visualizations comparing how digits (0-9) and their word representations (zero, one, etc.) are encoded in the model's embedding space

## Development

### Prerequisites

- Ruby (>= 2.5.0)
- Bundler gem

### Setup

```bash
# Install dependencies
bundle install

# Serve locally with auto-regeneration
bundle exec jekyll serve

# Build static files
bundle exec jekyll build
```

### Project Structure

- `_posts/` - Blog posts in Markdown format
- `assets/visualizations/` - Interactive HTML visualizations
- `py/` - Python scripts for generating visualizations
- `docs/` - Built static site (auto-generated)

### Adding New Posts

Create new posts in `_posts/` with the filename format: `YYYY-MM-DD-title.markdown`

Each post requires front matter:
```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD HH:MM:SS +TIMEZONE
categories: category1 category2
---
```

## Built With

- [Jekyll](https://jekyllrb.com/) - Static site generator
- [Minima](https://github.com/jekyll/minima) - Jekyll theme
- [Plotly](https://plotly.com/) - Interactive visualizations
- [GitHub Pages](https://pages.github.com/) - Hosting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.