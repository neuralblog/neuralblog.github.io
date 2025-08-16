# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based blog site using the Minima theme. The site is titled "Neural Blogs" and appears to be focused on technical content.

## Development Commands

- **Install dependencies**: `bundle install`
- **Serve locally**: `bundle exec jekyll serve` (launches web server with auto-regeneration)
- **Build site**: `bundle exec jekyll build`

## Project Structure

- `_config.yml` - Main Jekyll configuration file
- `_posts/` - Blog post content (Markdown files with YYYY-MM-DD-title.md format)
- `_site/` - Generated static site (auto-generated, don't edit directly)
- `index.markdown` - Homepage using the 'home' layout
- `about.markdown` - About page using the 'page' layout
- `Gemfile` - Ruby dependencies, includes Jekyll 4.4.1 and Minima theme

## Content Creation

Blog posts must be placed in `_posts/` with filename format: `YYYY-MM-DD-title.markdown`

Each post requires front matter with:
- `layout: post`
- `title: "Post Title"`
- `date: YYYY-MM-DD HH:MM:SS +TIMEZONE`
- `categories: space separated categories`

Pages use front matter with layout (`page` or `home`) and optionally `permalink` for custom URLs.
- always NOT use `—` in my blogs.