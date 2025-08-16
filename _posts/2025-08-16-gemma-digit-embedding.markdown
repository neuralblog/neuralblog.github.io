---
layout: post
title:  "Gemma 3 270M's Digit Embeddings"
date:   2025-08-16 13:55:31 +0800
categories: interpretability
---

Google released a tiny Gemma model recently, [Gemma 3 270M][google-gemma-3-270-blog]. What's interesting about the size of this model is that out of 270 million parameters, 170 million of them are embedding vectors, and only 100 million parameters are for transformer blocks. It is uncontroversial to say... a lot of information is encoded in these embedding vectors.

So, naturally, I want to take a quick look at these vectors from an interpretability perspective. The easy target is digit embeddings. Unlike Llama models, Gemma models simply split numbers into digits. For example, the number `123` is split into three tokens: `1`, `2`, and `3`. 

This makes things simple for us: we have 10 tokens to represent the 10 digits.

Here is how we get these 10 token embeddings:

{% highlight python %}
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

# Extract embeddings for digits 0-9
digits = "0123456789"
digit_tokens = tokenizer.encode(digits, add_special_tokens=False)
embeds = model.model.embed_tokens.weight
digit_embeddings = embeds[digit_tokens].detach().numpy()
{% endhighlight %}

Then, I do a simple PCA projection of these embeddings to a 3D space. Here are the results:

<div id="viz1-container" style="position: relative; border: 1px solid #ddd; border-radius: 4px;">
<button onclick="toggleFullscreen('viz1-container')" style="position: absolute; left: 8px; top: 8px; z-index: 10; background: rgba(255,255,255,0.8); border: 1px solid #ccc; border-radius: 3px; padding: 4px 6px; cursor: pointer; font-size: 12px;" title="Toggle Fullscreen">⛶</button>
<iframe src="/assets/visualizations/digit_embed_3d.html" width="100%" height="400px" frameborder="0" scrolling="no"></iframe>
</div>

*Click the button in the top left corner to see the plot in fullscreen.*

Playing with the visualization a bit, we can see that the number zero is separate from the others (unsurprisingly), and the other numbers seem to form a circle in the natural order of the digits.

How about plotting the words `zero`, `one`, ..., `nine` as well? I did that and here are the results:

<div id="viz2-container" style="position: relative; border: 1px solid #ddd; border-radius: 4px;">
<button onclick="toggleFullscreen('viz2-container')" style="position: absolute; left: 8px; top: 8px; z-index: 10; background: rgba(255,255,255,0.8); border: 1px solid #ccc; border-radius: 3px; padding: 4px 6px; cursor: pointer; font-size: 12px;" title="Toggle Fullscreen">⛶</button>
<iframe src="/assets/visualizations/digit_and_word_embed_3d.html" width="100%" height="400px" frameborder="0" scrolling="no"></iframe>
</div>

*Click the button in the top left corner to see the plot in fullscreen.*

[google-gemma-3-270-blog]: https://developers.googleblog.com/en/introducing-gemma-3-270m/

Notice how the word embeddings are (somewhat) parallel to the digit embeddings and follow the same structure as the digit embeddings.

The complete source code for generating these visualizations is available:

- [Digit embeddings visualization](https://github.com/neuralblog/neuralblog.github.io/blob/main/py/gemma-3-digit-embedding/get_digit_embeddings.py)
- [Digit and word embeddings comparison](https://github.com/neuralblog/neuralblog.github.io/blob/main/py/gemma-3-digit-embedding/get_digit_and_word_embeddings.py)

<script>
function toggleFullscreen(elementId) {
  const element = document.getElementById(elementId);
  
  if (!document.fullscreenElement) {
    // Enter fullscreen
    if (element.requestFullscreen) {
      element.requestFullscreen();
    } else if (element.webkitRequestFullscreen) {
      element.webkitRequestFullscreen();
    } else if (element.msRequestFullscreen) {
      element.msRequestFullscreen();
    }
    // Style adjustments for fullscreen
    element.style.position = 'fixed';
    element.style.top = '0';
    element.style.left = '0';
    element.style.width = '100vw';
    element.style.height = '100vh';
    element.style.zIndex = '9999';
    element.style.backgroundColor = 'white';
    const iframe = element.querySelector('iframe');
    if (iframe) {
      iframe.style.height = '100%';
    }
  } else {
    // Exit fullscreen
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
    // Reset styles
    element.style.position = 'relative';
    element.style.top = 'auto';
    element.style.left = 'auto';
    element.style.width = 'auto';
    element.style.height = 'auto';
    element.style.zIndex = 'auto';
    element.style.backgroundColor = 'transparent';
    const iframe = element.querySelector('iframe');
    if (iframe) {
      iframe.style.height = '400px';
    }
  }
}

// Handle ESC key and fullscreen change events
document.addEventListener('fullscreenchange', function() {
  if (!document.fullscreenElement) {
    // Reset all containers when exiting fullscreen
    ['viz1-container', 'viz2-container'].forEach(id => {
      const element = document.getElementById(id);
      if (element) {
        element.style.position = 'relative';
        element.style.top = 'auto';
        element.style.left = 'auto';
        element.style.width = 'auto';
        element.style.height = 'auto';
        element.style.zIndex = 'auto';
        element.style.backgroundColor = 'transparent';
        const iframe = element.querySelector('iframe');
        if (iframe) {
          iframe.style.height = '400px';
        }
      }
    });
  }
});
</script>