---
title: Assignment 14 - Generative Pre-Trained Transformers (GPTs) Part 3
toc_sticky: true 
toc_h_max: 1
layout: problemset
---

# Learning Objectives

{% capture content %}
* How to generalize from single-headed to multi-headed attention
* How to Interleave attention and MLPs to create a transformer
* Understand the importance of skip connections and layer normalization
* Perform an ablation experiment to understand the parts of the NanoGPT model that are relevant to text generation
* Consider issues in dataset collection and curation for training and LLM
{% endcapture %}
{% include learning_objectives.html content=content %}

# Optional: Exploring AI, Building Equity into Innovation at Wellesley on Sunday
Sunday, November 3, 2024, 3-5pm  
Wellesley College, Tishman Commons  
Register here: https://forms.gle/C5REwqZ4B5f7Jx5y6  

Please join us for the 2024 World of Wellesley Community Book Read of Unmasking AI: My Mission to Protect What Is Human in a World of Machines by Joy Buolamwini. In her research, Dr. Joy Buolamwini uncovered what she calls “the coded gaze”, evidence of encoded discrimination and exclusion in tech products. How can we benefit from AI and also build equity into innovation? Come hear from a panel of Artificial Intelligence experts including:  
Aaron Pressman, The Boston Globe, Technology Industry Journalist  
Tanuj Barman, WHS student class of 2025, Co-founder nextgenkids.ai  
Carolyn Anderson, Wellesley College, Asst. Professor of Computer Science  
Michael Dupin, Wolters Kluwer, Director Technology - Generative AI, Merrimack  
College, Data Science Adjunct Professor  
Zachary Ziegler, Co-founder and CTO, OpenEvidence  



# Review of What We've Done So Far
Before getting into some new stuff, let's review what we did in assignments 12 and 13.
* We learned that GPT stands for "Generative Pre-trained Transform"
* A GPT model consists of a pipeline of interleaving two major types of layers: attention and MLPs.
* The attention layers are responsible for allowing tokens to pass information to other tokens. The degree to which a token passes information to another token depends on taking a dot product between a key and query vector, which is then passed through a softmax.  The specific value passed to the other token depends on a value vector which is computed from the input to the attention layer multiplied by a matrix ($W_V$).
* While we haven't gotten our hands dirty with MLPs in this module, we've seen them in previous parts of the course.  The MLPs in a GPT take the output of the attention block and perform computation on them.  In the 3B1B video, we saw that one theory of what these MLPs are doing is that they are representing facts that the GPT has learned.
* We started to implement NanoGPT by starting with a simple Bigram model and then adding in a self-attention mechanism so that tokens could communicate with each other.


# Finishing Our Implementation of NanoGPT

We'll continue to work through Karpathy's video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).  Follow along with our notes and suggestions for things to try below.

{% capture external %}
* [1:11:37](https://youtu.be/kCc8FmEb1nY?t=4297): this is our starting point for this assignment.
* [1:12:08](https://youtu.be/kCc8FmEb1nY?t=4328): Karpathy is now linking the concept of attention to a more general idea of information flowing between nodes in a graph.  We don't think you need to be too concerned about this concept as we haven't learned the necessary background to think about graphs (although, you may have seen this in DSA, FOCS, or Discrete).
* [1:15:41](https://youtu.be/kCc8FmEb1nY?t=4541): self-attention is not the only type of attention (as has we've already heard about, e.g., cross attention is used in language translation tasks). This could be an interesting thing to explore in a final project if you find this concept interesting.
* [1:16:56](https://youtu.be/kCc8FmEb1nY?t=4616): we are now seeing why the normalization term $\frac{1}{\sqrt{d_k}}$ is needed.  Karpathy does a nice job showing that this term allows us to achieve the variance we want (this is called *scaled attention*).
* [1:19:18](https://youtu.be/kCc8FmEb1nY?t=4758): we can now take our self-attention code and package it up into the class ``Head``.  As we've seen before in this class, in ``pytorch`` you can create your own machine learning modules by inheriting from ``nn.Module`` (e.g., as we did with our ``MLP`` implementation).  In this part of the video, we also modify our text generation code, which you don't need to worry about.
* [1:21:59](https://youtu.be/kCc8FmEb1nY?t=4919): we'll now scale up from a single head of attention to multiple heads of attention.  Notice the use of the ``nn.ModuleList`` class, which allows multiple ``nn.Module`` objects to be grouped together into a single list.  The key idea here is that our query dimension $n_q$ and the space that our value vectors live in (also $n_q$) is now different than the number of embedding dimensions.  We concatenate the output from each attention head together to get back to the same number of dimensions as our original embedding.  Karpathy makes a reference to this idea of convolutions, which we'll learn about in the next module of this class.
* [1:24:27](https://youtu.be/kCc8FmEb1nY?t=5067): now we are going to bring in the concept of the multi-layer perceptron.  Based on the 3B1B videos, we have a conceptual idea of where these MLPs fit in and what they might do (e.g., store facts that the LLM has learned).  For the MLPs in our model, we'll follow a pretty similar implementation to what we've done previously in the course.  Initially, the MLP that Karpathy implements will look a little strange (it will be a linear layer followed by a non-linearity with no subsequent linear layer), but eventually the second linear layer will be added (matching what we did in the previous module).  Karpathy also abstracts the sequence of self-attention and an MLP into a block which can be reused / repeated.
* [1:27:59](https://youtu.be/kCc8FmEb1nY?t=5279): now we will introduce the idea of skip connections (or residual connections).  There are many reasons why this helps with the performance of the network, which the video touches upon.  The 3B1B videos give us one more way to think about this.  In those videos we talk about self-attention computing a vector that we can add onto our original embedding to modify a word's meaning in some way.  Up until now, we have actually used attention to completely overwrite the original embedding. These skip connections allow us to, instead, compute a vector that we add to our embedding to get our output.  We'll be seeing in more detail how important these connections are later in this assignment.  The concept of the projection self-attention / MLP block back into the residual pathway is confusing.  As with most matrices in neural networks, we can add this project matrix to give our network a bit more flexibility in how it integrates the results of self-attention / MLP with the original embedding.
* [1:32:56](https://youtu.be/kCc8FmEb1nY?t=5576): next we are going to meet the concept of layer norm (this is [the link to the documentation page he pulls up on layer norm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)).  The explanation given here is not particular accessible since we didn't go through the original video on batch norm that Karpathy references.  For our purposes we can understand that layer norm is a way of standardizing the inputs to various parts of our model.  Given a batch of data, we would like each of the input features of our data to have mean 0 and standard deviation 1.  This standardization is achieved with ``LayerNorm``, which builds on some additional bells and whistles that we don't really need to worry about.  This sort of normalization can significantly improve the performance of deep (meaning with lots of layers) neural networks.
* [1:37:57](https://youtu.be/kCc8FmEb1nY?t=5877): now we'll have some fun scaling up our network!
* [1:38:46](https://youtu.be/kCc8FmEb1nY?t=5926): we touch on the idea of dropout, which we discussed a bit in our class on preventing overfitting.
* [1:42:40](https://youtu.be/kCc8FmEb1nY?t=6160): don't worry about this part.  We are just connecting back to the "Attention is All You Need" paper with its focus on cross-attention.
* [1:46:31](https://youtu.be/kCc8FmEb1nY?t=6391): Karpathy walks through of the NanoGPT repo.  The quick summary is that some changes have been made to clean up the code and make it more efficient.
* [1:48:55](https://youtu.be/kCc8FmEb1nY?t=6535): Karpathy talks about some important steps that would happen after the pre-training step that we've learned about if you were going to train a ChatGPT-like system.  This is fascinating stuff, and it could be great fodder for a final project!
{% endcapture %}
{% include external_resources.html content=external %}

# Visualizing NanoGPT and Connecting to NanoGPT

{% capture prob %}
There are some fantastic visualizations of LLMs out there.  Please check out [this visualization](https://bbycroft.net/llm), which shows the structure of the model from the video we just watched.  The visualizer also allows you to step through the main steps of the model and has some explanations of what's going on as well as animations that show the computations happening at each stage.

* Please step through the visualizations and try to link what you are seeing to Karpathy's video.  Take some notes about anything that you don't understand.
* Below we have reproduced a selection of ``model.py``, which defines the NanoGPT model.  Try to find as many pieces of the visualization of NanoGPT in the code for ``model.py``.  For example, you might determine which class implements a particular box in the visualization.

{% highlight python linenos %}
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
{% endhighlight %}
{% endcapture %}
{% capture sol %}
Here are a few notes to get you started (more could be said).
* The class ``GPT`` is the overall model shown in the visualization
* All of the ``V Output``s are computed on line 54 (doing this all at once instead of once per head is an optimization that Karpathy did for computational reasoning).
* The ``Attention Output`` is computed on line 58 (notice the projection there)
* The layer norms in the visualization are shown at a few lines in the code (e.g., 87 and 88).
* The ``Attention Residual`` is computed on line 87.
* The input embeddings are computed on lines 138 and 139.
* etc.
{% endcapture %}
{% include problem.html problem=prob solution=sol %}

# Ablation and NanoGPT

An [ablation experiment in machine learning](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)#:~:text=In%20artificial%20intelligence%20(AI)%2C,resultant%20performance%20of%20the%20system.) seeks to "to determine the contribution of a component to an AI system by removing the component, and then analyzing the resultant performance of the system." (Wikipedia).  We think that this is a particularly interesting idea to apply to the NanoGPT model.  We saw, as the model was being built up, that adding on new features seemed to improve performance.  Now that we have the entire model built, we will take away several aspects of the model and analyze the change in performance.  This can give us a sense for how important each aspect of the model is to the overall functioning of the system.

{% capture prob %}
{% capture parta %}
Describe how you would modify the excerpt from ``model.py`` shown in Exercise 1 to remove each of the following components from the model.
1. Remove the residual (or skip) connections from the self-attention and MLP steps.
2. Remove the layer norms from the self-attention and MLP steps.
3. Remove the position embedding
4. Use a head size of 1 (instead of multiheaded attention)
{% endcapture %}
{% capture partasol %}
<ol>
<li>Lines 87-88 would become
{% highlight python %}
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
{% endhighlight %}
</li>
<li>Lines 87-88 would become
{% highlight python %}
        x = x + self.attn(x)
        x = x + self.mlp(x)
{% endhighlight %}
</li>
<li>Line 140 would become
{% highlight python %}
        x = self.transformer.drop(tok_emb)
{% endhighlight %}
</li>
<li>There are a few places you could introduce this.  An easy way is to have Line 24 become
{% highlight python %}
        self.n_head = 1
{% endhighlight %}
</li>
</ol>
{% endcapture %}
{% include problem_part.html subpart=parta solution=partasol label="A" %}
{% capture partb %}
We went ahead and [performed the ablation experiments described](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment14.ipynb) above (removing each of the aforementioned components of the model, indpendently, and then training the model on the Shakespeare character-level dataset).  We'd like you to look at our results and provide your interpretation of the results.  What have you learned about the model from these experiments?  For example, what model components are the most important?

> ***Optional:*** If you'd like to run these ablation experiments yourself, you can do so either in your own environment or on Colab.  If you do this on Colab, we highly recommend you upgrade to Colab Pro (details on reimbursement for this are on Canvas) and use an L4 or an A100 GPU runtime when training.  We've made [a starter notebook for you to build from](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_assignment14_optional.ipynb).
{% endcapture %}
{% capture partbsol %}
The experiments show that the skip connections are tremendously important.  Without them the model loss is quite bad.  The model seems to converge to a good solution even without layer norm (although it takes longer to get there).  Not having position embedding seems to be detrimental (the loss never gets as low as the unablated model).  Having just one head of attention also does suprisingly well.  It's probably best not to extrapolate too much from these results.  These features might be more important on a larger dataset.
{% endcapture %}
{% include problem_part.html subpart=partb solution=partbsol label="B" %}
{% endcapture %}
{% include problem_with_parts.html problem=prob %}

# Proposing an LLM for an Application and Context You Care About

{% capture problem %}
Before closing out this module, we'd like to give some creative space to think through how LLMs might apply in some context you care about.  Please think respond to the following prompts. 
* What do you think is an interesting application of of LLMs (either at Olin or in some other context you care about).  You can choose something you think is positive, negative, or neutral (no judgment).  Describe your chosen application.  What value does it create and for whom?
* If you were to develop such an application, at a high-level how would you come about doing so.  Some areas to focus on could be dataset collection and curation and model evaluation and testing.  When sourcing your data to train your model, how would you navigate risks of data privacy, legal / regulator compliance, avoiding model bias, while achieving good performance with respect to the application you've chosen.  What guardrails would you need to put in place to make sure your system is not used in a harmful way that, presumably, you did not intend.  These guardrails could be technical in nature or specific licensing conditions you would impose on your system.
* For many of these prompts you will probably not have a very detailed idea of how to go about achieving these outcomes.  That's okay.  Please write at a high level and make a note if you don't know something or would need to do more research.
{% endcapture %}
{% include problem_with_parts.html problem=problem %}
