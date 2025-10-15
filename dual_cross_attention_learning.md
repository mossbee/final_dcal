# Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification

## Introduction

In this work, we investigate how to extend self-attention modules to better learn subtle feature embeddings for recognizing fine-grained objects, e.g., different bird species or person identities. 

We adopt a different way to incorporate local information based on vision Transformer. To this end, we propose global-local cross-attention (GLCA) to enhance the interactions between global images and local high-response regions. Specifically, we compute the cross-attention between a selected subset of query vectors and the entire set of key-value vectors. By coordinating with self-attention learning, GLCA can help reinforce the spatial-wise discriminative clues to recognize fine-grained objects.

We employ a pair-wise learning scheme to establish the interactions between image pairs. Different from optimizing the feature distance, we propose pair-wise cross-attention (PWCA) to regularize the attention learning of an image by treating another image as distractor. Specifically, we compute the cross-attention between query of an image and combined key-value from both images. By introducing confusion in key and value vectors, the attention scores are diffused to another image so that the difficulty of the attention learning of the current image increases. Such regularization allows the network to discover more discriminative regions and alleviate overfitting to sample-specific features. It is noted that PWCA is only used for training and thus does not introduce extra computation cost during inference. 

## Proposed Approach

### Revisit Self-Attention

``Attention is all you need`` originally proposes the self-attention mechanism to address NLP tasks by calculating the correlation between each word and all the other words in the sentence. ``Vision Transformer`` inherits the idea by taking each patch in the image / feature map as a word for general image classification. In general, a self-attention function can be depicted as mapping a query vector and a set of key and value vectors to an output. The output is computed as a weighted sum of value vectors, where the weight assigned to each value is computed by a scaled inner product of the query with the corresponding key. Specifically, a query $q \in \mathbb{R}^{1\times d}$ is first matched against $N$ key vectors ($K=[k_1;k_2;\cdots ;k_N]$, where each $k_i \in \mathbb{R}^{1\times d}$) using inner product. The products are then scaled and normalized by a softmax function to obtain $N$ attention weights. The final output is the weighted sum of $N$ value vectors ($V=[v_1;v_2;\cdots ;v_N]$, where each $v_i \in \mathbb{R}^{1\times d}$). By packing $N$ query vector into a matrix $Q=[q_1;q_2;\cdots ;q_N]$, the output matrix of self-attention (SA) can be represented as:

$$
    f_{\text{SA}}(Q,K,V)\ =\text{softmax}(\frac{QK^T}{\sqrt{d}})V = SV
\tag{eq:sa}
$$

where $\frac{1}{\sqrt{d}}$ is a scaling factor.  Query, key and value matrices are computed from the same input embedding $X \in \mathbb{R}^{N\times D}$ with different linear transformations: $Q=XW_Q$, $K=XW_K$, $V = XW_V$, respectively. $S \in \mathbb{R}^{N\times N}$ denotes the attention weight matrix. 

To jointly attend to information from different representation subspaces at different positions, multi-head self-attention (MSA) is defined by considering multiple attention heads. The process of MSA can be computed as linear transformation on the concatenations of self-attention blocks with subembeddings. To encode positional information, fixed / learnable position embeddings are added to patch embeddings and then fed to the network. To predict the class, an extra class embedding $\hat{\texttt{CLS}} \in \mathbb{R}^{1\times d}$ is prepended to the input embedding $X$ throughout the network, and finally projected with a linear classifer layer for prediction. Thus, the input embeddings as well as query, key and value matrices become $(N+1)\times d$ and the self-attention function (Eq. [eq:sa]) allows to spread information between patch and class embeddings. 

Based on self-attention, a Transformer encoder block can be constructed by an MSA layer and a feed forward network (FFN). FFN consists of two linear transformation with a GELU activation. Layer normalization (LN) is put prior to each MSA and FFN layer and residual connections are used for both layers.

### Global-Local Cross-Attention 

Self-attention treats each query equally to compute global attention scores according to Eq. [eq:sa]. In other words, each local position of image is interacted with all the positions in the same manner. For recognizing fine-grained objects, we expect to mine discriminative local information to facilitate the learning of subtle features. To this end, we propose global-local cross-attention to emphasize the interaction between global images and local high-response regions. 

First, we use **attention rollout** to calculate the accumulated attention scores for the $i$-th block. Attention rollout is a method to track how information propagates from input tokens to higher layers by recursively computing attention across all previous layers. This is necessary because in deeper layers, embeddings become increasingly mixed and contextualized, making raw attention weights unreliable for identifying important input regions.

The attention rollout for the $i$-th block is computed as:

$$
    \hat{S}_i = \bar{S}_i \otimes \bar{S}_{i-1} \otimes \cdots \otimes \bar{S}_1
    \tag{eq:rollout}
$$

where $\bar{S}_l = 0.5S_l + 0.5E$ represents the re-normalized attention weights at layer $l$, with $S_l$ being the raw attention matrix and $E$ being the identity matrix. This formulation accounts for residual connections in the Transformer architecture, where each layer's output is added to its input. The $\otimes$ operation denotes matrix multiplication.

The key insight is that by multiplying attention matrices from all previous layers, we can track how much each position in the current layer "attends to" the original input tokens, rather than just the immediate previous layer. This gives us a more accurate picture of which input regions are most important for the model's decisions.

Then, we use the aggregated attention map to mine the high-response regions. According to Eq. [eq:rollout], the first row of $\hat{S}_i = [\hat{s}_{i,j}]_{(N+1)\times (N+1)}$ means the accumulated weights of class embedding $\hat{\texttt{CLS}}$. We select top $R$ query vectors from $Q_i$ that correspond to the top $R$ highest responses in the accumulated weights of $\hat{\texttt{CLS}}$ to construct a new query matrix $Q^l$, representing the most attentive local embeddings. Finally, we compute the cross attention between the selected local query and the global set of key-value pairs as below.

$$
f_{\text{GLCA}}(Q^l,K^g,V^g)=\text{softmax}(\frac{Q^l{K^g}^T}{\sqrt{d}})V^g 
\tag{eq:glca}
$$

In self-attention (Eq. [eq:sa]), all the query vectors will be interacted with the key-value vectors. In our GLCA (Eq. [eq:glca]), only a subset of query vectors will be interacted with the key-value vectors. We observe that GLCA can help reinforce the spatial-wise discriminative clues to promote recognition of fine-grained classes. Another possible choice is to compute the self-attention between local query $Q^l$ and local key-value vectors ($K^l$, $V^l$). However, through establishing the interaction between local query and global key-value vectors, we can relate the high-response regions with not only themselves but also with other context outside of them. Figure [figure:overview] (a) illustrates the proposed global-local cross-attention and we use $M=1$ GLCA block in our method. 

### Pair-Wise Cross-Attention

The scale of fine-grained recognition datasets is usually not as large as that of general image classification, e.g., ImageNet  contains over 1 million images of 1,000 classes while CUB  contains only 5,994 images of 200 classes for training. Moreover, smaller visual differences between classes exist in FGVC and Re-ID compared to large-scale classification tasks. Fewer samples per class may lead to network overfitting to sample-specific features for distinguishing visually confusing classes in order to minimize the training error. 

To alleviate the problem, we propose pair-wise cross attention to establish the interactions between image pairs. PWCA can be viewed as a novel regularization method to regularize the attention learning. Specifically, we randomly sample two images ($I_1$, $I_2$) from the same training set to construct the pair. The query, key and value vectors are separately computed for both images of a pair. For training $I_1$, we concatenate the key and value matrices of both images, and then compute the attention between the query of the target image and the combined key-value pairs as follows:

$$
    f_{\text{PWCA}}(Q_1,K_c,V_c) =\text{softmax}(\frac{Q_1 K_c^T}{\sqrt{d}})V_c
\tag{eq:pwca}
$$

where $K_c=[K_1;K_2] \in \mathbb{R}^{(2N+2)\times d}$ and $V_c=[V_1;V_2] \in \mathbb{R}^{(2N+2)\times d}$. For a specific query from $I_1$, we compute $N+1$ self-attention scores within itself and $N+1$ cross-attention scores with $I_2$ according to Eq. [eq:pwca]. All the $2N+2$ attention scores are normalized by the softmax function together and thereby contaminated attention scores for the target image $I_1$ are learned. 

Optimizing this noisy attention output increases the difficulty of network training and reduces the overfitting to sample-specific features. In the proposed pair-wise cross-attention, we use $T=12$ PWCA blocks. Note that PWCA is only used for training and will be removed for inference without consuming extra computation cost.  

## Experiments

### Experimental Setting

**Datasets.**

We conduct extensive experiments on two fine-grained recognition tasks: fine-grained visual categorization (FGVC) and object re-identification (Re-ID). For FGVC, we use three standard benchmarks for evaluations: CUB-200-2011 , Stanford Cars , FGVC-Aircraft .

For Re-ID, we use four standard benchmarks: Market1501 , DukeMTMC-ReID , MSMT17  for Person Re-ID and VeRi-776  for Vehicle Re-ID. In all experiments, we use the official train and validation splits for evaluation.

**Baselines.** 

We use DeiT and ViT as our self-attention baselines. In detail, ViT backbones are pre-trained on ImageNet-21k  and DeiT backbones are pre-trained on ImageNet-1k . We use multiple architectures of DeiT-T/16, DeiT-S/16, DeiT-B/16, ViT-B/16, R50-ViT-B/16 with $L=12$ SA blocks for evaluation.

Attention map is generated using attention rollout.

**Implementation Details.** 

We coordinate the proposed two types of cross-attention with self-attention in the form of multi-task learning. We build $L=12$ SA blocks, $M=1$ GLCA blocks and $T=12$ PWCA blocks as the overall architecture for training. The PWCA branch shares weights with the SA branch while GLCA does not share weights with SA.


To balance the losses from different attention mechanisms, we adopt the uncertainty-based loss weighting strategy proposed by Kendall et al. (2018). The total loss is computed as:

$$
L_{\text{total}} = \frac{1}{2}(\frac{1}{e^{w_1}}L_{\text{SA}} + \frac{1}{e^{w_2}}L_{\text{GLCA}} + \frac{1}{e^{w_3}}L_{\text{PWCA}} + w_1 + w_2 + w_3)
$$

where $w_1$, $w_2$, and $w_3$ are learnable parameters that automatically balance the three attention mechanisms. This dynamic loss weighting approach avoids exhausting manual hyper-parameter search and allows the network to automatically determine which attention mechanism needs more focus during training. The PWCA branch has the same ground truth target as the SA branch since we treat another image as distractor.

For FGVC, we resize the original image into 550 $ \times $ 550 and randomly crop to 448 $ \times $ 448 for training. The sequence length of input embeddings for self-attention baseline is $28\times 28=784$. We select input embeddings with top $R=10\%$ highest attention responses as local queries. We apply stochastic depth and use Adam optimizer with weight decay of 0.05 for training. The learning rate is initialized as ${\rm lr}_{scaled}=\frac{5e-4}{512}\times batchsize$ and decayed with a cosine policy. We train the network for 100 epochs with batch size of 16 using the standard cross-entropy loss. 

For Re-ID, we resize the image into 256 $ \times $ 128 for pedestrian datasets, and 256 $ \times $ 256 for vehicle datasets. We select input embeddings with top $R=30\%$ highest attention responses as local queries. We use SGD optimizer with a momentum of 0.9 and a weight decay of 1e-4. The batch size is set to 64 with 4 images per ID. The learning rate is initialized as 0.008 and decayed with a cosine policy. We train the network for 120 epochs using the cross-entropy and triplet losses.

All of our experiments are conducted on PyTorch with Nvidia Tesla V100 GPUs. Our method costs 3.8 hours with DeiT-Tiny backbone for training using 4 GPUs on CUB, and 9.5 hours with ViT-Base for training using 1 GPU on MSMT17. During inference, we remove all the PWCA modules and only use the SA and GLCA modules. We add class probabilities output by classifiers of SA and GLCA for prediction for FGVC, and concat two final class tokens of SA and GLCA for prediction for Re-ID. A single image with the same input size as training is used for test. 

| Method        | Backbone    | Accuracy (%) |      |      |
|---------------|-------------|--------------|------|------|
|               |             | CUB          | CAR  | AIR  |
| Baseline      | DeiT-Tiny   | 82.1         | 87.2 | 84.7 |
| Baseline + DCAL | DeiT-Tiny   | 84.6         | 89.4 | 87.4 |
| Baseline      | DeiT-Small  | 85.8         | 90.7 | 88.1 |
| Baseline + DCAL | DeiT-Small  | 87.6         | 92.3 | 90.0 |
| Baseline      | DeiT-Base   | 88.0         | 92.9 | 90.3 |
| Baseline + DCAL | DeiT-Base   | 88.8         | 93.8 | 92.6 |
| Baseline      | ViT-Base    | 90.8         | 92.5 | 90.0 |
| Baseline + DCAL | ViT-Base    | 91.4         | 93.4 | 91.5 |
| Baseline      | R50-ViT-Base | 91.3         | 94.0 | 92.4 |
| Baseline + DCAL | R50-ViT-Base | 92.0         | 95.3 | 93.3 |

Table 1: Performance comparisons in terms of top-1 accuracy on three standard FGVC benchmarks: CUB-200-2011, Stanford Cars and FGVC-Aircraft. (fine-grained sota compare)

| Method                       | VeRi-776 |      | MSMT17 |      | Market1501 |      | DukeMTMC |      |
|------------------------------|----------|------|--------|------|------------|------|----------|------|
|                              | mAP (%)  | R1 (%) | mAP (%) | R1 (%) | mAP (%)    | R1 (%) | mAP (%)  | R1 (%) |
| DeiT-Tiny                    | 71.3     | 94.3 | 42.1   | 63.9 | 77.9       | 90.3 | 69.5     | 82.9 |
| DeiT-Tiny + DCAL (Ours)      | 74.1     | 94.7 | 44.9   | 68.2 | 79.8       | 91.8 | 71.7     | 84.9 |
| DeiT-Small                   | 76.7     | 95.5 | 53.3   | 75.0 | 84.3       | 93.7 | 75.7     | 87.6 |
| DeiT-Small + DCAL (Ours)     | 78.1     | 95.9 | 55.1   | 77.3 | 85.3       | 94.0 | 77.4     | 87.9 |
| DeiT-Base                    | 78.3     | 95.9 | 60.5   | 81.6 | 86.6       | 94.4 | 79.1     | 88.7 |
| DeiT-Base + DCAL (Ours)      | 80.0     | 96.5 | 62.3   | 83.1 | 87.2       | 94.5 | 80.2     | 89.6 |
| ViT-Base                     | 78.1     | 96.0 | 61.6   | 81.4 | 87.1       | 94.3 | 78.9     | 89.4 |
| ViT-Base + DCAL (Ours)       | 80.2     | 96.9 | 64.0   | 83.1 | 87.5       | 94.7 | 80.1     | 89.0 |

Table 2: Performance comparisons on four Re-ID benchmarks: VeRi-776, MSMT17, Market1501, DukeMTMC. The input size is 256x128 for pedestrian datasets and 256x256 for vehicle datasets. * means results without side information for fair comparison. (reid sota compare)

### More Transformer Baselines
We conduct two more experiments on CaiT and Swin Transformer. CaiT-XS24 obtains 88.5\% while our method obtains 89.7\% top-1 accuracy on CUB. Swin-T obtains 84.9\% while our method obtains 85.8\% top-1 accuracy on CUB. For Re-ID on MSMT, Swin-T achieves 55.7\% while we achieve 56.7\% mAP. As locality has been incorporated by windows in Swin Transformer, we only apply PWCA into it. 