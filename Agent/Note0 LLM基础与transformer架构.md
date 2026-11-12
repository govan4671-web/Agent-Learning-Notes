## 前言
本篇内容来自[datawhalechina/hello-agents: 📚 《从零开始构建智能体》——从零开始的智能体原理与实践教程](https://github.com/datawhalechina/hello-agents)
已经过时的技术有**N-gram**，其核心思想是，一个句子出现的概率，等于该句子中每个词出现的条件概率的连乘  

## tranformer架构
1. 整体上遵循**编码器-译码器**架构：
   - **编码器 (Encoder)** ：任务是“**理解**”输入的整个句子。它会读取所有输入词元(这个概念会在3.2.2节介绍)，最终为每个词元生成一个富含上下文信息的向量表示。
   - **解码器 (Decoder)** ：任务是“**生成**”目标句子。它会参考自己已经生成的前文，并“咨询”编码器的理解结果，来生成下一个词。
   基本框架：
```python
		import torch
		import torch.nn as nn
		import math
		
		# --- 占位符模块，将在后续小节中实现 ---
		
		class PositionalEncoding(nn.Module):
		    """
		    位置编码模块
		    """
		    def forward(self, x):
		        pass
		
		class MultiHeadAttention(nn.Module):
		    """
		    多头注意力机制模块
		    """
		    def forward(self, query, key, value, mask):
		        pass
		
		class PositionWiseFeedForward(nn.Module):
		    """
		    位置前馈网络模块
		    """
		    def forward(self, x):
		        pass
		
		# --- 编码器核心层 ---
		
		class EncoderLayer(nn.Module):
		    def __init__(self, d_model, num_heads, d_ff, dropout):
		        super(EncoderLayer, self).__init__()
		        self.self_attn = MultiHeadAttention() # 待实现
		        self.feed_forward = PositionWiseFeedForward() # 待实现
		        self.norm1 = nn.LayerNorm(d_model)
		        self.norm2 = nn.LayerNorm(d_model)
		        self.dropout = nn.Dropout(dropout)
		
		    def forward(self, x, mask):
		        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
		        # 1. 多头自注意力
		        attn_output = self.self_attn(x, x, x, mask)
		        x = self.norm1(x + self.dropout(attn_output))
		
		        # 2. 前馈网络
		        ff_output = self.feed_forward(x)
		        x = self.norm2(x + self.dropout(ff_output))
		
		        return x
		
		# --- 解码器核心层 ---
		
		class DecoderLayer(nn.Module):
		    def __init__(self, d_model, num_heads, d_ff, dropout):
		        super(DecoderLayer, self).__init__()
		        self.self_attn = MultiHeadAttention() # 待实现
		        self.cross_attn = MultiHeadAttention() # 待实现
		        self.feed_forward = PositionWiseFeedForward() # 待实现
		        self.norm1 = nn.LayerNorm(d_model)
		        self.norm2 = nn.LayerNorm(d_model)
		        self.norm3 = nn.LayerNorm(d_model)
		        self.dropout = nn.Dropout(dropout)
		
		    def forward(self, x, encoder_output, src_mask, tgt_mask):
		        # 1. 掩码多头自注意力 (对自己)
		        attn_output = self.self_attn(x, x, x, tgt_mask)
		        x = self.norm1(x + self.dropout(attn_output))
		
		        # 2. 交叉注意力 (对编码器输出)
		        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
		        x = self.norm2(x + self.dropout(cross_attn_output))
		
		        # 3. 前馈网络
		        ff_output = self.feed_forward(x)
		        x = self.norm3(x + self.dropout(ff_output))
		
		        return x
```
2. 多头注意力
   核心是**自注意力机制**——它允许模型在处理序列中的每一个词时，都能兼顾句子中的所有其他词，并为这些词分配不同的“注意力权重”。权重越高的词，代表其与当前词的关联性越强，其信息也应该在当前词的表示中占据更大的比重。
   
   为了实现上述过程，自注意力机制为每个输入的词元向量引入了三个可学习的角色：
	
	- **查询 (Query, Q)**：代表当前词元，它正在主动地“查询”其他词元以获取信息。
	- **键 (Key, K)**：代表句子中可被查询的词元“标签”或“索引”。
	- **值 (Value, V)**：代表词元本身所携带的“内容”或“信息”
	
  具体过程（以查询北京的天气怎么样为例）：

|向量|含义|在这个例子中|
|---|---|---|
|**Query（查询）**|当前词发出的“搜索请求”|“怎么样”问：哪个词和我最相关？|
|**Key（键）**|每个词的“标签”，用于被搜索匹配|“北京”的标签偏向“地名”，“天气”的标签偏向“气象”|
|**Value（值）**|每个词的“内容”，匹配成功后被提取|匹配到“天气”后，提取它的语义信息|

简单来说，每个新词都在问“前面哪些词跟我最相关？”，通过打分找到最相关的词，然后重点参考它的信息来理解当前语境。
更具体地说，计算过程分三步：首先，“怎么样”生成自己的 Query 向量（一串数字，代表“我在找什么”）；然后，Query 与每个词的 Key 做点积（可以理解为“匹配度打分”——两组数字逐位相乘再加起来，结果越大说明越匹配），得到注意力权重；最后，用这些权重对所有词的 Value 加权求和——打分高的词贡献多，打分低的词贡献少，就像考试按权重算总分一样，最终合成出一个综合理解。

2. 前馈神经网络
   每个 Encoder 和 Decoder 层中，多头注意力子层之后都跟着一个**逐位置前馈网络(Position-wise Feed-Forward Network, FFN)**，如果说注意力层的作用是从整个序列中“动态地聚合”相关信息，那么前馈网络的作用从这些聚合后的信息中提取更高阶的特征。
3. 残差连接与层归一化
   在 Transformer 的每个编码器和解码器层中，所有子模块（如多头注意力和前馈网络）都被一个 `Add & Norm` 操作包裹。这个组合是为了保证 Transformer 能够稳定训练。
   这个操作由两个部分组成：

- **残差连接 (Add)**：该操作将子模块的输入 `x` 直接加到该子模块的输出 `Sublayer(x)` 上。这一结构解决了深度神经网络中的**梯度消失 (Vanishing Gradients)** 问题。在反向传播时，梯度可以绕过子模块直接向前传播，从而保证了即使网络层数很深，模型也能得到有效的训练。
  > 个人理解就是保留了完整的上文
- **层归一化 (Norm)**：该操作对单个样本的所有特征进行归一化，使其均值为0，方差为1。这解决了模型训练过程中的**内部协变量偏移 (Internal Covariate Shift)** 问题，使每一层的输入分布保持稳定，从而加速模型收敛并提高训练的稳定性。
5. 位置编码
   Transformer 的核心是自注意力机制，它通过计算序列中任意两个词元之间的关系来捕捉依赖，但它本身不包含任何关于词元顺序或位置的信息。对于自注意力来说，“agent learns” 和 “learns agent” 这两个序列是完全等价的，因为它只关心词元之间的关系，而忽略了它们的排列。为了解决这个问题，Transformer 引入了**位置编码 (Positional Encoding)** 。
   - 核心思想：为输入序列中的每一个词元嵌入向量，都额外加上一个能代表其绝对位置和相对位置信息的“位置向量”（通过固有数学公式得出，而非学习得到）
## Decoder-Only架构
由来：GPT基于transformer架构，**抛弃了编码器，只保留解码器**
Decoder-Only 架构的工作模式被称为**自回归 (Autoregressive)**，其详细过程：
1. 给模型一个起始文本（例如 “Datawhale Agent is”）。
2. 模型预测出下一个最有可能的词（例如 “a”）。
3. 模型将自己刚刚生成的词 “a” 添加到输入文本的末尾，形成新的输入（“Datawhale Agent is a”）。
4. 模型基于这个新输入，再次预测下一个词（例如 “powerful”）。
5. 不断重复这个过程，直到生成完整的句子或达到停止条件。
在训练阶段先对训练的数据采用**掩码自注意力**，将所有位于当前位置之后的词元对应的分数替换为一个非常大的负数。避免“偷看”后面的答案