# Underground-Engineering-Text-Mining-and-Visualization

### 本项实现了地下工程文本报告关键信息的提取和可视化，同时还建立了网页端一站式可视化平台。

Text-Mining，包括使用TF-IDF进行关键词提取；使用TextRank4ZH进行关键词和摘要提取，使用BK-TextRank进行关键词和摘要提取。

Visualization，包括使用WordCloud构建词云，进行关键词中心性相似性分析并绘制网络图，进行LDA主题建模并可视化。

为了获得准确的分词结果，一个良好的分词模型是必须的，我们整理了地下工程语料，结合通用语料(PKU人民日报)训练了BERT-BiLSTM-CRF分词模型，实际上还训练了BERT-CRF和BiLSTM-CRF模型，但在BERT-BiLSTM-CRF中取得了最好的效果。分词模型的构建参考[WordSeg](https://github.com/hemingkx/WordSeg),详情可自行研究，在此表示感谢。

我们的目的是处理各种文本报告，可能涉及到doc/docx、可编辑PDF、扫描的图像PDF以及图片等常见的格式，在网页端可以上传各种格式而不必前处理。

目前项目还不是很完善，我们也在继续更新，欢迎 Star 和 issue。
