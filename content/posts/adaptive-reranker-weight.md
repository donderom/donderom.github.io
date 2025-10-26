---
title: "Adaptive Reranker Weight"
date: 2022-10-15T11:38:30+02:00
draft: false
features: ["math"]
---

A method for improving the search results (e.g. for question answering retrieval) is known as *Retrieve & Re-rank*. The Retrieve & Re-rank pipeline consists of at least two components: the retriever and the re-ranker. The retriever is responsible for retrieving a number of documents given a search query. The relevancy of the retrieved documents can be further improved by the re-ranker, which allows retrieving more documents than required for the next pipeline stage.

Let's see an example of the output at each stage.

```py
query = 'what is re-ranking?' # Search query
n = 10 # Number of documents to retrieve

# Assume this function returns a list of documents with score
docs = retrieve(query=query, n=n)
scores = [doc['score'] for doc in docs]

# [0.9782995053726794,
#  0.9504939500760989,
#  0.8765814146070106,
#  0.8623934128019434,
#  0.842523354483268,
#  0.7736853461402741,
#  0.7713904667955406,
#  0.6740331628686816,
#  0.6378117863548827,
#  0.5634670917387724]

# And this function re-ranks the documents based on the
# relevancy to the query and returns the new scores
reranked = rerank(query=query, docs=docs)

# [0.8958727100108653,
#  0.9704265468563152,
#  0.8037856351531634,
#  0.4605732745735953,
#  0.9991750843646917,
#  0.7299899568668072,
#  0.6836966943663378,
#  0.6294383998509153,
#  0.5605524792499585,
#  0.41810846856511075]
```

Now, given two sets of scores per document, what is the total score? Calculating the average is one of the simplest approaches that could work.

```py
total_scores = [(s + r) / 2 for s, r in zip(scores, reranked)]

# [0.9370861076917724,
#  0.960460248466207,
#  0.840183524880087,
#  0.6614833436877694,
#  0.9208492194239799,
#  0.7518376515035406,
#  0.7275435805809392,
#  0.6517357813597985,
#  0.5991821328024206,
#  0.4907877801519416]
```

After combining both scores, the second document has the highest score now. If we want to assign different weights to the retriever and re-ranker scores, we can do so by multiplying the scores with their respective weight parameters: 

```py
# The weight values are completely arbitrary
retriever_weight = 1.2
reranker_weight = 1.5 # Boost reranker score a little bit

scoring = lambda s, r: (s * retriever_weight + r * reranker_weight) / 2
total_scores = [scoring(s, r) for s, r in zip(scores, reranked)]
```

Now the question is, what value should be used for the re-ranker weight? Looking at the total scores, it appears that the positions of the first and second documents have been swapped, with the fifth document occupying third place and so on. There is nothing wrong with keeping the re-ranker weight static. But what if, instead swapping the first two places, the first document ends up becoming the last one? The same weight might not be sufficient for so many search queries and numbers of documents.

{{< newthought >}}The intuition{{< /newthought >}} behind the idea is that the more spread out the documents are after re-ranking, the greater the weight. What makes it intuitive is that it's about the positional difference, rather than the score. Although the total score difference might be insignificant, moving several positions down from the top can be detrimental to the pipeline performance. It's not a big deal if we retrieve 10 documents and show/process them all. In case when we retrieve 100 documents to return top-5 of them, any document ranked higher than 5 will be filtered out.

If we view the document positions before and after re-ranking as observed and predicted values, we can use any deviation or error function that is used to measure the difference between values. One such function is *Root Mean Square Error*:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{i}-x_{i})^{2}}$$

Using the scores from above, it can be calculated as follows:

```py
import numpy as np

retrieved_pos = np.argsort(scores)
reranked_pos = np.argsort(reranked)

np.sqrt(np.mean(np.square(reranked_pos - retrieved_pos)))

# 2.23606797749979
```

Another example is *Mean Absolute Error*:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_{i}-x_{i}|$$

```py
np.mean(np.abs(reranked_pos - retrieved_pos))

# 1.6
```

Using different error functions will result in different values given the same magnitude of errors. In this case, the error of the MAE is smaller than the RMSE. We cannot compare the errors of different functions, but using the same magnitude of numbers allows us to choose any of them based on how large we want the resulting value to be.

*Adaptive reranker weight* is the measure between the retrieved and reranked positions. In other words, the adaptive weight is determined by how much the retrieved documents are rearranged by the re-ranker.

$$ARW = max(e, w), \qquad \forall e,w\in\mathbb{R}$$
, where $e$ is error value and $w$ is minimum weight.

With minimum weight set to zero, the adaptive reranker weight is just the error:

$$ARW = max(RMSE, 0) = RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{i}-x_{i})^{2}}$$

The adaptive reranker weight with MAE error and minimum weight of 1:

$$ARW = max(\frac{1}{n}\sum_{i=1}^{n}|y_{i}-x_{i}|, 1)$$

With scores changing per query, the re-ranker weight is adapted based on the difference between document positions. This approach can be further extended by incorporating the scores or by using different error functions. Logging the adaptive weight can also serve the purpose of monitoring the scoring deviations. The same idea can be applied down the pipeline to the components that calculate new sets of scores, thus affecting the document ranks.
