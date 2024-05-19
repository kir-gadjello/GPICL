# A replication of [General-Purpose In-Context Learning by Meta-Learning Transformers](https://arxiv.org/abs/2212.04458)

This is a self-contained replication of a paper by Kirsch et al. 2022 (Google), which proposes an intriguingly small and focused setup for training a Transformer with the capacity for meta-learning in context (in runtime). Unlike most prior research into inducing meta-learning, this method does not depend on bespoke handcrafted losses, optimizers or architectural elaborations and seeks to discover training regimes where generalization follows from virtuous distributional properties of the dataset. Thus I found it more relevant for the age of the Bitter Lesson and large scale pretraining. The final objective of work in this vein would be to create an initializing data mixture that enriches the NN with a meta-learning prior and minimizes tendencies for brittle memorization.

Concretely, in the replicated example the training set is permuted MNIST, while validation sets are unseen MNIST (to track in-distribution learning) and FashionMNIST (indicating generalization out of distribution). Simply put, the Transformer trained solely to classify permuted MNIST samples is provided with a sequence of sample-label pairs of FashionMNIST (a significantly dissimilar image disstribution); in the case where it has developed meta-learning prior, its accuracy improves above chance and scales with the number of in-context examples.

Running the experiment takes about 6 hours on Mac M1 Pro, on which I've been reading the paper.

To run:
`pip install -r requirements.txt`
`python gpicl_minimal_replication.py`